from causal_model.Treatment import loadTreatment
import pandas as pd
import os
import pickle
from sklearn.preprocessing import normalize
import numpy as np
from DDI_module.genKGembd import getEmbd
from multiprocessing import Pool
from scipy.spatial.distance import cdist

def drop_unseen_nodes(orign_info, cmp_col_name, reserved_ids_set):
    if reserved_ids_set != set(orign_info[cmp_col_name].values):
        pd_rating_ids = pd.DataFrame(list(reserved_ids_set), columns=["id_graph"])
        data_info = orign_info.merge(pd_rating_ids, left_on=cmp_col_name, right_on='id_graph', how='outer')
        data_info = data_info.dropna(subset=[cmp_col_name, 'id_graph'])
        data_info = data_info.drop(columns=["id_graph"])
        data_info = data_info.reset_index(drop=True)
        return data_info
    else:
        orign_info = orign_info.reset_index(drop=True)
        return orign_info

def get_CF_single(params):
    """ single process for getting CF edges """
    node_pairs, patient_simi_mat, patient_node_nns, patient_thresh, \
    drug_simi_mat, drug_node_nns, drug_thresh,\
    T_f, node_pairs, verbose = params

    T_cf = np.zeros(T_f.shape)
    node_pairs_cf = []
    edges_cf_t0 = []
    edges_cf_t1 = []
    c = 0
    for a, b in node_pairs:
        # for each node pair (a,b), find the nearest node pair (c,d)
        nns_a = patient_node_nns[a]
        nns_b = drug_node_nns[b]
        i, j = 0, 0
        while i < len(nns_a)-1 and j < len(nns_b)-1:
            if patient_simi_mat[a, nns_a[i]] > patient_thresh or\
                    drug_simi_mat[b, nns_b[j]] > drug_thresh:
                T_cf[a, b] = T_f[a, b]
                node_pairs_cf.append([a, b])
                break
            if T_f[nns_a[i], nns_b[j]] != T_f[a, b]:
                T_cf[a, b] = 1 - T_f[a, b] # T_f[nns_a[i], nns_b[j]] when treatment not binary
                node_pairs_cf.append([nns_a[i], nns_b[j]])
                if T_cf[a, b] == 0:
                    edges_cf_t0.append([nns_a[i], nns_b[j]])
                else:
                    edges_cf_t1.append([nns_a[i], nns_b[j]])
                break
            if patient_simi_mat[a, nns_a[i+1]] < drug_simi_mat[b, nns_b[j+1]]:
                i += 1
            else:
                j += 1
        c += 1
        if verbose and c % 20000 == 0:
            print(f'{c} / {len(node_pairs)} done')
    node_pairs_cf = np.asarray(node_pairs_cf)
    edges_cf_t0 = np.asarray(edges_cf_t0)
    edges_cf_t1 = np.asarray(edges_cf_t1)
    return T_cf, node_pairs_cf, edges_cf_t0, edges_cf_t1

def get_node_nns(embs,thresh,dist):
    if dist == 'cosine':
        # cosine similarity (flipped to use as a distance measure)
        embs = normalize(embs, norm='l1', axis=1)
        simi_mat = embs @ embs.T
        simi_mat = 1 - simi_mat

    elif dist == 'euclidean':
        # Euclidean distance
        simi_mat = cdist(embs, embs, 'euclidean')

    thresh = np.percentile(simi_mat, thresh)
    # give selfloop largest distance
    np.fill_diagonal(simi_mat, np.max(simi_mat)+1)
    # nearest neighbor nodes index for each node
    node_nns = np.argsort(simi_mat, axis=1)
    return simi_mat, node_nns, thresh


def get_CF(node_pairs, patient_embs, drug_embs, T_f, dist='euclidean', patient_thresh=100, drug_thresh=20, n_workers=20):
    patient_simi_mat, patient_node_nns, patient_thresh = get_node_nns(patient_embs, patient_thresh, dist)
    drug_simi_mat, drug_node_nns, drug_thresh = get_node_nns(drug_embs, drug_thresh, dist)
    # find nearest CF node-pair for each node-pair
    print('This step may be slow, please adjust args.n_workers according to your machine')
    pool = Pool(n_workers)
    batches = np.array_split(node_pairs, n_workers)
    results = pool.map(get_CF_single,
                       [(node_pairs, patient_simi_mat, patient_node_nns, patient_thresh,
                         drug_simi_mat, drug_node_nns, drug_thresh,
                         T_f, np_batch, True) for np_batch in batches])
    results = list(zip(*results))
    T_cf = np.add.reduce(results[0])
    node_pairs_cf = np.concatenate(results[1])
    edges_cf_t0 = np.concatenate(results[2])
    edges_cf_t1 = np.concatenate(results[3])
    return T_cf, node_pairs_cf, edges_cf_t0, edges_cf_t1

def genCFLinks(dir, K, max_patient_neighbor, max_drug_neighbor):
    T = loadTreatment(dir, K)

    #load f_adj
    all_rating_info = pd.read_excel(os.path.join(dir, 'rating_total_compound_only1.xlsx'))

    # load all user feats
    UserfulUser = pd.read_excel(os.path.join(dir, 'userful_userId.xlsx')).values.tolist()
    UserfulUser = [id[0] for id in UserfulUser]
    all_rating_info = all_rating_info[all_rating_info['userId'].isin(UserfulUser)]
    filepath = os.path.join(dir, 'feat_total_selected_withMU.xlsx')
    user_info = pd.read_excel(filepath)
    user_info = drop_unseen_nodes(orign_info=user_info,
                                  cmp_col_name="userId",
                                  reserved_ids_set=set(all_rating_info["userId"].values))
    feat_col_list = user_info.columns[2:]
    feat_np_list = []
    for i in range(len(feat_col_list)):
        np_feat = user_info[feat_col_list[i]].values.astype(np.float32).reshape((user_info.shape[0], 1))
        feat_np_list.append(np_feat)
    user_features = np.concatenate(feat_np_list, axis=1)
    where_are_nan = np.isnan(user_features)
    where_are_inf = np.isinf(user_features)
    user_features[where_are_nan] = 0
    user_features[where_are_inf] = 0
    user_features = normalize(user_features, axis=0, norm='l2')

    #load item_features
    file_path = os.path.join(dir, 'DBtotal_compound_list_1v1.xlsx')
    item_info = pd.read_excel(file_path)
    item_info = drop_unseen_nodes(orign_info=item_info,
                                             cmp_col_name="compoundIndex",
                                             reserved_ids_set=set(all_rating_info["itemId"].values))
    drugID = item_info['compoundID'].values.tolist()
    item_features = np.array(getEmbd(drugID))

    #node_pairs
    global_user_id_map = {ele: i for i, ele in enumerate(user_info['userId'])}
    global_item_id_map = {ele: i for i, ele in enumerate(item_info['compoundIndex'])}
    node_pairs = np.array([[global_user_id_map[ele] for ele in all_rating_info["userId"]],
                    [global_item_id_map[ele] for ele in all_rating_info["itemId"]]],
                             dtype=np.int64).T

    #gen T_cf, node_pair_cf
    T_cf, node_pairs_cf, edges_cf_t0, edges_cf_t1 = get_CF(
        node_pairs, user_features, item_features, T, dist='euclidean', patient_thresh=max_patient_neighbor,
        drug_thresh=max_drug_neighbor, n_workers=20)

    adj_cf = np.zeros([len(global_user_id_map.keys()),len(global_item_id_map.keys())])
    for a,b in node_pairs_cf:
        adj_cf[a][b] = 1

    pickle.dump(T_cf, open(dir + '/Treatment_CF_'+str(K)+'.pkl', 'wb'))
    pickle.dump(node_pairs_cf, open(dir + '/node_pairs_cf_' + str(K) + '_'+str(max_patient_neighbor)+'_'+ str(max_drug_neighbor)+ '.pkl', 'wb'))
    pickle.dump(adj_cf, open(
        dir + '/adj_cf_' + str(K) + '_' + str(max_patient_neighbor) + '_' + str(max_drug_neighbor) + '.pkl',
        'wb'))
    print("saved T_cf, node_pairs_cf, adj_cf!")


def  loadCFLinks(dir, K, max_patient_neighbor, max_drug_neighbor):
    T_cf, node_pairs_cf = [], []
    if os.path.exists(dir):
        T_cf = pickle.load(open(dir+'/Treatment_CF_'+str(K)+'.pkl', 'rb'))
        node_pairs_cf = pickle.load(open(
            dir + '/node_pairs_cf_' + str(K) + '_' + str(max_patient_neighbor) + '_' + str(max_drug_neighbor) + '.pkl',
            'rb'))
        adj_cf = pickle.load(open(
            dir + '/adj_cf_' + str(K) + '_' + str(max_patient_neighbor) + '_' + str(max_drug_neighbor) + '.pkl',
            'rb'))
        print('loaded cached T_cf, node_pairs_cf, adj_cf files!')
    return T_cf, node_pairs_cf, adj_cf



if __name__ == '__main__':
    dir = "../data"
    K = 15
    max_patient_neighbor = 100
    max_drug_neighbor = 20
    genCFLinks(dir, K, max_patient_neighbor, max_drug_neighbor)
    T_cf, node_pairs_cf, adj_cf = loadCFLinks(dir, K, max_patient_neighbor, max_drug_neighbor)
    print(T_cf.sum() / (T_cf.shape[0] * T_cf.shape[1]))
    print(adj_cf.sum())