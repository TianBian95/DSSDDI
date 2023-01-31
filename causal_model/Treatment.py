import torch as th
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import normalize

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

def getDDIrelation(dir, drugID):
    DBID2NodeIndex_dic = {ele: i for i, ele in enumerate(drugID)}
    compd_relations = pd.read_excel(
        dir + '/compound_relations_syn&ant_1v1.xlsx')
    compd_relations = compd_relations[compd_relations.iloc[:, 0].isin(drugID)]
    compd_relations = compd_relations[compd_relations.iloc[:, 1].isin(drugID)]
    edge_row = compd_relations.iloc[:, 0].values.tolist()
    edge_col = compd_relations.iloc[:, 1].values.tolist()
    edge_type = compd_relations.iloc[:, 2]
    src = np.array([DBID2NodeIndex_dic[edgeID] for edgeID in edge_row])
    dst = np.array([DBID2NodeIndex_dic[edgeID] for edgeID in edge_col])
    labels = list(edge_type)
    return src, dst, labels


def genTreatment(dir, K, feat_norm='l2'):
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
    user_features = normalize(user_features, axis=0, norm=feat_norm)

    # cluster users
    clustering = KMeans(n_clusters=K, random_state=0).fit(user_features)
    clustering_labels = clustering.labels_

    #load drugID_list
    file_path = os.path.join(dir, 'DBtotal_compound_list_1v1.xlsx')
    item_info = pd.read_excel(file_path)
    item_info = drop_unseen_nodes(orign_info=item_info,
                                             cmp_col_name="compoundIndex",
                                             reserved_ids_set=set(all_rating_info["itemId"].values))
    drugID_list = item_info['compoundID'].values.tolist()

    #generate rating_pair
    global_user_id_map = {ele: i for i, ele in enumerate(user_info['userId'])}
    global_item_id_map = {ele: i for i, ele in enumerate(item_info['compoundIndex'])}
    userId = [global_user_id_map[ele] for ele in all_rating_info["userId"]]
    itemId = [global_item_id_map[ele] for ele in all_rating_info["itemId"]]

    #generate T
    T = th.zeros([len(user_features), len(drugID_list)])
    num_edge = len(userId)
    for edge_index in range(num_edge):
        # patients in same cluster
        same_cluster_users = np.where(clustering_labels == clustering_labels[userId[edge_index]])[0].tolist()
        # drugs has same effect
        src, dst, labels = getDDIrelation(dir, drugID_list)
        syn_drugs = []
        for i in range(len(src)):
            if itemId[edge_index] == src[i] and labels[i] == 1:
                syn_drugs.append(dst[i])
            if itemId[edge_index] == dst[i] and labels[i] == 1:
                syn_drugs.append(src[i])
        same_cluster_users.append(userId[edge_index])
        syn_drugs.append(itemId[edge_index])
        for user_index in same_cluster_users:
            for drug_index in syn_drugs:
                T[user_index][drug_index] = 1

    print(T.sum() / (T.shape[0] * T.shape[1]))
    pickle.dump(T, open(dir+'/Treatment_'+str(K)+'.pkl', 'wb'))
    print("saved T!")

def loadTreatment(dir, K):
    if os.path.exists(dir):
        T_file = dir+'/Treatment_'+str(K)+'.pkl'
        T = pickle.load(open(T_file, 'rb'))
        print('loaded cached T files!')
    return T

if __name__ == '__main__':
    dir = "../data"
    K = 15
    genTreatment(dir, K)
    T = loadTreatment(dir, K)