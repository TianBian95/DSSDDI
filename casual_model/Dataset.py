import numpy as np
import scipy.sparse as sp
import os
import pandas as pd
import torch as th
from DDI_module.genKGembd import getEmbd
from sklearn.preprocessing import normalize
from casual_model.Treatment import loadTreatment
from casual_model.CounterfactualLinks import loadCFLinks
import random

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

def generate_pair_value(global_user_id_map,global_item_id_map,rating_info):
    rating_pairs = (np.array([global_user_id_map[ele] for ele in rating_info["userId"]],
                             dtype=np.int64),
                    np.array([global_item_id_map[ele] for ele in rating_info["itemId"]],
                             dtype=np.int64))
    rating_values = rating_info["rating"].values.astype(np.float32)
    return rating_pairs, rating_values

def generate_label(rating_info, user_id, global_item_id_map):
    y = []
    num_drug = []
    drug_cnt = np.zeros((86))
    for uid in user_id:
        itemId = rating_info[rating_info['userId']==uid]['itemId'].values.tolist()
        itemId = [global_item_id_map[ele] for ele in itemId]
        for id in itemId:
            drug_cnt[id] += 1
        y.append(itemId)
        num_drug.append(len(itemId))
    return y, num_drug, drug_cnt


def loadData(dir, feat_norm, test_ratio, valid_ratio, K, max_patient_neighbor, max_drug_neighbor, num_np):
    all_rating_info = pd.read_excel(os.path.join(dir, 'rating_total_compound_only1.xlsx'))
    UserfulUser = pd.read_excel(os.path.join(dir, 'userful_userId.xlsx')).values.tolist()
    UserfulUser = [id[0] for id in UserfulUser]
    all_rating_info = all_rating_info[all_rating_info['userId'].isin(UserfulUser)]

    filepath = os.path.join(dir, 'feat_total_selected_withMU.xlsx')
    user_info = pd.read_excel(filepath)
    file_path = os.path.join(dir, 'DBtotal_compound_list_1v1.xlsx')
    item_info = pd.read_excel(file_path)

    user_info = drop_unseen_nodes(orign_info=user_info,
                                             cmp_col_name="userId",
                                             reserved_ids_set=set(all_rating_info["userId"].values))
    item_info = drop_unseen_nodes(orign_info=item_info,
                                             cmp_col_name="compoundIndex",
                                             reserved_ids_set=set(all_rating_info["itemId"].values))

    global_user_id_map = {ele: i for i, ele in enumerate(user_info['userId'])}
    global_item_id_map = {ele: i for i, ele in enumerate(item_info['compoundIndex'])}

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
    user_features = th.FloatTensor(user_features)


    drugID = item_info['compoundID'].values.tolist()
    item_features = np.array(getEmbd(drugID))
    item_features = th.FloatTensor(item_features)

    info_line = "Feature dim: "
    info_line += "\nuser: {}".format(user_features.shape)
    info_line += "\nitem: {}".format(item_features.shape)
    # print(info_line)

    num_test = int(np.ceil(num_user * test_ratio))
    shuffled_idx = list(global_user_id_map.keys())
    random.shuffle(shuffled_idx)
    test_id = shuffled_idx[: num_test]
    all_train_id = shuffled_idx[num_test:]
    test_rating_info = all_rating_info[all_rating_info['userId'].isin(test_id)]
    all_train_rating_info = all_rating_info[all_rating_info['userId'].isin(all_train_id)]
    num_valid = int(np.ceil(num_user * valid_ratio))
    random.shuffle(all_train_id)
    valid_id = all_train_id[:num_valid]
    train_id = all_train_id[num_valid:]
    valid_rating_info = all_rating_info[all_rating_info['userId'].isin(valid_id)]
    train_rating_info = all_rating_info[all_rating_info['userId'].isin(train_id)]

    train_mask = th.tensor([global_user_id_map[ele] for ele in train_id])
    valid_mask = th.tensor([global_user_id_map[ele] for ele in valid_id])
    test_mask = th.tensor([global_user_id_map[ele] for ele in test_id])

    # print("All rating pairs : {}".format(all_rating_info.shape[0]))
    # print("\tAll train rating pairs : {}".format(all_train_rating_info.shape[0]))
    # print("\t\tTrain rating pairs : {}".format(train_rating_info.shape[0]))
    # print("\t\tValid rating pairs : {}".format(valid_rating_info.shape[0]))
    # print("\tTest rating pairs  : {}".format(test_rating_info.shape[0]))
    all_train_rating_pairs, all_train_rating_values = generate_pair_value(global_user_id_map,global_item_id_map,all_train_rating_info)
    train_rating_pairs, train_rating_values = generate_pair_value(global_user_id_map,global_item_id_map,train_rating_info)

    train_trainMat = sp.csc_matrix((train_rating_values, train_rating_pairs), shape=(num_user, num_item))
    valid_trainMat = sp.csc_matrix((train_rating_values, train_rating_pairs), shape=(num_user, num_item))
    test_trainMat = sp.csc_matrix((all_train_rating_values, all_train_rating_pairs), shape=(num_user, num_item))

    tmp_trainMat = train_trainMat.todok()
    pos_train_rating_pairs=th.tensor(train_rating_pairs)
    length = pos_train_rating_pairs.shape[1]
    train_userId = pos_train_rating_pairs[0]
    train_pos_itemId = pos_train_rating_pairs[1]
    train_neg_itemId = np.random.randint(low=0, high=num_item, size=length)
    for i in range(length):
        uid = train_userId[i]
        iid = train_neg_itemId[i]
        if (uid, iid) in tmp_trainMat:
            while (uid, iid) in tmp_trainMat:
                iid = np.random.randint(low=0, high=num_item)
            train_neg_itemId[i] = iid
    train_neg_itemId = th.tensor(train_neg_itemId)
    neg_train_rating_pairs = th.cat([train_userId.reshape(1,-1), train_neg_itemId.reshape(1,-1)],dim=0)
    total_train_rating_pairs = th.cat([pos_train_rating_pairs,neg_train_rating_pairs],dim=-1).numpy().T

    #construct pos and neg Treatment
    T_f = loadTreatment(dir, K)
    T_cf, node_pairs_cf, adj_cf = loadCFLinks(dir, K, max_patient_neighbor, max_drug_neighbor)

    labels_f = th.cat([th.ones(train_pos_itemId.shape[0]), th.zeros(train_neg_itemId.shape[0])])

    train_T_f, train_T_cf, labels_cf = [], [], []
    for a,b in total_train_rating_pairs:
        train_T_f.append(T_f[a][b])
        train_T_cf.append(T_cf[a][b])
        labels_cf.append(adj_cf[a][b])
    train_T_f, train_T_cf, labels_cf = th.tensor(train_T_f), th.tensor(train_T_cf), th.tensor(labels_cf)
    # print(train_T_f.sum().item(), train_T_cf.sum().item(), labels_cf.sum().item())

    nodepairs_f = pos_train_rating_pairs.numpy().T
    nodepairs_cf = []
    for a,b in node_pairs_cf:
        if a in train_mask:
            nodepairs_cf.append([a,b])
    nodepairs_cf = np.asarray(nodepairs_cf)
    sample_nodepairs_f, sample_nodepairs_cf = sample_nodepairs(num_np, nodepairs_f, nodepairs_cf)

    train_y, train_drug_num, train_drug_cnt = generate_label(all_rating_info, train_id, global_item_id_map)
    valid_y, valid_drug_num, valid_drug_cnt = generate_label(all_rating_info, valid_id, global_item_id_map)
    test_y, test_drug_num, test_drug_cnt = generate_label(all_rating_info, test_id, global_item_id_map)

    return train_trainMat, valid_trainMat, test_trainMat, \
           train_userId, train_pos_itemId, train_neg_itemId, \
           train_mask, valid_mask, test_mask, \
           train_y, valid_y, test_y, \
           train_drug_num, valid_drug_num, test_drug_num, \
           num_user, num_item, drugID, user_features, item_features,\
           labels_f, labels_cf, T_f, train_T_f, train_T_cf, sample_nodepairs_f, sample_nodepairs_cf



def statFre1(rating_info, user_id):
    user_list = []
    for uid in user_id:
        itemId = rating_info[rating_info['userId']==uid]['itemId'].values.tolist()
        itemId = [ele for ele in itemId]
        if len(itemId) ==1:
            user_list.append(uid)
    return user_list


def create_dataset(ufeat, umask, label, drug_feat_len):
    output_len = drug_feat_len
    # input_len = subject_feat_len
    X = []
    y = []
    for i,drug_eat in enumerate(label):
        multi_hot_output = np.zeros(output_len)
        multi_hot_output[drug_eat] = 1
        u_index = umask[i]
        X.append(ufeat[u_index])
        y.append(multi_hot_output)

    return np.array(X), np.array(y)

def sample_nodepairs(num_np, nodepairs_f, nodepairs_cf):
    f_idx = np.random.choice(len(nodepairs_f), min(num_np,len(nodepairs_f)), replace=False)
    np_f = nodepairs_f[f_idx]
    cf_idx = np.random.choice(len(nodepairs_cf), min(num_np,len(nodepairs_f)), replace=False)
    np_cf = nodepairs_cf[cf_idx]
    return np_f, np_cf