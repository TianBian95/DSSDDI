import csv
import re
import torch as th
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

class MetricLogger(object):
    def __init__(self, attr_names, parse_formats, save_path):
        self._attr_format_dict = OrderedDict(zip(attr_names, parse_formats))
        self._file = open(save_path, 'w')
        self._csv = csv.writer(self._file)
        self._csv.writerow(attr_names)
        self._file.flush()

    def log(self, **kwargs):
        self._csv.writerow([parse_format % kwargs[attr_name]
                            for attr_name, parse_format in self._attr_format_dict.items()])
        self._file.flush()

    def close(self):
        self._file.close()


def torch_total_param_num(net):
    return sum([np.prod(p.shape) for p in net.parameters()])


def torch_net_info(net, save_path=None):
    info_str = 'Total Param Number: {}\n'.format(torch_total_param_num(net)) +\
               'Params:\n'
    for k, v in net.named_parameters():
        info_str += '\t{}: {}, {}\n'.format(k, v.shape, np.prod(v.shape))
    info_str += str(net)
    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write(info_str)
    return info_str


def get_activation(act):
    """Get the activation based on the act string

    Parameters
    ----------
    act: str or callable function

    Returns
    -------
    ret: callable function
    """
    if act is None:
        return lambda x: x
    if isinstance(act, str):
        if act == 'leaky':
            return nn.LeakyReLU(0.1)
        elif act == 'relu':
            return nn.ReLU()
        elif act == 'tanh':
            return nn.Tanh()
        elif act == 'sigmoid':
            return nn.Sigmoid()
        elif act == 'softsign':
            return nn.Softsign()
        else:
            raise NotImplementedError
    else:
        return act


def get_optimizer(opt):
    if opt == 'sgd':
        return optim.SGD
    elif opt == 'adam':
        return optim.Adam
    else:
        raise NotImplementedError


def to_etype_name(rating):
    return str(rating).replace('.', '_')


def apk(actual, predicted, k=10):
    """
    Source: https://github.com/statisticianinstilettos/recmetrics/blob/master/recmetrics/metrics.py
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Source: https://github.com/statisticianinstilettos/recmetrics/blob/master/recmetrics/metrics.py
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def mr(actual, predicted):
    """
    Computes the recall of each user's list of recommendations, and averages precision over all users.
    ----------
    actual : a list of lists
        Actual items to be predicted
        example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    Returns:
    -------
        recall: int
    """
    def calc_recall(predicted, actual):
        reca = [value for value in predicted if value in actual]
        reca = np.round(float(len(reca)) / float(len(actual)), 4)
        return reca

    recall = np.mean(list(map(calc_recall, predicted, actual)))
    return recall


def evaluate_model(R_hat, y_true, top_n):
    top_N_res = np.array(np.argsort(-R_hat, axis=1)[:, :top_n]).tolist()

    # rec_at_k = mr(y_true, top_N_recs)
    r = getLabel(y_true, top_N_res)
    ndcg=NDCGatK_r(y_true, r, top_n)
    ret = RecallPrecision_ATk(y_true, r, top_n)
    prec,recall=ret['precision'], ret['recall']
    F1_score = 2 * (prec * recall)/(prec + recall)
    return top_N_res,F1_score,prec,recall,ndcg

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

def NDCGatK_r(test_data, r, k):
	"""
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
	assert len(r) == len(test_data)
	pred_data = r[:, :k]

	test_matrix = np.zeros((len(pred_data), k))
	for i, items in enumerate(test_data):
		length = k if k <= len(items) else len(items)
		test_matrix[i, :length] = 1
	max_r = test_matrix
	idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
	dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
	dcg = np.sum(dcg, axis=1)
	idcg[idcg == 0.] = 1.
	ndcg = dcg / idcg
	ndcg[np.isnan(ndcg)] = 0.
	return np.mean(ndcg)


def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k

    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall_n = np.where(recall_n != 0, recall_n, 1)
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall/len(test_data), 'precision': precis/len(test_data)}


