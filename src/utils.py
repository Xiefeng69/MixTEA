import math
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import preprocessing
import scipy
import scipy.sparse as sp
import random
import faiss
import json

def set_random_seed(seed_value=0):
    print(f"current seed is set to \033[92m{seed_value}\033[0m \n")
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)

def save_args(args):
    args_dict = args.__dict__
    with open(f'args/{args.task}.json', 'w', encoding='utf-8') as f:
        json.dump(args_dict, f)

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.FloatTensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def adjust_learning_rate(optimizer, epoch, lr, total_epoch=100, s="step"):
    '''adjust learning rate'''
    if epoch % 10 == 1 and s=="step":
        _lr = lr * 0.5 ** (epoch//10)
        for param_group in optimizer.param_groups:
            param_group['lr'] = _lr
    elif s=="cosine":
        _lr = 0.5 * (1 + math.cos((math.pi * epoch) / total_epoch)) * lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = _lr

def csls_sim(sim_mat, csls_k):
    """
        https://github.com/cambridgeltl/eva/blob/948ffbf3bf70cd3db1d3b8fcefc1660421d87a1f/src/utils.py#L287
        Compute pairwise csls similarity based on the input similarity matrix.
        Parameters
        ----------
        sim_mat : matrix-like
            A pairwise similarity matrix.
        csls_k : int
            The number of nearest neighbors.
        Returns
        -------
        csls_sim_mat : A csls similarity matrix of n1*n2.
    """

    nearest_values1 = torch.mean(torch.topk(sim_mat, csls_k)[0], 1)
    nearest_values2 = torch.mean(torch.topk(sim_mat.t(), csls_k)[0], 1)
    sim_mat = 2 * sim_mat.t() - nearest_values1
    sim_mat = sim_mat.t() - nearest_values2
    return sim_mat

def cal_distance(Lvec, Rvec, sim_measure, csls_k=None):
    '''calculate similarity matrix based on <sim_measure>'''
    if sim_measure == "cityblock":
        sim_mat = scipy.spatial.distance.cdist(Lvec, Rvec, metric="cityblock")
    elif sim_measure == "euclidean":
        sim_mat = scipy.spatial.distance.cdist(Lvec, Rvec, metric="euclidean")
    elif sim_measure == "cosine":
        sim_mat = scipy.spatial.distance.cdist(Lvec, Rvec, metric="cosine") # 注意这里计算的余弦距离，即1-cos()，而不是余弦相似度
    elif sim_measure == "csls":
        sim_mat = scipy.spatial.distance.cdist(Lvec, Rvec, metric="cosine") # calculate pair-wise similarity
        del Lvec, Rvec
        sim_mat = 1 - csls_sim(1 - torch.from_numpy(sim_mat), csls_k=csls_k) # calculate csls similarity and transfer to distance form
        sim_mat = sim_mat.numpy()
    return sim_mat

def cal_metrics(sim_mat, Lvec, Rvec, data, k):
    '''calculate evaluation metrics: hit@1. hit@k, MRR, Mean etc.'''
    test_num = len(data)
    assert test_num == Lvec.shape[0]
    mrr = 0
    mean = 0
    hit_1_score = 0
    hit_k_score = 0
    for idx in range(Lvec.shape[0]):
        # np.argsort generate ascending sort: smaller is more similar
        rank = sim_mat[idx,:].argsort()
        assert idx in rank
        rank_index = np.where(rank==idx)[0][0]
        rank_index += 1
        mean += (rank_index)
        mrr += 1.0 / (rank_index)
        if rank_index <= 1: # hit@1
            hit_1_score += 1
        if rank_index <= k: # hit@k
            hit_k_score += 1
    mrr = mrr / test_num
    hit_1_score = hit_1_score / test_num
    hit_k_score = hit_k_score / test_num
    mean = mean / test_num
    return hit_1_score, hit_k_score, mrr, mean

def get_act_function(activate_function):
    """
        Get activation function by name
        :param activation_fuction: Name of activation function 
    """
    if activate_function == 'sigmoid':
        activate_function = nn.Sigmoid()
    elif activate_function == 'relu':
        activate_function = nn.ReLU()
    elif activate_function == 'tanh':
        activate_function = nn.Tanh()
    else:
        return None
    return activate_function

def eval_entity_alignment(Lvec, Rvec, test_num, k, eval_normalize=True):
    '''calculate evaluation metrics: hit@1, hit@k, MRR, Mean etc.'''
    if eval_normalize:
        Lvec = preprocessing.normalize(Lvec)
        Rvec = preprocessing.normalize(Rvec)
    sim = cal_distance(Lvec, Rvec, sim_measure="cityblock")
    top_k=(1, k)
    mr = 0
    mrr = 0
    top_lr = [0] * len(top_k)
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        mr += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    mr /=  test_num
    mrr /= test_num

    return top_lr[0] / test_num, top_lr[1] / test_num, mrr, mr

def eval_entity_alignment_faiss(Lvec, Rvec, test_num, k, eval_metric, eval_normalize=True):
    '''
        calculate evaluation metrics: hit@1, hit@k, MRR, Mean etc.
        using faiss accelerate alignment inference: https://github.com/facebookresearch/faiss
    '''
    if eval_normalize:
        Lvec = preprocessing.normalize(Lvec)
        Rvec = preprocessing.normalize(Rvec)
    assert test_num == Lvec.shape[0]
    mrr = 0
    mean = 0
    hit_1_score = 0
    hit_k_score = 0
    if eval_metric == "l2":
        index = faiss.IndexFlatL2(Rvec.shape[1]) # create index base with fixed dimension
    elif eval_metric == "inner":
        index = faiss.IndexFlatIP(Rvec.shape[1])
    else:
        assert ValueError
    index.add(np.ascontiguousarray(Rvec)) # add key to index base
    del Rvec;
    _, I = index.search(np.ascontiguousarray(Lvec), test_num) # search query in index base
    for idx in range(Lvec.shape[0]):
        rank_index = np.where(I[idx,:]==idx)[0][0]
        rank_index += 1
        mean += (rank_index)
        mrr += 1.0 / (rank_index)
        if rank_index <= 1: # hit@1
            hit_1_score += 1
        if rank_index <= k: # hit@k
            hit_k_score += 1
    mrr = mrr / test_num
    hit_1_score = hit_1_score / test_num
    hit_k_score = hit_k_score / test_num
    mean = mean / test_num
    return hit_1_score, hit_k_score, mrr, mean

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length

def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def sim_norm(csls_sim_mat):
    min_val = torch.min(csls_sim_mat)
    max_val = torch.max(csls_sim_mat)
    val_range = max_val - min_val
    return (csls_sim_mat - min_val) / val_range

def mcd_matrix_fn(sim_matrix):
    '''"From Diversity-based Prediction to Better Ontology & Schema Matching." WWW2016'''
    n, m = sim_matrix.shape[0], sim_matrix.shape[1]
    row_sum = torch.sum(sim_matrix, dim=1, keepdims=True)
    col_sum = torch.sum(sim_matrix, dim=0, keepdims=True)
    row_mean = torch.mean(sim_matrix, dim=1, keepdims=True)
    col_mean = torch.mean(sim_matrix, dim=0, keepdims=True)
    mu_mat = (- sim_matrix + row_sum + col_sum) / (n + m - 1)
    mat = sim_matrix - mu_mat
    mat = torch.square(mat)
    return sim_norm(mat)