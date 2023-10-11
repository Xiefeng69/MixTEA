import torch
import time
import scipy
import numpy as np
from sklearn import preprocessing

from src.utils import cal_distance

# pls: Pseudo Labeling Strategy

def find_new_alignment(train_pair, dev_s, dev_t, vec, strategy="sim_th", threshold=0):
    '''find potential alignment and add them into training data'''
    print(f"[find potential alignments...", end="")
    
    # find new entity pairs
    new_pair = []
    t_start = time.time()
    if strategy == "sim_th":
        new_pair = threshold_based_pls(dev_s, dev_t, vec)
    elif strategy == "bi_sim":
        new_pair = dual_direction_based_pls(dev_s, dev_t, vec)
    
    # remove pseudo aligned entities
    for e1,e2 in new_pair:
        if e1 in dev_s:
            dev_s.remove(e1)
    for e1,e2 in new_pair:
        if e2 in dev_t:
            dev_t.remove(e2)

    # return new pairs
    if len(new_pair) == 0:
        print(f'no new aligned pairs, time: {round((time.time()-t_start), 2)}]\n')
        return None
    else:
        print(f'new aligned pairs: {len(new_pair)}, time: {round((time.time()-t_start), 2)}]\n')
        return np.array(new_pair)

def dual_direction_based_pls(entity_s, entity_t, vec):
    '''find potential pairs of aligned entities based on mutual nearest neighbours'''
    new_pair = []
    np.random.shuffle(entity_s)
    np.random.shuffle(entity_t)
    Lvec = np.array([vec[e] for e in entity_s])
    Rvec = np.array([vec[e] for e in entity_t])

    Lvec = preprocessing.normalize(Lvec)
    Rvec = preprocessing.normalize(Rvec)
    sim_mat_left2right = np.matmul(Lvec, Rvec.T)
    del Lvec, Rvec;
    left2right = np.argmax(sim_mat_left2right, axis=1) # form like entid_from_kg1, entid_from,_kg2 
    sim_mat_right2left = sim_mat_left2right.T
    right2left = np.argmax(sim_mat_right2left, axis=1)
    del sim_mat_left2right
    del sim_mat_right2left
    
    # mutually nearest neighbor
    for leftid, rightid in enumerate(left2right):
        if right2left[rightid] == leftid:
            new_pair.append((entity_s[leftid], entity_t[rightid]))

    return new_pair

def threshold_based_pls(entity_s, entity_t, vec, sim_th=0.9, k=10):
    '''find potential pairs of aligned entities based on similarity threshold'''
    new_pair = []
    np.random.shuffle(entity_s)
    np.random.shuffle(entity_t)
    Lvec = np.array([vec[e] for e in entity_s])
    Rvec = np.array([vec[e] for e in entity_t])

    Lvec = preprocessing.normalize(Lvec)
    Rvec = preprocessing.normalize(Rvec)
    sim_mat = np.matmul(Lvec, Rvec.T)
    del Lvec, Rvec;
    max_sim = np.max(sim_mat, axis=1)
    max_sim_index = np.argmax(sim_mat, axis=1)

    for es_id in range(len(entity_s)):
        if max_sim[es_id] > sim_th:
            new_pair.append((entity_s[es_id], entity_t[max_sim_index[es_id]]))

    # potential_aligned_pairs = filter_sim_mat(sim_mat, sim_th)
    # return list(potential_aligned_pairs)

    return new_pair

def filter_sim_mat(mat, threshold, greater=True, equal=False):
    # np.where receive one argument will return coordinate
    if greater and equal:
        x, y = np.where(mat >= threshold)
    elif greater and not equal:
        x, y = np.where(mat > threshold)
    elif not greater and equal:
        x, y = np.where(mat <= threshold)
    else:
        x, y = np.where(mat < threshold)
    return set(zip(x, y))
