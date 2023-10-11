import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
import random
import numpy as np
import scipy.sparse as sp
from collections import Counter

from src.load_lm import *
from src.pls import find_new_alignment

class KGData():

    def __init__(self, task, fold, train_ratio=0.3, setting="semisup") -> None:
        self.task = self.task_mapping(task)
        self.fold = fold
        self.train_ratio = train_ratio
        self.setting = setting

        # load KG information about entities/relations/triples          
        self.entity1, self.rel1, self.triples1 = self.load_triples('data/' + self.task + '/mapped_triples_1')
        self.entity2, self.rel2, self.triples2 = self.load_triples('data/' + self.task + '/mapped_triples_2')
        
        self.kg1_ent_num = len(self.entity1)
        self.kg2_ent_num = len(self.entity2)
        self.ent_num  = len(self.entity1.union(self.entity2))

        print(f'KG 1 info: #ent. {len(self.entity1)}, #rel. {len(self.rel1)}, #tri. {len(self.triples1)}')
        print(f'KG 2 info: #ent. {len(self.entity2)}, #rel. {len(self.rel2)}, #tri. {len(self.triples2)}\n')
    
    def task_mapping(self, task):
        dict = {
            "d_w_15k": "D_W_15K",
            "d_y_15k": "D_Y_15K",
            "en_fr_15k": "EN_FR_15K",
            "en_de_15k": "EN_DE_15K",
        }
        return dict[task]

    def load_pair_data(self):
        train_pair = self.load_alignment_pair(f'data/{self.task}/721_5fold/{self.fold}/mapped_train_links')
        valid_pair = self.load_alignment_pair(f'data/{self.task}/721_5fold/{self.fold}/mapped_valid_links')
        test_pair = self.load_alignment_pair(f'data/{self.task}/721_5fold/{self.fold}/mapped_test_links')
        return np.array(train_pair), np.array(valid_pair), np.array(test_pair)

    def load_matrix_data(self):
        # obtain adjacency matrix
        adj_matrix, r_index, r_val, adj_features, rel_features, rdict, rel_in, rel_out = self.get_matrix(self.triples1+self.triples2, self.entity1.union(self.entity2), self.rel1.union(self.rel2))
        assert self.ent_num == adj_features.shape[0]
        rel_features_top = self.load_relation_topR(e=self.ent_num, KG=self.triples1+self.triples2)
        self.d_v = self.get_degree_vector(adj_matrix)
        self.rel_num = int(rel_features.shape[1] / 2)
        self.triple_num = adj_matrix.getnnz()
        return adj_matrix, np.array(r_index), np.array(r_val), adj_features, rel_features, rel_features_top, rdict, rel_in, rel_out

    def normalize_adj(self, adj):
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).T

    def load_triples(self, file_name):
        triples = []
        entity = set()
        rel = set()
        for line in open(file_name,'r'):
            head,r,tail = [int(item) for item in line.split()]
            entity.add(head); entity.add(tail); rel.add(r)
            triples.append((head, r, tail))
        return entity, rel, triples

    def load_alignment_pair(self, file_name):
        alignment_pair = []
        c = 0
        for line in open(file_name,'r'):
            e1,e2 = line.split()
            alignment_pair.append((int(e1),int(e2)))
        return alignment_pair

    def get_matrix(self, triples, entity, rel):
        '''
            @input: triples, entity ids, relation ids
            @output:
                adj_matrix: shape as [ent_size, ent_size].
                adj_features: shape as [ent_size, ent_size], normalized adj_matrix.
                rel_features: shape as [ent_size, 2*rel_size], concatenated by rel_in (shape as [ent_size, rel_size]) and rel_out (shape as [ent_size, rel_size]).
        '''
        ent_size = max(entity) + 1
        rel_size = max(rel) + 1
        
        adj_matrix = sp.lil_matrix((ent_size,ent_size))
        adj_features = sp.lil_matrix((ent_size,ent_size))
        radj = []
        rel_in = np.zeros((ent_size,rel_size))
        rel_out = np.zeros((ent_size,rel_size))
        
        for i in range(max(entity)+1):
            adj_features[i,i] = 1

        for h,r,t in triples:        
            adj_matrix[h,t] = 1; adj_matrix[t,h] = 1;
            adj_features[h,t] = 1; adj_features[t,h] = 1;
            radj.append([h,t,r]); radj.append([t,h,r+rel_size]); 
            rel_out[h][r] += 1; rel_in[t][r] += 1
        
        rdict={}
        for h,r,t in triples:
            if r not in rdict:
                rdict[r]=[[],[]]
                rdict[r][0].append(h)
                rdict[r][1].append(t)
            else:
                rdict[r][0].append(h)
                rdict[r][1].append(t)
        
        count = -1
        s = set()
        d = {}
        r_index,r_val = [],[]
        for h,t,r in sorted(radj,key=lambda x: x[0]*10e10+x[1]*10e5):
            if ' '.join([str(h),str(t)]) in s:
                r_index.append([count,r])
                r_val.append(1)
                d[count] += 1
            else:
                count += 1
                d[count] = 1
                s.add(' '.join([str(h),str(t)]))
                r_index.append([count,r])
                r_val.append(1)
        for i in range(len(r_index)):
            r_val[i] /= d[r_index[i][0]]
        
        rel_features = np.concatenate([rel_in, rel_out],axis=1)
        adj_features = self.normalize_adj(adj_features)
        rel_features = self.normalize_adj(sp.lil_matrix(rel_features))
        rel_in = sp.lil_matrix(rel_in)
        rel_out = sp.lil_matrix(rel_out)
        return adj_matrix, r_index, r_val, adj_features, rel_features, rdict, rel_in, rel_out

    def load_relation_topR(self, e, KG, topR=1000):
        rel_mat = np.zeros((e, topR), dtype=np.float32)
        rels = np.array(KG)[:,1]
        top_rels = Counter(rels).most_common(topR)
        rel_index_dict = {r:i for i,(r,cnt) in enumerate(top_rels)}
        for tri in KG:
            h = tri[0]
            r = tri[1]
            o = tri[2]
            if r in rel_index_dict:
                rel_mat[h][rel_index_dict[r]] += 1.
                rel_mat[o][rel_index_dict[r]] += 1.
        return sp.lil_matrix(rel_mat)

    def get_degree_vector(self, adj_matrix):
        rowsum = np.array(adj_matrix.sum(1))
        rowsum = np.array([int(v[0]) for v in rowsum])
        return rowsum

    def negative_sampling(self, bsize, kg1_align, kg2_align, neg_samples_size, target_left=None, target_right=None, vec=None, e=0.9):
        print("[generate negative samples...]\n")
        if e == 0: # uniform negative sampling
            if target_left is not None:
                neg_left = random.choices(target_left, k=neg_samples_size * bsize)
                neg_right = random.choices(target_right, k=neg_samples_size * bsize)
            else:
                neg_left = np.random.choice(range(0,self.kg1_ent_num-1), size=neg_samples_size * bsize)
                neg_right = np.random.choice(range(self.kg1_ent_num, self.kg1_ent_num + self.kg2_ent_num - 1), size=neg_samples_size * bsize)
        else: # epsilon-truncated negative sampling
            vec = preprocessing.normalize(vec)
            def generate_hard_samples(embedding, ent_num):
                sim_mat = np.dot(embedding, embedding.transpose())
                neg_samples_list = self.get_nearest_neighbor(sim_mat, samples_size=math.ceil(ent_num * (1-e)))
                tmp = []
                for item in neg_samples_list:
                    tmp.append(random.sample(list(item), neg_samples_size))
                neg_samples = np.array(tmp)
                neg_samples = neg_samples.reshape((bsize*neg_samples_size,))
                del sim_mat
                return neg_samples
            neg_left = generate_hard_samples(vec[kg1_align], ent_num=self.kg1_ent_num)
            neg_right = generate_hard_samples(vec[kg2_align], ent_num=self.kg2_ent_num)

        return np.array(neg_left), np.array(neg_right)

    def get_nearest_neighbor(self, sim_mat, samples_size):
        '''calculate nearest samples for each entity'''
        ranks = np.argsort(sim_mat, axis=1)
        candidates = ranks[:, 1:samples_size + 1]
        return candidates

class Dataset(object):
    def __init__(self, anchor):
        self.anchor = torch.from_numpy(anchor)
        self.a1anchor = self.anchor[:,0].numpy()
        self.a2anchor = self.anchor[:,1].numpy()
    def __getitem__(self, index):
        return self.a1anchor[index], self.a2anchor[index]
    def __len__(self):
        return len(self.a1anchor)