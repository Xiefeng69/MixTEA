import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.layers import *
from src.utils import mcd_matrix_fn

class KGEncoder(nn.Module):
    def __init__(self, args, ent_num, adj_matrix, rel_features, device, name) -> None:
        super().__init__()
        self.name = name
        self.ent_num = ent_num
        self.adj_matrix = adj_matrix
        self.rel_in, self.rel_out = rel_features
        self.ent_dim = args.ent_dim
        self.rel_dim = args.rel_dim
        self.layer = args.layer
        self.head = args.head
        self.ema_decay = args.ema_decay
        self.device = device
        self.k = self.layer + 2

        self.entity_embedding = nn.Embedding(num_embeddings=self.ent_num, embedding_dim=self.ent_dim, dtype=torch.float32)
        self.relation_embedding = nn.Embedding(num_embeddings=self.rel_in.shape[1], embedding_dim=self.rel_dim, dtype=torch.float32)
        self.encoder = nn.ModuleList([GraphMultiHeadAttLayer(in_features=self.ent_dim, out_features=self.ent_dim, n_head=self.head) for i in range(self.layer)])
        self.weight_raw = torch.ones(self.k, requires_grad=True, device=device)
        if name == "student":
            self.dropout = nn.Dropout(p=args.s_dropout)
        elif name == "teacher":
            self.dropout = nn.Dropout(p=args.t_dropout)

        self.init_state()

    def init_state(self):
        nn.init.xavier_uniform_(self.entity_embedding.weight.data)
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)

    def update(self, network: nn.Module, epoch):
        alpha = min(1 - 1 / epoch, self.ema_decay)
        for key_param, query_param in zip(self.parameters(), network.parameters()):
            key_param.data *= alpha
            key_param.data += (1 - alpha) * query_param.data
        self.eval()
    
    def forward(self, adj_matrix=None, rel_features=None, phase="train"):
        
        adj_matrix = self.adj_matrix
        rel_in = self.rel_in
        rel_out = self.rel_out

        # structural embedding
        str_embedding = self.entity_embedding.weight
        str_embedding_matrix = list()
        for layer in self.encoder:
            str_embedding = layer(str_embedding, adj_matrix)
            str_embedding_matrix.append(self.dropout(str_embedding))
        str_embedding = torch.cat(str_embedding_matrix, dim=1)

        # relation embedding
        rel_embedding = self.relation_embedding.weight
        rel_rowsum_in, rel_rowsum_out = torch.sum(rel_in.to_dense(), dim=-1).unsqueeze(-1), torch.sum(rel_out.to_dense(), dim=-1).unsqueeze(-1)
        rel_embedding_in = torch.mm(rel_in, rel_embedding)
        rel_embedding_in = rel_embedding_in.div(rel_rowsum_in + 1e-5)
        rel_embedding_out = torch.mm(rel_out, rel_embedding)
        rel_embedding_out = rel_embedding_out.div(rel_rowsum_out + 1e-5)
        rel_embedding = torch.cat([rel_embedding_in, rel_embedding_out], dim=1)

        # weighted concatenation
        w_normalized = F.softmax(self.weight_raw, dim=0)
        joint_embedding = [
            w_normalized[self.k-2] * F.normalize(rel_embedding_in),
            w_normalized[self.k-1] * F.normalize(rel_embedding_out)
        ]
        for k in range(self.k-2):
            joint_embedding.append(w_normalized[k] * F.normalize(str_embedding_matrix[k]))
        joint_embedding = torch.cat(joint_embedding, dim=1)

        return joint_embedding

    def alignment_loss(self, vec, align_left, align_right, neg_left, neg_right, neg_samples_size, neg_margin, dist=2):
        '''apply margin-based loss for supervised alignment learning'''
        t = len(align_left) # t means the length of labeled data 
        L = np.ones((t, neg_samples_size)) * (align_left.reshape((t,1))) # element-wise multiplication
        align_left = L.reshape((t * neg_samples_size,))
        R = np.ones((t, neg_samples_size)) * (align_right.reshape((t,1))) # element-wise multiplication
        align_right = R.reshape((t * neg_samples_size,))
        del L, R

        align_left = vec[align_left]
        align_right = vec[align_right]
        neg_left = vec[neg_left]
        neg_right = vec[neg_right]

        pos_score = F.pairwise_distance(align_left, align_right, p=dist)
        pos_score = torch.cat([pos_score, pos_score], dim=0)
        neg_score1 = F.pairwise_distance(align_left, neg_right, p=dist)
        neg_score2 = F.pairwise_distance(neg_left, align_right, p=dist)
        neg_score = torch.cat([neg_score1, neg_score2], dim=0)
        
        loss = F.relu(pos_score + neg_margin - neg_score)
        loss = torch.sum(loss)
        return loss
    
    def pseudo_ce_loss(self, ent_embedding1, ent_embedding2, ent_embedding3, ent_embedding4, hit1_st, hit1_ts):
        '''generate probabilistic pseudo mappings to provide more alignment signals for semi-supervised learning'''
        ent_embedding1 = F.normalize(ent_embedding1)
        ent_embedding2 = F.normalize(ent_embedding2)
        ent_embedding3 = F.normalize(ent_embedding3)
        ent_embedding4 = F.normalize(ent_embedding4)

        '''calculate similarity matrices'''
        sim_mat_stu = torch.mm(ent_embedding1, ent_embedding2.T)
        sim_mat_tea_st = torch.mm(ent_embedding3, ent_embedding4.T)
        sim_mat_tea_ts = sim_mat_tea_st.T

        # sim_mat_psl = torch.eye(n=sim_mat_tea.shape[0], m=sim_mat_tea.shape[1]).to(self.device)
        '''bi-directional voting (BDV) strategy'''
        sim_mat_psl_st = (sim_mat_tea_st == sim_mat_tea_st.max(dim=1, keepdim=True)[0]).to(self.device, dtype=torch.float32)
        sim_mat_psl_ts = (sim_mat_tea_ts == sim_mat_tea_ts.max(dim=1, keepdim=True)[0]).to(self.device, dtype=torch.float32)
        sim_mat_psl = (hit1_st / (hit1_st+hit1_ts)) * sim_mat_psl_st + (hit1_ts / (hit1_st+hit1_ts)) * sim_mat_psl_ts.T

        '''matching diversity-based rectification (MDR) module'''
        col_label_sum = torch.sum(sim_mat_psl, dim=0, keepdim=True) # [1 ent_num]
        row_label_sum = torch.sum(sim_mat_psl, dim=1, keepdim=True) # [ent_num 1]
        matching_diversity = col_label_sum.expand(sim_mat_psl.shape[0], col_label_sum.shape[1]) + row_label_sum.expand(row_label_sum.shape[0], sim_mat_psl.shape[1])
        sim_mat_psl = sim_mat_psl / (matching_diversity - sim_mat_psl)
        
        loss_ce = F.cross_entropy(sim_mat_stu, sim_mat_psl, reduction="none")
        loss = torch.sum(loss_ce)
        return loss