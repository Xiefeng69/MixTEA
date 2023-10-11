import torch
import torch.nn as nn
from torch.nn import *
import torch.nn.functional as F
import numpy as np

from src.layers import GraphConvLayer, SpGraphAttentionLayer

class GAT(nn.Module):
    def __init__(self, args, ent_num, adj_matrix) -> None:
        super().__init__()
        self.ent_num = ent_num
        self.adj_matrix = adj_matrix
        self.embedding_dim = args.ent_dim
        self.layer = args.layer
        self.dropout = args.dropout

        self.entity_embedding = nn.Embedding(num_embeddings=self.ent_num, embedding_dim=self.embedding_dim)
        self.gatblocks = nn.ModuleList([SpGraphAttentionLayer(in_features=self.embedding_dim, out_features=self.embedding_dim, dropout=self.dropout) for i in range(self.layer)])
    
    def forward(self, **args):
        ent_embedding = self.entity_embedding.weight
        for layer in self.gatblocks:
            ent_embedding = layer(ent_embedding, self.adj_matrix)
        return ent_embedding