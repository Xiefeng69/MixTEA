import math
import torch
import torch.nn as nn
from torch import sparse
import numpy as np
import torch.nn.functional as F

class GraphConvLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, alpha=0.2, concat=True, w_adj=True, cuda=True, residual=False):
        super(SpGraphAttentionLayer, self).__init__()
        assert in_features == out_features
        self.w_adj = w_adj
        self.is_cuda = cuda
        self.concat = concat
        self.residual = residual
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.zeros(size=(out_features,), dtype=torch.float32))
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features), dtype=torch.float32))
        nn.init.ones_(self.W.data)
        stdv = 1. / math.sqrt(in_features * 2)
        nn.init.uniform_(self.a.data, -stdv, stdv)
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, inputs, adj):
        N = inputs.size()[0]
        ones = torch.ones(size=(N, 1), dtype=torch.float32)
        if self.is_cuda:
            ones = ones.to(inputs.device)

        edge = adj._indices()
        h = torch.mul(inputs, self.W)  # transformation

        assert not torch.isnan(h).any()
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        edge_e = torch.exp(self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()

        # for relation weighting
        # edge_e = edge_e * adj.values()

        e_rowsum = sparse.mm(torch.sparse_coo_tensor(edge, edge_e), ones)
        h_prime = sparse.mm(torch.sparse_coo_tensor(edge, edge_e), h)

        assert not torch.isnan(h_prime).any()
        h_prime = h_prime.div(e_rowsum)
        assert not torch.isnan(h_prime).any()
        # h_prime = (h_prime2 + h_prime) / 2
        if self.concat:
            output = F.elu(h_prime)
        else:
            output = h_prime

        if self.residual:
            output = inputs + output
            assert output.size() == inputs.size()
        return output

class SpGraphAttentionLayerV2(nn.Module):
    """
    Sparse version GATv2 layer, https://openreview.net/pdf?id=F72ximsx7C1
    """

    def __init__(self, in_features, out_features, alpha=0.2, concat=True, w_adj=True, cuda=True, residual=False):
        super(SpGraphAttentionLayerV2, self).__init__()
        assert in_features == out_features
        self.w_adj = w_adj
        self.is_cuda = cuda
        self.concat = concat
        self.residual = residual
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features), dtype=torch.float32))
        self.a = nn.Parameter(torch.zeros(size=(1, out_features), dtype=torch.float32))
        # nn.init.ones_(self.W.data)
        # stdv = 1. / math.sqrt(in_features * 2)
        # nn.init.uniform_(self.a.data, -stdv, stdv)
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, inputs, adj):
        N = inputs.size()[0]
        ones = torch.ones(size=(N, 1), dtype=torch.float32)
        if self.is_cuda:
            ones = ones.to(inputs.device)
        
        learnedW = torch.concat([self.W, self.W], dim=0) # constraint
        edge = adj._indices()
        edge_inputs = torch.cat((inputs[edge[0, :], :], inputs[edge[1, :], :]), dim=1) # concatenation original inputs
        leanredH = torch.mm(edge_inputs, learnedW)  # transformation after concatenation: learnH shape as [353543, 256]
        assert not torch.isnan(leanredH).any()
        edge_e = torch.exp(self.a.mm(self.leakyrelu(leanredH.t())).squeeze())
        assert not torch.isnan(edge_e).any()

        h = torch.mm(inputs, self.W)  # transformation
        e_rowsum = sparse.mm(torch.sparse_coo_tensor(edge, edge_e), ones)
        h_prime = sparse.mm(torch.sparse_coo_tensor(edge, edge_e), h)

        assert not torch.isnan(h_prime).any()
        h_prime = h_prime.div(e_rowsum)
        assert not torch.isnan(h_prime).any()
        # h_prime = (h_prime2 + h_prime) / 2
        if self.concat:
            output = F.relu(h_prime)
        else:
            output = h_prime

        if self.residual:
            output = inputs + output
            assert output.size() == inputs.size()
        return output

class GraphMultiHeadAttLayer(nn.Module):
    def __init__(self, in_features, out_features, n_head=2, alpha=0.2, cuda=True):
        super(GraphMultiHeadAttLayer, self).__init__()
        self.attentions = nn.ModuleList([SpGraphAttentionLayer(in_features, out_features, alpha, cuda) for _ in range(n_head)])

    def forward(self, inputs, adj):
        # inputs shape = [num, ent_dim]
        outputs = torch.cat(tuple([att(inputs, adj).unsqueeze(-1) for att in self.attentions]), dim=-1) # shape = [num, ent_dim, nheads]
        outputs = torch.mean(outputs, dim=-1)
        return outputs