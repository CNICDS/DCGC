# -*- encoding: utf-8 -*-

import math
import torch

######################################## GAT Layer ########################################
class GAT_Layer(torch.nn.Module):
    def __init__(self, dim_in, dim_out, negative_slope=0.2):
        super(GAT_Layer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.w = torch.nn.Parameter(torch.FloatTensor(self.dim_in, self.dim_out))
        self.a_target = torch.nn.Parameter(torch.FloatTensor(self.dim_out, 1))
        self.a_neighbor = torch.nn.Parameter(torch.FloatTensor(self.dim_out, 1))
        torch.nn.init.xavier_normal_(self.w, gain=1.414)
        torch.nn.init.xavier_normal_(self.a_target, gain=1.414)
        torch.nn.init.xavier_normal_(self.a_neighbor, gain=1.414)

        self.leakyrelu = torch.nn.LeakyReLU(negative_slope)

    def forward(self, x, adj):
        # x (num_nodes, dim_in)
        x_ = torch.mm(x, self.w) # (num_nodes, dim_out)
        scores_target = torch.mm(x_, self.a_target) # (num_nodes, )
        scores_neighbor = torch.mm(x_, self.a_neighbor) # (num_nodes, )
        scores = scores_target + torch.transpose(scores_neighbor, 0, 1) # (num_nodes, num_nodes)
        
        scores = torch.mul(adj, scores)
        scores = self.leakyrelu(scores)
        scores = torch.where(adj>0, scores, -9e15*torch.ones_like(scores)) # score of non-negihbor is -9e15
        coefficients = torch.nn.functional.softmax(scores, dim=1)
        x_ = torch.nn.functional.elu(torch.mm(coefficients, x_))
        return x_