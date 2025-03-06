import torch
import numpy
import scipy,scipy.linalg
from torch_geometric.nn import GCNConv, Sequential, GINConv, ChebConv
from torch import nn
from torch_geometric.utils import to_undirected
from torch_geometric.nn.models import MLP

class GIN(nn.Module):
    def __init__(self, hidden_dim, input_dim, gcn_layers=1, mlp_layers=3, nbclasses=6, 
                 node_level=False, directed=True, regression=False, bias=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.gcn_layers = gcn_layers
        self.nbclasses = nbclasses
        self.node_level = node_level
        self.bias = bias
        self.act = nn.ReLU(inplace=True)

        layers = []
        for _ in range(gcn_layers):
            if _ == 0:
                mlp = MLP([input_dim]+[hidden_dim]*(mlp_layers-1), act=self.act, act_first=True, norm=None, bias=bias)
                layers += [(GINConv(mlp), 'x, edge_index -> x'),
                            nn.ReLU(inplace=True)]
            else:
                mlp = MLP([hidden_dim]*(mlp_layers), act=self.act, act_first=True, norm=None, bias=bias)
                layers += [(GINConv(mlp), 'x, edge_index -> x'),
                            nn.ReLU(inplace=True)]
        self.gin = Sequential('x, edge_index', layers)
        self.linear = torch.nn.Linear(hidden_dim, nbclasses, bias=bias)

        self.params = list(self.gin.parameters()) + [self.linear.weight, self.linear.bias]
        self.directed = directed
        self.regression = regression

    def forward(self, x, edge_index, edge_attr=None):
        # x: [V, F]
        # edge_index: [2, E]
        if not self.directed:
            edge_index = to_undirected(edge_index)
        
        out = self.gin.forward(x, edge_index)            
        
        y = self.linear(out)
        
        # y = h
        if not self.node_level:
            y = y.sum(axis=0)
        if self.regression:
            y = torch.relu(y)
        
        return y