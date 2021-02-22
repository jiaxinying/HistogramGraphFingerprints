'''
This file is largely adapted from https://github.com/chrsmrrs/k-gnn
'''
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import GINConv
import torch.nn as nn

class Net(torch.nn.Module):
    def __init__(self, config,args):
        super(Net, self).__init__()
        self.conv1 = GINConv(nn.Sequential(nn.Linear(config.num_features, 32)))
        self.conv2 = GINConv(nn.Sequential(nn.Linear(32, 64)))
        self.conv3 = GINConv(nn.Sequential(nn.Linear(64, 64)))
        self.fc1 = torch.nn.Linear(64, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)
        if args.type=="TU":
            self.fc3 = torch.nn.Linear(32, config.num_classes)
        self.args=args
        self.sigmoid = nn.Sigmoid()

    def reset_parameters(self):
        for (name, module) in self._modules.items():
            module.reset_parameters()

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index))
        data.x = F.elu(self.conv2(data.x, data.edge_index))
        data.x = F.elu(self.conv3(data.x, data.edge_index))
        x = scatter_mean(data.x, data.batch, dim=0)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        if self.args.type == "regression":
            return x.view(-1)
        elif self.args.type == "TU":
            return F.log_softmax(x, dim=1)
        else:
            return self.sigmoid(x).view(-1)