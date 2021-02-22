'''
This file is largely adapted from https://github.com/chrsmrrs/k-gnn
'''
from torch_scatter import scatter_mean,scatter_add
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean
from k_gnn import GraphConv, avg_pool
import torch.nn as nn
class Net(torch.nn.Module):
    def __init__(self,config,args):
        super(Net, self).__init__()
        self.conv1 = GraphConv(config.num_features, 64)
        self.conv2 = GraphConv(64, 64)
        self.conv3 = GraphConv(64, 64)
        self.conv4 = GraphConv(64 + config.num_i_2, 64)
        self.conv5 = GraphConv(64, 64)
        self.conv6 = GraphConv(64 + config.num_i_3, 64)
        self.conv7 = GraphConv(64, 64)
        self.fc1 = torch.nn.Linear(3 * 64, 64)
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
        x = data.x
        x_1 = scatter_add(data.x, data.batch, dim=0)

        data.x = avg_pool(x, data.assignment_index_2)
        data.x = torch.cat([data.x, data.iso_type_2], dim=1)

        data.x = F.elu(self.conv4(data.x, data.edge_index_2))
        data.x = F.elu(self.conv5(data.x, data.edge_index_2))
        x_2 = scatter_mean(data.x, data.batch_2, dim=0)

        data.x = avg_pool(x, data.assignment_index_3)
        data.x = torch.cat([data.x, data.iso_type_3], dim=1)

        data.x = F.elu(self.conv6(data.x, data.edge_index_3))
        data.x = F.elu(self.conv7(data.x, data.edge_index_3))
        x_3 = scatter_mean(data.x, data.batch_3, dim=0)

        x = torch.cat([x_1, x_2, x_3], dim=1)

        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        if self.args.type == "regression":
            return x.view(-1)
        elif self.args.type == "TU":
            return F.log_softmax(x, dim=1)
        else:
            return self.sigmoid(x).view(-1)
