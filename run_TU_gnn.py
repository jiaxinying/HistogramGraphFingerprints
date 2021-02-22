import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree
from k_gnn import DataLoader as Loader
import torch.nn as nn
from k_gnn import TwoMalkin, ConnectedThreeMalkin
from models.GIN import Net as GINNet
from models.GNN import Net as GNNNet
from models.GNN_123 import Net as GNN123Net
import warnings
import argparse
import random
import numpy as np
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset_directory_path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data')
lr = 0.001
num_epochs = 200
batch_size = 32
degrees = {'MUTAG': 5, 'NCI1': 5, 'PROTEINS': 26, 'IMDB-BINARY': 136, 'IMDB-MULTI': 89}

def get_net(name):
    if name == 'GNN':
        return GNNNet
    elif name == 'GIN':
        return GINNet
    elif name == '1-2-3-GNN':
        return GNN123Net

class PreTransform(object):
    def __init__(self,args):
        self.args=args
    def __call__(self,data):
        if data.x is None:
            data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)
            data = TwoMalkin()(data)
            data = ConnectedThreeMalkin()(data)
            data.x = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
            data.x = F.one_hot(data.x, num_classes=int(degrees[self.args.dataset])).to(torch.float)
        else:
            x=data.x
            data=TwoMalkin()(data)
            data=ConnectedThreeMalkin()(data)
            data.x=x
        return data

criterion = nn.BCELoss()
def train(model, loader, optimizer):
    model.train()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(loader.dataset)


def val(model, loader):
    model.eval()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        loss_all += F.nll_loss(model(data), data.y).item()*data.num_graphs
    return loss_all / len(loader.dataset)


def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='the name of the TU dataset')
    parser.add_argument('--model', type=str, default=None,help='model to run, GNN, GIN or 1-2-3-GNN')
    args = parser.parse_args()
    args.type='TU'
    path = osp.join(osp.dirname(osp.realpath(__file__)), dataset_directory_path, 'gnn')
    dataset = TUDataset(path,name=args.dataset,pre_transform=PreTransform(args))
    class Config:
        def __init__(self):
            self.num_features = dataset.data.x.size()[1]
            self.num_node = dataset.data.edge_index.max() + 1
            self.name = args.dataset
            if self.name == 'IMDB-MULTI':
                self.num_classes = 3
            else:
                self.num_classes = 2

    config = Config()
    dataset.data.iso_type_2 = torch.unique(dataset.data.iso_type_2, True, True)[1]
    config.num_i_2 = dataset.data.iso_type_2.max().item() + 1
    dataset.data.iso_type_2 = F.one_hot(
        dataset.data.iso_type_2, num_classes=config.num_i_2).to(torch.float)

    dataset.data.iso_type_3 = torch.unique(dataset.data.iso_type_3, True, True)[1]
    config.num_i_3 = dataset.data.iso_type_3.max().item() + 1
    dataset.data.iso_type_3 = F.one_hot(
        dataset.data.iso_type_3, num_classes=config.num_i_3).to(torch.float)

    perm = torch.randperm(len(dataset), dtype=torch.long)
    dataset = dataset[perm]
    acc = []
    for i in range(10):
        Net=get_net(args.model)
        model = Net(config,args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=0.000001)
        test_mask = torch.zeros(len(dataset), dtype=torch.uint8)
        n = len(dataset) // 10
        test_mask[i * n:(i + 1) * n] = 1
        test_dataset = dataset[test_mask]
        train_dataset = dataset[1 - test_mask]

        n = len(train_dataset) // 10
        val_mask = torch.zeros(len(train_dataset), dtype=torch.uint8)
        val_mask[i * n:(i + 1) * n] = 1
        val_dataset = train_dataset[val_mask]
        train_dataset = train_dataset[1 - val_mask]
        if args.model=='1-2-3-GNN':
            val_loader = Loader(val_dataset, batch_size=batch_size)
            test_loader = Loader(test_dataset, batch_size=batch_size)
            train_loader = Loader(train_dataset, batch_size=batch_size, shuffle=True)
        else:
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        print('---------------- Split {} ----------------'.format(i))

        test_acc = None
        best_val_error = None
        for epoch in range(num_epochs):
            learning_rate = scheduler.optimizer.param_groups[0]['lr']
            loss = train(model, train_loader, optimizer)
            val_error = val(model, val_loader)
            scheduler.step(val_error)
            if best_val_error is None or val_error <= best_val_error:
                test_acc = test(model, test_loader)
                best_val_error = val_error
                print('Epoch: {:03d}, LR: {:.7f}, Loss: {:.7f}, Validation Loss: {:.7f}, '
                     'Test Acc: {:.7f}'.format(epoch, learning_rate, loss, val_error, test_acc))
            if learning_rate <= 0.000001:
                print("Converged.")
                break
        acc.append(test_acc)
    acc = torch.tensor(acc)
    print('---------------- Final Result ----------------')
    print('Mean: {:7f}, Std: {:7f}'.format(acc.mean(), acc.std()))
