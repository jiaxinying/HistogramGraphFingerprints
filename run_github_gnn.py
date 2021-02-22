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
from pytorch_lightning.metrics.functional import auroc
import random
import numpy as np

warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset_directory_path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data')
lr = 0.001
num_epochs = 200
batch_size = 5
degrees = {'github_stargazers': 756, 'twitch_egos': 102, 'reddit_threads': 94, 'deezer_ego_nets': 363}
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
        self.div = 20  # avoid out of GPU memory
    def __call__(self,data):
        data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)
        data = TwoMalkin()(data)
        data = ConnectedThreeMalkin()(data)
        data.x = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
        data.x = F.one_hot(data.x//self.div, num_classes=(degrees[self.args.dataset])//self.div+1).to(torch.float)
        return data

criterion = nn.BCELoss()
def train(model, loader, optimizer):
    model.train()
    loss_all = 0
    for (iter, data) in enumerate(loader):
        data = data.to(device)
        if iter % 5 == 0:
            optimizer.zero_grad()
        loss = criterion(model(data), data.y.float())
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        if iter % 5 == 0:
            optimizer.step()
    return loss_all / len(loader.dataset)

def val(model,loader):
    model.eval()
    loss_all = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            loss_all += criterion(model(data), data.y.float()).item() * data.num_graphs
    return loss_all / len(loader.dataset)

def test(model,loader):
    model.eval()
    pred = []
    y = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            y.append(data.y.detach().cpu())
            pred.append(model(data).detach().cpu())
    return auroc(torch.cat(pred), torch.cat(y))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='the name of the TU dataset')
    parser.add_argument('--model', type=str, default=None,help='GNN,GIN or 1-2-3-GNN')
    parser.add_argument('--seed', type=int, default=0, help='the random seed to choose')
    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    args.type='karateclub'
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
    tenpercent = int(len(dataset) * 0.1)
    test_dataset = dataset[:2 * tenpercent]
    val_dataset = dataset[2 * tenpercent:int(3.5 * tenpercent)]
    train_dataset = dataset[int(3.5 * tenpercent):]
    if args.model == '1-2-3-GNN':
        val_loader = Loader(val_dataset, batch_size=batch_size)
        test_loader = Loader(test_dataset, batch_size=batch_size)
        train_loader = Loader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    Net = get_net(args.model)
    model = Net(config, args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5,
                                                           min_lr=0.000001)

    best_val_loss, test_auc = 100, 0
    for epoch in range(1, 201):
        learning_rate = scheduler.optimizer.param_groups[0]['lr']
        train_loss = train(model, train_loader, optimizer)
        val_loss = val(model, val_loader)
        scheduler.step(val_loss)
        if best_val_loss >= val_loss:
            test_auc = test(model, test_loader)
            best_val_loss = val_loss
            print('Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, '
                'Val Loss: {:.7f}, Test AUC: {:.7f}'.format(
             epoch, learning_rate, train_loss, val_loss, test_auc))
            if learning_rate <= 0.000001:
                print("Converged.")
                break
    print(test_auc)
