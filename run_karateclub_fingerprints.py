import argparse
import torch
import torch.nn as nn
from pytorch_lightning.metrics.functional import auroc
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
import os.path as osp
from models.fingerprints_model import Net
from node_fingerprints.data_netlsd import MyPreTransform as NetLSDTransform
from node_fingerprints.data_geoscattering import MyPreTransform as GeoScatteringTransform
from node_fingerprints.data_feather import MyPreTransform as FEATHERTransform
import warnings
import random
import numpy as np
warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset_directory_path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data')
lr = 0.0001
num_epochs = 200
batch_size = 32

criterion = nn.BCELoss()

def train(model, loader, optimizer):
    model.train()
    loss_all = 0
    for (iter, data) in enumerate(loader):
        data = data.to(device)
        if iter % 3 == 0:
            optimizer.zero_grad()
        loss = criterion(model(data), data.y.float())
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        if iter % 3 == 0:
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
    parser.add_argument('--kernel_type', type=str, default=None,
                        help='the kernel type, square or gaussian or silverman')
    parser.add_argument('--initialization_type', type=str, default=None,
                        help='the initialization way of parameters, Uniform or GMM')
    parser.add_argument('--fingerprint_type', type=str, default=None,
                        help='the node fingerprint type, NetLSD or GeoScattering or FEATHER')
    parser.add_argument('--if_learn',type=str,default=None,help='whether to learn parameters')
    parser.add_argument('--seed', type=int, default=0, help='the random seed to choose')
    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    args.type='karateclub'
    path = osp.join(osp.dirname(osp.realpath(__file__)), dataset_directory_path, 'data')
    dataset = TUDataset(path,name=args.dataset)
    class Config:
        def __init__(self):
            self.num_node = dataset.data.edge_index.max() + 1
            self.name = args.dataset
            if self.name == 'IMDB-MULTI':
                self.num_classes = 3
            else:
                self.num_classes = 2

    config = Config()
    path = osp.join(osp.dirname(osp.realpath(__file__)), dataset_directory_path, args.fingerprint_type)
    if args.fingerprint_type=='NetLSD':
        dataset = TUDataset(path,name=args.dataset,pre_transform=T.Compose([NetLSDTransform(config)]))
    elif args.fingerprint_type=='GeoScattering':
        dataset=TUDataset(path,name=args.dataset,pre_transform=T.Compose([GeoScatteringTransform(config)]))
    elif args.fingerprint_type=='FEATHER':
        dataset=TUDataset(path,name=args.dataset,pre_transform=T.Compose([FEATHERTransform(config)]))


    perm = torch.randperm(len(dataset), dtype=torch.long)
    dataset = dataset[perm]
    tenpercent = int(len(dataset) * 0.1)
    test_dataset = dataset[:2 * tenpercent]
    val_dataset = dataset[2 * tenpercent:int(3.5 * tenpercent)]
    train_dataset = dataset[int(3.5 * tenpercent):]
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = Net(dataset,args,config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=0.000001)

    best_val_loss, test_auc = 100, 0
    for epoch in range(1, 201):
        learning_rate = scheduler.optimizer.param_groups[0]['lr']
        train_loss = train(model, train_loader, optimizer)
        val_loss = val(model,val_loader)
        scheduler.step(val_loss)
        if best_val_loss >= val_loss:
            test_auc = test(model,test_loader)
            best_val_loss = val_loss
            print('Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, '
                 'Val Loss: {:.7f}, Test AUC: {:.7f}'.format(
               epoch, learning_rate, train_loss, val_loss, test_auc))
            if learning_rate <= 0.000001:
                    print("Converged.")
                    break
    print(test_auc)
