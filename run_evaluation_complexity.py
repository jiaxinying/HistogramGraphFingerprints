import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree
from k_gnn import DataLoader as Loader
from k_gnn import TwoMalkin, ConnectedThreeMalkin
from models.GIN import Net as GINNet
from models.GNN import Net as GNNNet
from models.GNN_123 import Net as GNN123Net
from models.fingerprints_model import Net as FGNNNet
from models.PGNN import Net as PGNNNet
from data_generator.randomGraph import RandomGraph
from node_fingerprints.data_geoscattering import MyPreTransform as PreTransform3
import warnings
import argparse
import numpy as np
import time
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset_directory_path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data','random-graph')
lr = 0.001
num_epochs = 500
batch_size = 32
seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


def get_net(name):
    if name == 'GNN':
        return GNNNet
    elif name == 'GIN':
        return GINNet
    elif name == '1-2-3-GNN':
        return GNN123Net
    elif name=='powerful-GNN':
        return PGNNNet
    elif name=='fingerprint':
        return FGNNNet

class PreTransform1(object):
    def __call__(self, data):
        data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)
        data = TwoMalkin()(data)
        data = ConnectedThreeMalkin()(data)
        data.x = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
        data.x = F.one_hot(data.x//4, num_classes=128//4+1).to(torch.float)
        return data

class PreTransform2(object):
    def __call__(self, data):
        data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)
        x = data.x.numpy()
        edge_index = data.edge_index.numpy()
        n = x.shape[0]
        affinity = np.zeros((n, n))
        for ii in range(edge_index.shape[1]):
            affinity[edge_index[0, ii], edge_index[1, ii]] = 1
        nodes_num = x.shape[0]
        graph = np.empty((nodes_num, nodes_num, 1))
        graph[:, :, 0] = affinity
        graph = np.transpose(graph, [2, 0, 1])
        data.graph = torch.from_numpy(graph).float()
        return data

class PreTransform4(object):
    def __call__(self, data):
        data.x = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
        data.x = F.one_hot(data.x//4, num_classes=128//4+1).to(torch.float)
        return data
def train(model, loader, optimizer):
    model.train()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.mse_loss(model(data), data.y[:,0])
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(loader.dataset)

def val(model,loader):
    model.eval()
    loss_all = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            loss_all += F.mse_loss(model(data), data.y[:,0].float()).item() * data.num_graphs
    return loss_all / len(loader.dataset)

def test(model,loader):
    model.eval()
    error=0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            error += (model(data) - data.y[:,0]).abs().sum().item()
    return error / len(loader.dataset)

def train_process(dataset,args,config):
    perm = torch.randperm(len(dataset), dtype=torch.long)
    dataset = dataset[perm]
    error=[]
    Net=get_net(args.model)
    if args.model=='fingerprint':
        model = Net(dataset, args, config).to(device)
    else:
        model = Net(config,args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    test_mask = torch.zeros(len(dataset), dtype=torch.uint8)
    n = len(dataset) // 5
    test_mask[: n] = 1
    test_dataset = dataset[test_mask]
    train_dataset = dataset[1 - test_mask]

    n = len(train_dataset) // 5
    val_mask = torch.zeros(len(train_dataset), dtype=torch.uint8)
    val_mask[:n] = 1
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

    test_error = None
    best_val_error = None
    for epoch in range(num_epochs):
        loss = train(model, train_loader, optimizer)
        val_error = val(model, val_loader)
        if best_val_error is None or val_error <= best_val_error:
            test_error = test(model, test_loader)
            best_val_error = val_error
    return test_error

if __name__ == "__main__":
    start_time=time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None,
                        help='the model, GIN, GNN, 1-2-3-GNN, fingerprint or powerful-GNN')
    parser.add_argument('--num_node',type=int,default=8)
    args = parser.parse_args()
    args.type='regression'
    path = osp.join(osp.dirname(osp.realpath(__file__)), dataset_directory_path, args.model, str(args.num_node))
    config = None
    if args.model =='1-2-3-GNN':
        dataset = RandomGraph(path, args.num_node, pre_transform=PreTransform1())
        class Config:
            def __init__(self):
                self.num_features = dataset.data.x.size()[1]
                self.num_node = dataset.data.edge_index.max() + 1
        config = Config()
        dataset.data.iso_type_2 = torch.unique(dataset.data.iso_type_2, True, True)[1]
        config.num_i_2 = dataset.data.iso_type_2.max().item() + 1
        dataset.data.iso_type_2 = F.one_hot(
            dataset.data.iso_type_2, num_classes=config.num_i_2).to(torch.float)

        dataset.data.iso_type_3 = torch.unique(dataset.data.iso_type_3, True, True)[1]
        config.num_i_3 = dataset.data.iso_type_3.max().item() + 1
        dataset.data.iso_type_3 = F.one_hot(
            dataset.data.iso_type_3, num_classes=config.num_i_3).to(torch.float)

    elif args.model == 'powerful-GNN':
        dataset = RandomGraph(path, args.num_node, pre_transform=PreTransform2())
        class Config:
            def __init__(self):
                self.num_features = dataset.data.x.size()[1]
                self.num_node = dataset.data.edge_index.max() + 1
        config = Config()
    elif args.model == 'fingerprint':
        class Config:
            def __init__(self):
                self.num_node = args.num_node
        config = Config()
        dataset = RandomGraph(path, args.num_node, pre_transform=PreTransform3(config))
        args.kernel_type, args.initialization_type, args.if_learn, args.fingerprint_type = 'gaussian', 'Uniform', 'false', 'GeoScattering'
    elif args.model in ['GNN','GIN']:
        dataset = RandomGraph(path, args.num_node, pre_transform=PreTransform4())
        class Config:
            def __init__(self):
                self.num_features = dataset.data.x.size()[1]
                self.num_node = dataset.data.edge_index.max() + 1
        config = Config()
    else:
        exit(1)
    mid_time=time.time()
    _ = train_process(dataset, args, config)
    end_time=time.time()
    print('peak memory', torch.cuda.max_memory_allocated())
    print('preprocessing time',mid_time-start_time)
    print('total time', end_time-start_time)
    print('training time',end_time-mid_time)
    print('total epoch', num_epochs)
    print('time per epoch',(end_time-mid_time)/num_epochs)

