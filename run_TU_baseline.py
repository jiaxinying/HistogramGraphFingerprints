import os.path as osp
from torch_geometric.datasets import TUDataset
import torch
import numpy as np
from torch_geometric.utils import to_dense_adj
import networkx
from karateclub import FeatherGraph
from karateclub import NetLSD
from karateclub import GeoScattering
import networkx as nx
from torch_geometric.data import DataLoader
from sklearn.linear_model import LogisticRegression
import argparse
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
dataset_directory_path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'baseline')
embedding_size = {'NetLSD': 250, 'GeoScattering': 111, 'FEATHER': 500}


def connected_component_subgraphs(G):
    return [G.subgraph(c) for c in nx.connected_components(G)]


def to_connected(cur_graph):
    sub_graphs = connected_component_subgraphs(cur_graph)
    main_graph = sub_graphs[0]
    for sg in sub_graphs:
        if len(sg.nodes()) > len(main_graph.nodes()):
            main_graph = sg
    A = nx.to_numpy_matrix(main_graph)
    return nx.convert_matrix.from_numpy_matrix(A)


class PreTransform(object):
    def __init__(self, args):
        if args.fingerprint_type == 'NetLSD':
            self.model = NetLSD()
        elif args.fingerprint_type == 'GeoScattering':
            self.model = GeoScattering()
        elif args.fingerprint_type == 'FEATHER':
            self.model = FeatherGraph()

    def __call__(self, data):
        adj_mat = np.array(to_dense_adj(data.edge_index)[0])
        graph = networkx.convert_matrix.from_numpy_matrix(adj_mat)
        if not nx.is_connected(graph):
            graph = to_connected(graph)
        self.model.fit([graph])
        data.emb = torch.from_numpy(self.model.get_embedding()).float()
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='the name of the TU dataset')
    parser.add_argument('--fingerprint_type', type=str, default=None,
                        help='the node fingerprint type, NetLSD or GeoScattering or FEATHER')
    args = parser.parse_args()
    path = osp.join(osp.dirname(osp.realpath(__file__)), dataset_directory_path, args.fingerprint_type)
    dataset = TUDataset(path, name=args.dataset, pre_transform=PreTransform(args))

    perm = torch.randperm(len(dataset), dtype=torch.long)
    dataset = dataset[perm]
    acc = []
    for i in range(10):
        test_mask = torch.zeros(len(dataset), dtype=torch.uint8)
        n = len(dataset) // 10
        test_mask[i * n:(i + 1) * n] = 1
        test_dataset = dataset[test_mask]
        train_dataset = dataset[1 - test_mask]
        train_loader = DataLoader(train_dataset)
        test_loader = DataLoader(test_dataset)
        X_train = []
        X_test = []
        Y_train = []
        Y_test = []
        for data in train_loader:
            X_train.append(data.emb.tolist())
            Y_train.append(data.y.tolist())
        for data in test_loader:
            X_test.append(data.emb.tolist())
            Y_test.append(data.y.tolist())
        X_train, X_test, Y_train, Y_test = np.array(X_train).reshape(-1,
                                                                     embedding_size[args.fingerprint_type]), np.array(
            X_test).reshape(-1, embedding_size[args.fingerprint_type]), np.array(Y_train).reshape(-1), np.array(
            Y_test).reshape(-1)
        clf = LogisticRegression(random_state=0,max_iter=300).fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        acc.append((Y_pred == Y_test).mean())
    acc = np.array(acc)
    print(acc.mean())
    print(acc.std())
