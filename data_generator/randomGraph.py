import numpy as np
import torch
import random
from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.utils.sparse as sp
import networkx as nx
np.random.seed(0)
random.seed(0)

def generate_graph(n, seed):
    rand_graph = nx.fast_gnp_random_graph(n=n,p=0.2,seed=seed)
    A = nx.to_numpy_matrix(rand_graph)
    num_triangles = sum(nx.triangles(rand_graph).values()) / 3
    return A, num_triangles


class RandomGraph(InMemoryDataset):
    def __init__(self, root, num_node, transform=None, pre_transform=None):
        self.num_node = num_node
        super(RandomGraph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        return 'k_cycle.pt'

    @property
    def processed_file_names(self):
        return 'k_cycle.pt'

    def download(self):
        data_list = []
        seed=0
        for i in range(1000):
            data = Data()
            adj_matrix, num_triangles = generate_graph(n=self.num_node, seed=seed)
            seed+=1
            while(adj_matrix.sum()==0):
                seed+=1
                adj_matrix, num_triangles = generate_graph(n=self.num_node,seed=seed)
            adj_matrix=torch.from_numpy(adj_matrix).float()
            data.y = torch.tensor([num_triangles]).view(-1,1)
            edge_index = sp.dense_to_sparse(adj_matrix)[0]
            data.edge_index = edge_index
            data.name = i
            data.num_nodes = len(adj_matrix)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.raw_paths[0])

    def process(self):
        self.data, self.slices = torch.load(self.raw_paths[0])
        data_list = [self.get(i) for i in range(len(self))]

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return
