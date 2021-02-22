import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.data import DataLoader
from sklearn.mixture import GaussianMixture
import numpy as np
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_bin_edges(min_bin, max_bin, num_bin):
    bin_size = (max_bin - min_bin) / (num_bin - 1)
    edges = np.array(
        [min_bin + i * bin_size for i in range(num_bin)])
    return edges


class Net(torch.nn.Module):
    def __init__(self, dataset, args, config):
        super(Net, self).__init__()
        params = json.load(open('models/configs.json'))[args.fingerprint_type]
        self.num_channel = params['num_channel']
        self.num_bin = params['num_bin']
        self.max_bin = params['max_bin']
        self.min_bin = params['min_bin']
        self.blocks = params['blocks']
        hidden_layers = params['hidden_layers']
        self.eval_points = params['num_sample']
        self.conv = torch.nn.Sequential(nn.Conv2d(self.num_channel, self.blocks[0], kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(self.blocks[0]),
                                        nn.ReLU(),
                                        nn.Conv2d(self.blocks[0], self.blocks[1], kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(self.blocks[1]),
                                        nn.ReLU(),
                                        nn.MaxPool2d(stride=(2, 2), kernel_size=2),
                                        nn.Conv2d(self.blocks[1], self.blocks[2], kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(self.blocks[2]),
                                        nn.ReLU(),
                                        nn.MaxPool2d(stride=(2, 2), kernel_size=2))
        if args.type=='TU':
            self.dense = torch.nn.Sequential(
                nn.Linear(int(self.num_bin / 4) * int(self.eval_points / 4) * self.blocks[2], hidden_layers[0]),
                nn.ReLU(),
                nn.Linear(hidden_layers[0], config.num_classes))
        else:
            self.dense = torch.nn.Sequential(
                nn.Linear(int(self.num_bin / 4) * int(self.eval_points / 4) * self.blocks[2], hidden_layers[0]),
                nn.ReLU(),
                nn.Linear(hidden_layers[0], 1))
        self.args = args
        self.sigmoid = nn.Sigmoid()
        self.num_node = config.num_node
        if self.args.initialization_type == 'Uniform':
            self.gaussians = []
            self.sstd = []
            if self.args.kernel_type == 'gaussian':
                initial_std_params = 2
            else:
                initial_std_params = 1
            initial_std = 1 / (initial_std_params * self.num_bin)
            for i in range(self.num_channel):
                self.gaussians.append(
                    torch.FloatTensor(get_bin_edges(self.min_bin, self.max_bin, self.num_bin))[None, :])
                self.sstd.append((torch.ones(self.num_bin) * initial_std)[None, :])
            self.gaussians = torch.cat(self.gaussians, dim=0).repeat(self.eval_points, 1, 1).to(device)
            self.sstd = torch.cat(self.sstd, dim=0).repeat(self.eval_points, 1, 1).to(device)
        elif self.args.initialization_type == 'GMM':
            self.gaussians, self.sstd = self.calculate_params(dataset)
            self.gaussians = torch.from_numpy(self.gaussians).float().to(device)
            self.sstd = torch.from_numpy(self.sstd).float().to(device)

        if self.args.if_learn == 'true':
            self.gaussians = nn.Parameter(self.gaussians, requires_grad=True)
            self.sstd = nn.Parameter(self.sstd, requires_grad=True)

    def Gaussian(self, hks, mean, std):
        f = torch.exp(-(1 / 2) * ((hks - mean) ** 2) / (std ** 2))
        hist = (1 / torch.sqrt(2 * math.pi * (std ** 2))) * f
        return hist

    def Square(self, hks):
        bins_low = torch.tensor(
            [self.min_bin + (self.max_bin - self.min_bin) / self.num_bin * i for i in range(self.num_bin)])[None, None,
                   None, :, None].to(device)
        bins_upper = torch.tensor(
            [self.min_bin + (self.max_bin - self.min_bin) / self.num_bin * (i + 1) for i in range(self.num_bin)])[None,
                     None, None, :, None].to(device)
        hist = (hks >= bins_low) * (hks < bins_upper)
        return hist

    def Silverman(self, hks, mean, std):
        u = (hks - mean) / std
        hist = torch.exp(-torch.abs(u) / math.sqrt(2)) * torch.sin(torch.abs(u) / math.sqrt(2) + math.pi / 4)
        return hist

    def calculate_params(self, dataset):
        loader = DataLoader(dataset, batch_size=1)
        gaussians = np.zeros((self.eval_points, self.num_channel, self.num_bin))
        stds = np.zeros((self.eval_points, self.num_channel, self.num_bin))
        X = []
        for (i, data) in enumerate(loader):
            hks = data.emb.view(-1, self.num_node, self.num_channel, self.eval_points)
            num_node = len(data.edge_index[0])
            X.append(hks.view(-1, self.num_channel, self.eval_points)[:num_node])
            if i == 1000:
                break
        X = np.concatenate(X, axis=0)
        for i in range(self.num_channel):
            for j in range(self.eval_points):
                gm = GaussianMixture(n_components=self.num_bin, random_state=0).fit(X[:, i, j].reshape(-1, 1))
                gaussians[j][i] = np.sort(gm.means_.reshape(self.num_bin))
                stds[j][i] = np.sqrt(gm.covariances_).reshape(self.num_bin)[np.argsort(gm.means_.reshape(self.num_bin))]
        return gaussians, stds

    def calculate_gaussian(self, hks, masks, mean, std):
        mean = mean[:, :, :, None]
        std = std[:, :, :, None]
        hks = hks.transpose(1, 3)[:, :, :, None, :]
        if self.args.kernel_type == 'square':
            hist = self.Square(hks)
        elif self.args.kernel_type == 'gaussian':
            hist = self.Gaussian(hks, mean, std)
        elif self.args.kernel_type == 'silverman':
            hist = self.Silverman(hks, mean, std)
        masks = masks.transpose(1, 2)[:, :, None, :] * torch.ones(len(mean[0]))[:, None].to(device)
        hist = hist * masks[:, :, :, None, :]
        hist = hist.sum(dim=4).transpose(1, 2).transpose(2, 3)
        # if self.args.if_learn:
        #   hist = hist / torch.clamp(hist.sum(2), min=0.0001)[:, :, None, :]
        return hist

    def forward(self, data):
        x = data.emb.view(-1, self.num_node, self.num_channel, self.eval_points)
        masks = data.mask.view(-1, self.num_node, self.eval_points)
        x = self.calculate_gaussian(x, masks, self.gaussians, self.sstd)
        x = self.conv(x)
        x = x.view(-1, int(self.eval_points / 4) * int(self.num_bin / 4) * self.blocks[2])
        x = self.dense(x)
        if self.args.type == "regression":
            return x.view(-1)
        elif self.args.type == "TU":
            return F.log_softmax(x, dim=1)
        else:
            return self.sigmoid(x).view(-1)

