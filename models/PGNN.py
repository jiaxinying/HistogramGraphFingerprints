'''
This file is largely adapted from https://github.com/hadarser/ProvablyPowerfulGraphNetworks_torch
'''
import torch
import powerful_utils.layers.layers as layers
import powerful_utils.layers.modules as modules
from powerful_utils.utils.config import process_config
import torch.nn as nn

config = process_config('powerful_utils/configs/k_regular_config.json', dataset_name='QM9')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self, dataset_config,args):
        """
        Build the model computation graph, until scores/values are returned at the end
        """
        super().__init__()
        self.config = config
        self.num_node = dataset_config.num_node
        use_new_suffix = False  # True or False
        block_features = config.architecture.block_features  # List of number of features in each regular block
        original_features_num = config.node_labels + 1  # Number of features of the input

        # First part - sequential mlp blocks
        last_layer_features = original_features_num
        self.reg_blocks = nn.ModuleList()
        for layer, next_layer_features in enumerate(block_features):
            mlp_block = modules.RegularBlock(config, last_layer_features, next_layer_features)
            self.reg_blocks.append(mlp_block)
            last_layer_features = next_layer_features

        # Second part
        self.fc_layers = nn.ModuleList()
        if use_new_suffix:
            for output_features in block_features:
                # each block's output will be pooled (thus have 2*output_features), and pass through a fully connected
                fc = modules.FullyConnected(2 * output_features, self.config.num_classes, activation_fn=None)
                self.fc_layers.append(fc)

        else:  # use old suffix
            # Sequential fc layers
            self.fc_layers.append(modules.FullyConnected(2 * block_features[-1], 512))
            self.fc_layers.append(modules.FullyConnected(512, 256))
            self.fc_layers.append(modules.FullyConnected(256, self.config.num_classes, activation_fn=None))

    def forward(self, input):
        x = input.graph.to(device).view(-1, 1, self.num_node, self.num_node)
        scores = torch.tensor(0, device=device, dtype=x.dtype)

        for i, block in enumerate(self.reg_blocks):

            x = block(x)

            if self.config.architecture.new_suffix:
                # use new suffix
                scores = self.fc_layers[i](layers.diag_offdiag_maxpool(x)) + scores

        if not self.config.architecture.new_suffix:
            # old suffix
            x = layers.diag_offdiag_maxpool(x)  # NxFxMxM -> Nx2F
            for fc in self.fc_layers:
                x = fc(x)
            scores = x

        return scores.view(-1)