# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import config

class FacialGNN(nn.Module):
    def __init__(self, num_nodes, input_channels, feature_dim=64, gcn_hidden_dim=128, output_dim=2): # output_dim=2 for binary classification
        super(FacialGNN, self).__init__()
        # Simple CNN feature extractor:
        # One conv, relu, pooling, flatten, then linear to feature_dim
        self.cnn_feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # halves patch size
            nn.Flatten(),
            nn.Linear(8 * (config.PATCH_SIZE // 2) * (config.PATCH_SIZE // 2), feature_dim), #fully connected dense layer, each image patch is now represented as a vector of size feature_dim (3416*64)
            nn.ReLU(inplace=True)
        )

        # GCN layer
        self.gcn = GCNConv(feature_dim, gcn_hidden_dim) #size 64x128

        # MLP classifier
        self.mlp = nn.Sequential(
            nn.Linear(gcn_hidden_dim, gcn_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(gcn_hidden_dim // 2, output_dim)
        )

    #     # --- Register the hook --- not working !!!!!
    #     # self.cnn_feature_extractor[0] is the Conv2d layer
    #     self.cnn_feature_extractor[0].register_forward_hook(self.save_activations_hook())

    # def save_activations_hook(self):
    #     def hook(module, input, output):
    #         self.conv_activations = output
    #     return hook

    def forward(self, data):

        # data.x: image patches shaped (N_nodes, C, H, W)
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Extract per-node (patch) features
        node_representations = self.cnn_feature_extractor(x)  # (N_nodes, feature_dim)

        # Message passing 
        h = self.gcn(node_representations, edge_index)
        h = F.relu(h)

        # Pool node features into graph-level representation
        graph_representation = global_mean_pool(h, batch)     # (batch_size, gcn_hidden_dim) #pool all node features into a single graph feature

        # Classify
        output = self.mlp(graph_representation)               # (batch_size, output_dim)
        return output
