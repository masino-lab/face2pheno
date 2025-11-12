# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import config

class FacialGNN(nn.Module):
    """
    CNN to extract features from node patches, a GCN for message passing,
    and an MLP for the final classification.
    """
    def __init__(self, num_nodes, input_channels, feature_dim=64, gcn_hidden_dim=128, output_dim=2):
        super(FacialGNN, self).__init__()
        class CNNFeatureExtractor(nn.Module):

            def __init__(self, in_channels, out_features):
                super().__init__()
                # Define layers individually to allow capturing intermediate outputs.
                self.conv1 = nn.Conv2d(in_channels, config.NUM_KERNELS, kernel_size=3, stride=1, padding=1)
                self.relu1 = nn.ReLU(inplace=True)
                self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Halves height and width
                self.flatten = nn.Flatten()
                # The input size for the linear layer is calculated based on the output of the pooling layer.
                self.linear1 = nn.Linear(config.NUM_KERNELS * (config.PATCH_SIZE // 2) * (config.PATCH_SIZE // 2), out_features) # Shape: num_output_channels * (height // 2) * (width // 2)
                self.relu2 = nn.ReLU(inplace=True)
            
            def forward(self, x):
                """
                Forward pass for the CNN : returns both the final feature vector and the raw activations from the first conv layer for.
                """
                conv_activations = self.conv1(x)   ##
                # Pass data through the remaining layers.
                x = self.relu1(conv_activations)
                x = self.pool1(x)
                x = self.flatten(x)
                x = self.linear1(x)
                final_features = self.relu2(x)
                return final_features, conv_activations

        # Nested class for MLP Classifier 
        class MLP(nn.Module):
            def __init__(self, in_features, out_features): 
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(in_features, in_features // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(in_features // 2, out_features)
                )
            def forward(self, x):
                return self.mlp(x)
        
        self.cnn_feature_extractor = CNNFeatureExtractor(input_channels, feature_dim)
        self.gcn = GCNConv(feature_dim, gcn_hidden_dim)
        self.mlp = MLP(gcn_hidden_dim, output_dim)

    def forward(self, data, return_activations=False):
        """
        Main forward pass for the entire GNN model.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Node Feature Extraction: Get initial node features by passing each patch through the CNN.
        node_representations, activations = self.cnn_feature_extractor(x)
        # Graph Convolution: Perform message passing to update node features with neighborhood info.
        h = self.gcn(node_representations, edge_index)
        h = F.relu(h)
        # Aggregate all node features into a single graph-level representation.
        graph_representation = global_mean_pool(h, batch)
        output = self.mlp(graph_representation)
        kernels = self.cnn_feature_extractor.conv1.weight.data.clone()  # Capture kernel weights from the first conv layer.
        # return activations
        if return_activations:                      #this is false during training
            return output, activations, kernels
        else:
            return output