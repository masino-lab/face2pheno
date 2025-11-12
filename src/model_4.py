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
        
        # --- Store num_nodes for use in the forward pass ---
        self.num_nodes = num_nodes 
        
        # --- CNN Feature Extractor ---
        class CNNFeatureExtractor(nn.Module):
            def __init__(self, in_channels, out_features):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, config.NUM_KERNELS, kernel_size=3, stride=1, padding=1)
                self.relu1 = nn.ReLU(inplace=True)
                self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
                self.flatten = nn.Flatten()
                self.linear1 = nn.Linear(config.NUM_KERNELS * (config.PATCH_SIZE // 2) * (config.PATCH_SIZE // 2), out_features)
                self.relu2 = nn.ReLU(inplace=True)
            
            def forward(self, x):
                conv_activations = self.conv1(x)
                x = self.relu1(conv_activations)
                x = self.pool1(x)
                x = self.flatten(x)
                x = self.linear1(x)
                final_features = self.relu2(x)
                return final_features, conv_activations

        # This nested class is unchanged
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
        
       
        # self.cnn_feature_extractor = CNNFeatureExtractor(input_channels, feature_dim) # Previous
        
        # Create N unique CNN feature extractors
        self.cnn_feature_extractors = nn.ModuleList([
            CNNFeatureExtractor(input_channels, feature_dim) 
            for _ in range(num_nodes)
        ])
        
        self.gcn = GCNConv(feature_dim, gcn_hidden_dim)
        self.mlp = MLP(gcn_hidden_dim, output_dim)

    def forward(self, data, return_activations=False):
        """
        Main forward pass for the entire GNN model.

        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # We can no longer process all patches 'x' at once.
        # We must apply the correct CNN to the correct patch.
        
        # 1. Get the local node index for each patch in the batch
        # This tells us if a patch is node 0, node 1, ... node N-1
        local_indices = torch.arange(len(batch), device=x.device) - data.ptr[batch]

        # 2. Create empty tensors to store the results
        # To ensure the results are in the *original order* for the GCN
        node_representations = torch.empty(x.size(0), self.gcn.in_channels, device=x.device)
        
        # Pre-calculate activation/kernel shapes if needed for efficiency
        # (This assumes H and W are config.PATCH_SIZE)
        act_shape = (x.size(0), config.NUM_KERNELS, config.PATCH_SIZE, config.PATCH_SIZE)
        all_activations = torch.empty(act_shape, device=x.device)

        # 3. Loop through each *node type* (0 to N-1)
        for j in range(self.num_nodes):
            # Find all patches in the batch that correspond to node 'j'
            mask = (local_indices == j)
            
            # If any such patches exist in this batch
            if mask.any():
                # Get the specific CNN for node 'j'
                cnn = self.cnn_feature_extractors[j]
                
                # Get all patches for node 'j'
                patches_for_node_j = x[mask]
                
                # Apply the specific CNN
                features_j, activations_j = cnn(patches_for_node_j)
                
                # Place the results into the correct spots in the output tensors
                node_representations[mask] = features_j
                all_activations[mask] = activations_j


        # 4. Graph Convolution (This part is unchanged)
        # The GCN now operates on the 'node_representations' just as before.
        h = self.gcn(node_representations, edge_index)
        h = F.relu(h)
        
        # 5. Aggregate (This part is unchanged)
        graph_representation = global_mean_pool(h, batch)
        
        # 6. Classify (This part is unchanged)
        output = self.mlp(graph_representation)

        # Kernel and Activation handling 
        if return_activations:
            # 'activations' is now the tensor we built in the loop
            activations = all_activations 
            
            # 'kernels' is now a stack of kernels from *all* CNNs
            kernels = torch.stack([
                cnn.conv1.weight.data.clone() 
                for cnn in self.cnn_feature_extractors
            ])
            
            return output, activations, kernels
        else:
            return output
    