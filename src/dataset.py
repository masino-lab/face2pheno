# src/dataset.py
import os
import glob
import torch
import pandas as pd # <-- ADDED
import numpy as np
from torch_geometric.data import Dataset, Data
from skimage.io import imread
import config, utils

class FacialGraphDataset(Dataset):
    """
    Custom PyTorch Geometric Dataset for facial morphology graphs.
    
    If config.USE_SUPER_NODES is True, this dataset will:
    1.  Load the original 854 triangle patches.
    2.  Combine (average) them into N super-node patches based on 'supernode_group'.
    3.  Build and use a new, smaller edge_index for the super-nodes.
    """
    def __init__(self, root, transform=None, pre_transform=None):
        self.landmark_files = sorted(glob.glob(os.path.join(config.LANDMARK_PIXELS_DIR, "*_landmarks_pixels.txt")))
        self.image_dir = config.IMAGE_DIR
        
        # (Filtering logic is unchanged)
        self.valid_landmark_files = []
        for f in self.landmark_files:
            base_name = os.path.basename(f).replace('_landmarks_pixels.txt', '')
            filename_parts = base_name.split('_')
            if len(filename_parts) > 0 and filename_parts[0].isdigit():
                if config.TASK_TYPE == 'age_classification':
                    self.valid_landmark_files.append(f)
        self.landmark_files = self.valid_landmark_files
        print(f"Found {len(self.landmark_files)} valid landmark files for task '{config.TASK_TYPE}'.")
        self.triangles_df = pd.read_csv(config.TRIANGLE_LIST_PATH)
        
        if config.USE_SUPER_NODES:
            print(f"USE_SUPER_NODES=True. Building coarsened graph with {config.NUM_SUPER_NODES} nodes.")
            # Get the mapping of triangle_index -> group_id
            # We assume triangles_list.csv has triangle_index 0...853
            self.super_node_map = torch.tensor(self.triangles_df['supernode_group'].values, dtype=torch.long)
            
            # Build the new edge_index for the super-nodes
            self.edge_index = self._create_super_node_edge_index()
        else:
            print(f"USE_SUPER_NODES=False. Using original graph with {config.NUM_TRIANGLES} nodes.")
            # Load the original edge_index (this logic is from utils.py)
            adj_df = pd.read_csv(config.ADJACENT_TRIANGLES_PATH)
            u = adj_df.iloc[:, 0].values
            v = adj_df.iloc[:, 1].values
            edge_index_np = np.array([np.concatenate([u, v]), np.concatenate([v, u])])
            self.edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        # --- End of update ---

        super().__init__(root, transform, pre_transform)

    def _create_super_node_edge_index(self):
        """
        Builds a new edge_index for the super-nodes based on the
        original triangle adjacency.
        """
        adj_df = pd.read_csv(config.ADJACENT_TRIANGLES_PATH)
        
        super_edges = set()
        
        # Iterate over every edge (u, v) in the 854-node graph
        for _, row in adj_df.iterrows():
            u, v = row['triangle_index_1'], row['triangle_index_2']
            
            # Find the super-node group for each triangle
            group_u = self.super_node_map[u]
            group_v = self.super_node_map[v]
            
            # If they are in different groups, add an edge between the groups
            if group_u != group_v:
                # Add in a sorted way to avoid duplicates like (1, 2) and (2, 1)
                edge = tuple(sorted((group_u.item(), group_v.item())))
                super_edges.add(edge)
        
        # Convert the set of tuples to the [2, num_edges] format
        if not super_edges:
            return torch.empty((2, 0), dtype=torch.long)
            
        edge_list_np = np.array(list(super_edges)).T
        
        # Make the graph undirected
        u = edge_list_np[0, :]
        v = edge_list_np[1, :]
        edge_index_np = np.array([np.concatenate([u, v]), np.concatenate([v, u])])
        
        print(f"Created super-node graph with {len(super_edges)} unique edges.")
        return torch.tensor(edge_index_np, dtype=torch.long)


    def len(self):
        return len(self.landmark_files)

    def _get_age_label(self, age):
        # (This function is unchanged)
        if config.CLASSIFICATION_TYPE == 'binary':
            threshold = config.AGE_THRESHOLDS[0]
            label = 1 if age >= threshold else 0
        elif config.CLASSIFICATION_TYPE == 'multi-class':
            label = np.digitize(age, config.AGE_THRESHOLDS, right=False)
        else:
            raise ValueError(f"Unknown CLASSIFICATION_TYPE: {config.CLASSIFICATION_TYPE}")
        return label

    def get(self, idx):
        landmark_file = self.landmark_files[idx]
        base_name = os.path.basename(landmark_file).replace('_landmarks_pixels.txt', '')
        image_file = os.path.join(self.image_dir, f"{base_name}.jpg")

        if not os.path.exists(image_file):
            return None
            
        try:
            image = imread(image_file)
        except Exception:
            return None
        
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[2] == 4:
            image = image[:, :, :3]
        image = image.astype(np.float32) / 255.0
        
        # --- Node Feature Extraction (Original) ---
        landmarks = utils.load_landmarks(landmark_file) 
        original_patches = []
        for i, row in self.triangles_df.iterrows():
            node_indices = [row['node_1'], row['node_2'], row['node_3']]
            try:
                triangle_vertices = [landmarks[i] for i in node_indices]
            except KeyError as e:
                print(f"Warning: Landmark index {e} not found in {landmark_file}. Skipping.")
                return None
            patch = utils.extract_superpixel_patch(image, triangle_vertices)
            original_patches.append(patch)
            
        node_features_np = np.stack(original_patches, axis=0) # Shape: [854, 32, 32, 3]
        
        # --- UPDATED: Combine patches into super-nodes if enabled ---
        if config.USE_SUPER_NODES:
            # Create an empty array for the super-node patches
            super_node_features_np = np.zeros(
                (config.NUM_SUPER_NODES, config.PATCH_SIZE, config.PATCH_SIZE, config.IMAGE_CHANNELS),
                dtype=np.float32
            )
            
            # Loop from 0 to 9 (for 10 groups)
            for k in range(config.NUM_SUPER_NODES):
                # Find the indices of all triangles belonging to this group
                indices = (self.super_node_map == k).nonzero(as_tuple=True)[0]
                
                if len(indices) > 0:
                    # Get all patches for this group
                    patches_for_group_k = node_features_np[indices]
                    
                    # Average them to create the super-patch
                    super_patch_k = np.mean(patches_for_group_k, axis=0)
                    super_node_features_np[k] = super_patch_k
                else:
                    # If a group has 0 triangles (shouldn't happen, but safe to check)
                    # It will just be a patch of zeros
                    pass 
            
            # The final features are the super-node features
            final_features_np = super_node_features_np # Shape: [10, 32, 32, 3]
        
        else:
            # Use the original 854 patches
            final_features_np = node_features_np # Shape: [854, 32, 32, 3]
        # --- End of update ---

        # Convert to tensor. This works for both [10, 32, 32, 3] and [854, 32, 32, 3]
        x = torch.from_numpy(final_features_np).permute(0, 3, 1, 2)
        
        # --- Label Generation (Unchanged) ---
        y = None
        if config.TASK_TYPE == 'age_classification':
            try:
                age_str = base_name.split('_')[0]
                age = int(age_str)
                label = self._get_age_label(age)
                y = torch.tensor(label, dtype=torch.long)
            except (ValueError, IndexError):
                return None
        
        if y is None:
            return None

        # self.edge_index is now the correct one (either super-node or original)
        graph_data = Data(x=x, edge_index=self.edge_index, y=y)
        return graph_data