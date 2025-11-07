# src/dataset.py
import os
import glob
import torch
import numpy as np
from torch_geometric.data import Dataset, Data
from skimage.io import imread
import config, utils

class FacialGraphDataset(Dataset):
    """
    Custom PyTorch Geometric Dataset for facial morphology graphs.
    """
    def __init__(self, root, transform=None, pre_transform=None):
        self.landmark_files = sorted(glob.glob(os.path.join(config.LANDMARK_PIXELS_DIR, "*_landmarks_pixels.txt")))
        self.image_dir = config.IMAGE_DIR
        
        # Filter out landmark files that don't have a corresponding image filename structure
        self.valid_landmark_files = []
        for f in self.landmark_files:
            # Assumes UTKFace format: [age]_[gender]_[race]_[date&time].jpg
            base_name = os.path.basename(f).replace('_landmarks_pixels.txt', '')
            filename_parts = base_name.split('_')
            
            # Check if the filename starts with an age (number)
            if len(filename_parts) > 0 and filename_parts[0].isdigit():
                # Check if task is age-based
                if config.TASK_TYPE == 'age_classification':
                    self.valid_landmark_files.append(f)
                # (Future) Add other checks, e.g., if task is emotion, check for emotion tag
            
        self.landmark_files = self.valid_landmark_files
        print(f"Found {len(self.landmark_files)} valid landmark files for task '{config.TASK_TYPE}'.")

        self.triangles_df, self.edge_index = utils.get_static_graph_data()
        super().__init__(root, transform, pre_transform)

    def len(self):
        return len(self.landmark_files)

    def _get_age_label(self, age):
        """Helper function to determine label based on config."""
        
        if config.CLASSIFICATION_TYPE == 'binary':
            # Assumes the first threshold is the binary cutoff
            # Label 0: < threshold, Label 1: >= threshold
            threshold = config.AGE_THRESHOLDS[0]
            label = 1 if age >= threshold else 0
            
        elif config.CLASSIFICATION_TYPE == 'multi-class':
            # Uses numpy.digitize to find the correct bin
            # thresholds = [12, 18, 30]
            # age < 12 -> bin 0
            # 12 <= age < 18 -> bin 1
            # 18 <= age < 30 -> bin 2
            # age >= 30 -> bin 3
            label = np.digitize(age, config.AGE_THRESHOLDS, right=False)
            
        else:
            raise ValueError(f"Unknown CLASSIFICATION_TYPE: {config.CLASSIFICATION_TYPE}")
            
        return label

    def get(self, idx):
        landmark_file = self.landmark_files[idx]
        base_name = os.path.basename(landmark_file).replace('_landmarks_pixels.txt', '')
        image_file = os.path.join(self.image_dir, f"{base_name}.jpg")

        if not os.path.exists(image_file):
            print(f"Warning: Image file not found: {image_file}. Skipping.")
            return None
            
        try:
            image = imread(image_file)
        except Exception as e:
            print(f"Warning: Could not read image {image_file}: {e}. Skipping.")
            return None
        
        # --- Image Preprocessing ---
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[2] == 4:
            image = image[:, :, :3]
            
        image = image.astype(np.float32) / 255.0
        
        # --- Node Feature Extraction ---
        landmarks = utils.load_landmarks(landmark_file) 
        node_features = []
        for _, row in self.triangles_df.iterrows():
            node_indices = [row['node_1'], row['node_2'], row['node_3']]
            try:
                triangle_vertices = [landmarks[i] for i in node_indices]
            except KeyError as e:
                # This check is critical for data integrity
                print(f"Warning: Landmark index {e} not found in {landmark_file}. Skipping.")
                return None

            patch = utils.extract_superpixel_patch(image, triangle_vertices)
            node_features.append(patch)
            
        node_features_np = np.stack(node_features, axis=0)
        x = torch.from_numpy(node_features_np).permute(0, 3, 1, 2)
        
        # --- Label Generation ---
        y = None
        if config.TASK_TYPE == 'age_classification':
            try:
                age_str = base_name.split('_')[0]
                age = int(age_str)
                label = self._get_age_label(age)
                y = torch.tensor(label, dtype=torch.long)
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse age from filename: {base_name}: {e}. Skipping.")
                return None
        
        # (Future) Add elif for 'emotion_detection' here
        
        if y is None:
            print(f"Warning: No label generated for {base_name}. Skipping.")
            return None

        graph_data = Data(x=x, edge_index=self.edge_index, y=y)
        return graph_data
