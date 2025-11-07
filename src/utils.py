# src/utils.py
import os
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn.functional as F # <-- ADDED
import json
from skimage.io import imread
from skimage.transform import resize
import config

def load_landmarks(landmark_file_path):
    """
    Loads landmark pixel coordinates from a text file into a dictionary.
    Assumes format: 'landmark_index,x,y'
    
    Args:
        landmark_file_path (str): The path to the landmark file.
        
    Returns:
        dict: A dictionary mapping landmark index (int) to (x, y) coordinates (tuple).
    """
    landmarks = {}
    with open(landmark_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 3:
                try:
                    node_idx = int(parts[0])
                    x = int(parts[1])
                    y = int(parts[2])
                    landmarks[node_idx] = (x, y)
                except ValueError:
                    print(f"Warning: Skipping malformed line in {landmark_file_path}: {line}")
            else:
                 print(f"Warning: Skipping malformed line in {landmark_file_path}: {line}")
    return landmarks

def get_static_graph_data():
    """
    Loads the static (shared across all images) graph information:
    the triangle definitions and the adjacency matrix (edge index).
    
    Returns:
        pd.DataFrame: DataFrame with triangle node indices.
        torch.Tensor: The edge index for PyTorch Geometric, shape [2, num_edges].
    """
    # Load triangle definitions
    triangles_df = pd.read_csv(config.TRIANGLE_LIST_PATH)
    
    # Load adjacency list
    adj_df = pd.read_csv(config.ADJACENT_TRIANGLES_PATH)
    
    u = adj_df.iloc[:, 0].values
    v = adj_df.iloc[:, 1].values
    
    edge_index = np.array([np.concatenate([u, v]), np.concatenate([v, u])])
    
    return triangles_df, torch.tensor(edge_index, dtype=torch.long)

def extract_superpixel_patch(image, triangle_vertices):
    """
    Extracts a fixed-size pixel patch (superpixel) centered at the
    centroid of a given triangle.
    
    Args:
        image (np.array): The full facial image.
        triangle_vertices (list of tuples): A list of three (x, y) coordinates for the triangle's vertices.
        
    Returns:
        np.array: A square patch of pixels of size [PATCH_SIZE, PATCH_SIZE, CHANNELS].
    """
    # Calculate the centroid of the triangle
    centroid = np.mean(triangle_vertices, axis=0).astype(int)
    cx, cy = centroid
    
    patch_half = config.PATCH_SIZE // 2
    
    # Define the bounding box for the patch
    top, bottom = cy - patch_half, cy + patch_half
    left, right = cx - patch_half, cx + patch_half
    
    num_channels = image.shape[2]
    
    padded_image = np.pad(
        image, 
        pad_width=((patch_half, patch_half), (patch_half, patch_half), (0, 0)), 
        mode='constant', 
        constant_values=0
    )
    
    # Adjust coordinates for the padded image
    padded_top, padded_bottom = top + patch_half, bottom + patch_half
    padded_left, padded_right = left + patch_half, right + patch_half
    
    # Extract the patch
    patch = padded_image[padded_top:padded_bottom, padded_left:padded_right, :]
    
    if patch.size == 0:
        return np.zeros(
            (config.PATCH_SIZE, config.PATCH_SIZE, num_channels),
            dtype=np.float32
        )

    if patch.shape[0] != config.PATCH_SIZE or patch.shape[1] != config.PATCH_SIZE:
        patch = resize(
            patch,
            (config.PATCH_SIZE, config.PATCH_SIZE, num_channels),
            preserve_range=True,
            anti_aliasing=True
        )
            
    return patch.astype(np.float32)

# --- AGGREGATE STATS FUNCTIONS ---

def collect_activation_stats(model, test_loader, num_patches, num_kernels, num_classes, device):
    """
    Runs the model in evaluation mode and collects activation statistics
    AGGREGATED across all images.
    
    Returns:
        tuple: (stats_data, y_true, y_scores)
            - stats_data (list): 3D list [class][patch][kernel] containing lists of activations.
            - y_true (list): Ground truth labels.
            - y_scores (list): Model prediction scores (format depends on task).
    """
    stats_data = [
        [
            [[] for _ in range(num_kernels)] for _ in range(num_patches)
        ] 
        for _ in range(num_classes)
    ]

    y_true = []
    y_scores = []

    with torch.no_grad():
        for batch_data in test_loader:
            if batch_data is None: continue
            batch_data = batch_data.to(device)
            
            out, activations, kernels = model(batch_data, return_activations=True)
            
            max_activations = torch.amax(activations, dim=(2, 3)) 
            graph_labels = batch_data.y
            local_indices = torch.arange(len(batch_data.batch)).to(device) - batch_data.ptr[batch_data.batch]
            patch_labels = graph_labels[batch_data.batch]

            max_activations_cpu = max_activations.cpu()
            local_indices_cpu = local_indices.cpu()
            patch_labels_cpu = patch_labels.cpu()

            for i in range(len(patch_labels_cpu)):
                class_idx = patch_labels_cpu[i].item()
                patch_idx = local_indices_cpu[i].item()

                # Safety check
                if class_idx >= num_classes or patch_idx >= num_patches:
                    print(f"Warning: Skipping out-of-bounds index: class {class_idx}, patch {patch_idx}")
                    continue
                
                activations_for_this_patch = max_activations_cpu[i]
                
                for kernel_idx in range(num_kernels):
                    activation_val = activations_for_this_patch[kernel_idx].item()
                    stats_data[class_idx][patch_idx][kernel_idx].append(activation_val)
            
            # --- UPDATED: Handle score collection for binary vs multi-class ---
            scores = F.softmax(out, dim=1)
            
            if config.CLASSIFICATION_TYPE == 'binary':
                # For ROC/PR, we need the score of the positive class (class 1)
                y_scores.extend(scores[:, 1].cpu().numpy())
            else:
                # For Confusion Matrix, we need the full probability array
                y_scores.extend(scores.cpu().numpy())
                
            y_true.extend(batch_data.y.cpu().numpy())
            # --- End of update ---

    return stats_data, y_true, y_scores

def calculate_mean_variance(stats_data, num_patches, num_kernels, num_classes):
    """
    Calculates the mean, variance, and sample size from the AGGREGATED activation data.
    
    Returns:
        dict: A dictionary with the calculated statistics.
    """
    results = {f'class_{i}': {} for i in range(num_classes)}

    for class_idx in range(num_classes):
        class_key = f'class_{class_idx}'
        for patch_idx in range(num_patches):
            patch_key = f'patch_{patch_idx}'
            results[class_key][patch_key] = {}
            
            for kernel_idx in range(num_kernels):
                kernel_key = f'kernel_{kernel_idx}'
                
                values = stats_data[class_idx][patch_idx][kernel_idx]
                
                if len(values) > 0:
                    mean_val = float(np.mean(values))
                    var_val = float(np.var(values)) 
                    sample_size = len(values)
                else:
                    mean_val = None
                    var_val = None
                    sample_size = 0
                
                results[class_key][patch_key][kernel_key] = {
                    'mean': mean_val,
                    'variance': var_val,
                    'n': sample_size
                }
    return results

# --- PER-IMAGE RECORDING FUNCTION ---

def record_image_activations(model, test_loader, device):
    """
    Records the raw activation values for each patch for each image.
    This function is compatible with both binary and multi-class.
    """
    all_image_records = []
    global_image_index = 0 # Tracks the index of the image in the test set

    with torch.no_grad():
        for batch_data in test_loader:
            if batch_data is None: continue
            batch_data = batch_data.to(device)
            
            out, activations, _ = model(batch_data, return_activations=True)
            
            # Get max activations (shape: [TotalPatchesInBatch, NumKernels])
            max_activations = torch.amax(activations, dim=(2, 3)) 
            
            # Get graph-level info
            graph_labels = batch_data.y # Shape: [NumGraphsInBatch]
            num_graphs_in_batch = len(graph_labels)
            
            # Get patch-level info
            local_indices = torch.arange(len(batch_data.batch)).to(device) - batch_data.ptr[batch_data.batch]
            patch_ptr = batch_data.ptr # (start and end index for each graph's patches)

            # Move to CPU once for efficiency
            max_activations_cpu = max_activations.cpu()
            local_indices_cpu = local_indices.cpu()

            # Iterate through each GRAPH in the batch
            for i in range(num_graphs_in_batch):
                image_id = global_image_index + i
                class_label = graph_labels[i].item() # This works for multi-class too
                
                # Find the slice of patches belonging to this graph
                start_patch_idx = patch_ptr[i]
                end_patch_idx = patch_ptr[i+1]
                num_patches_for_this_graph = end_patch_idx - start_patch_idx

                image_record = {
                    "image_id": image_id,
                    "class_label": class_label,
                    "activations": []
                }
                
                # Get the activation and index data just for this one graph
                patches_activations = max_activations_cpu[start_patch_idx:end_patch_idx]
                patches_local_indices = local_indices_cpu[start_patch_idx:end_patch_idx]

                # Iterate through each PATCH in this one graph
                for j in range(num_patches_for_this_graph):
                    patch_idx = patches_local_indices[j].item()
                    
                    # .tolist() converts the tensor row to a simple Python list
                    kernel_activations_list = patches_activations[j].tolist() 
                    
                    patch_record = {
                        "patch_idx": patch_idx,
                        "kernel_activations": kernel_activations_list
                    }
                    image_record["activations"].append(patch_record)

                # Sort the patch records by patch_idx for consistency
                image_record["activations"].sort(key=lambda p: p['patch_idx'])
                all_image_records.append(image_record)

            # Update the global counter
            global_image_index += num_graphs_in_batch

    return all_image_records

