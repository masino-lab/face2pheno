import config
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
# import torch.nn.functional as F # No longer needed here
from dataset import FacialGraphDataset
from model_2 import FacialGNN
from evaluation import plot_curves
from datetime import datetime
import numpy as np  
import json
import os
# --- UPDATED IMPORTS ---
from utils import (
    collect_activation_stats, 
    calculate_mean_variance, 
    record_image_activations
)

def main():
    print(f"Using device: {config.DEVICE}")
    # --- NEW: Print task configuration ---
    print(f"Running task: {config.TASK_TYPE}")
    print(f"Classification type: {config.CLASSIFICATION_TYPE} ({config.NUM_CLASSES} classes)")
    if config.TASK_TYPE == 'age_classification':
        print(f"Age Thresholds: {config.AGE_THRESHOLDS}")

    # 1. Load the full dataset
    # IMPORTANT: Your FacialGraphDataset class must now read 
    # config.CLASSIFICATION_TYPE and config.AGE_THRESHOLDS 
    # to generate the correct 'y' labels.
    full_dataset = FacialGraphDataset(root=f"{config.BASE_DATA_PATH}processed/pyg_dataset")
    if len(full_dataset) == 0:
        print("Error: The dataset is empty.")
        return
    
    # 2. Split dataset into training and testing sets
    test_size = int(len(full_dataset) * config.TEST_SPLIT_RATIO)
    train_size = len(full_dataset) - test_size
    
    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)
    
    print(f"Dataset split into {len(train_dataset)} training samples and {len(test_dataset)} testing samples.")

    # 3. Create DataLoaders for each set
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 4. Initialize Model, Optimizer, and Loss Function
    # --- UPDATED ---
    # The model's output dimension is now set dynamically from config
    model = FacialGNN(
        num_nodes=config.NUM_SUPER_NODES if config.USE_SUPER_NODES else config.NUM_TRIANGLES,
        input_channels=config.IMAGE_CHANNELS,
        output_dim=config.NUM_CLASSES 
    ).to(config.DEVICE)
    
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = CrossEntropyLoss()
    
    # --- 5. Training Phase ---
    print("Starting training...")
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_data in train_loader:
            if batch_data is None: continue
            batch_data = batch_data.to(config.DEVICE)
            
            optimizer.zero_grad()
            out = model(batch_data)
            loss = criterion(out, batch_data.y)
            loss.backward()
            optimizer.step()        
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1:03d}/{config.EPOCHS}, Training Loss: {avg_loss:.6f}")
        
    print("Training finished.")


    # ####################################################################################
    # --- 6. Model Evaluation and Aggregate Statistics ---
    # ####################################################################################
    print("\nStarting evaluation for aggregate statistics...")
    model.eval()
    NUM_PATCHES = 854 
    NUM_KERNELS = config.NUM_KERNELS  
    NUM_CLASSES = config.NUM_CLASSES # Now read from config
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Call the first utility function to collect raw stats
    # NOTE: Your 'utils.py' functions must be able to handle multi-class labels/scores
    stats_data, y_true, y_scores = collect_activation_stats(
        model, 
        test_loader, 
        NUM_PATCHES, 
        NUM_KERNELS, 
        NUM_CLASSES, 
        config.DEVICE
    )

    print("Evaluation complete. Calculating aggregate statistics...")

    # Call the second utility function to process the stats
    results = calculate_mean_variance(
        stats_data, 
        NUM_PATCHES, 
        NUM_KERNELS, 
        NUM_CLASSES
    )

    # # Save the aggregate stats to a JSON file
    # output_stats_filename = f"activation_stats_{timestamp}.json"
    # with open(output_stats_filename, 'w') as f:
    #     json.dump(results, f, indent=4)
        
    # print(f"\nSaved aggregate activation statistics to '{output_stats_filename}'")
    
    # ####################################################################################
    # --- 7. Record Per-Image Activations ---
    # ####################################################################################
    print("\nRecording per-image activation values...")
    
    # NOTE: This re-runs the evaluation loop to keep concerns separate.
    # The model is already in model.eval() mode.
    image_activation_records = record_image_activations(
        model,
        test_loader,
        config.DEVICE
    )
    
    # Ensure the activation output directory exists
    os.makedirs(config.ACTIVATION_OUTPUT_DIR, exist_ok=True)
    
    # Save the per-image records to the config-specified directory
    # We add the timestamp to this file to make it unique
    output_records_filename = os.path.join(
        config.ACTIVATION_OUTPUT_DIR, 
        f"image_activation_records_{config.TASK_TYPE}_{config.CLASSIFICATION_TYPE}_{timestamp}.json"
    )
    
    with open(output_records_filename, 'w') as f:
        json.dump(image_activation_records, f) 
        
    print(f"\nSaved per-image activation records to '{output_records_filename}'")


    # ####################################################################################
    # --- 8. Generate Plots ---
    # ####################################################################################
    print("\nGenerating plots...")
    # NOTE: plot_curves might need updating if y_scores becomes multi-dimensional
    # (e.g., for multi-class AUC). 
    plot_curves(y_true, y_scores)

if __name__ == '__main__':
    main()

