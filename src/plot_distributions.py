import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import config # <-- ADDED

def plot_patch_distribution(stats_data, patch_idx, output_dir):
    """
    Generates and saves a bar chart for a single patch, comparing mean
    kernel activations between ALL classes defined in config.

    Args:
        stats_data (dict): The loaded activation statistics from the JSON file.
        patch_idx (int): The index of the patch to visualize.
        output_dir (str): The directory where the plot image will be saved.
    """
    patch_key = f'patch_{patch_idx}'
    num_classes = config.NUM_CLASSES
    num_kernels = config.NUM_KERNELS

    class_means_data = []
    
    # --- Extract Mean Values for all classes ---
    for i in range(num_classes):
        class_key = f'class_{i}'
        try:
            class_data = stats_data[class_key][patch_key]
            # Get mean values, replacing None with 0
            means = [class_data[f'kernel_{k}']['mean'] or 0 for k in range(num_kernels)]
            class_means_data.append(means)
        except KeyError:
            print(f"Warning: Data for Patch {patch_idx}, Class {i} not found. Plotting as zeros.")
            class_means_data.append([0] * num_kernels)

    # --- Plotting ---
    x = np.arange(num_kernels)  # the label locations (0, 1, ... 7)
    
    # Calculate total width for all bars and individual bar width
    total_bar_width = 0.8
    width = total_bar_width / num_classes
    
    fig, ax = plt.subplots(figsize=(16, 7))

    # Get a color map
    colors = plt.cm.get_cmap('tab10', num_classes)

    for i in range(num_classes):
        # Calculate the offset for this class's bar
        offset = (i - (num_classes - 1) / 2) * width
        
        means = class_means_data[i]
        label = f'Class {i}'
        
        rects = ax.bar(x + offset, means, width, label=label, color=colors(i))
        ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=8)

    # --- Add labels, title, and custom x-axis tick labels, etc. ---
    ax.set_ylabel('Mean Activation Value')
    ax.set_xlabel('Kernel Index')
    ax.set_title(f'Mean Kernel Activation Distribution for Patch {patch_idx}')
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()

    # --- Save the plot ---
    output_filename = os.path.join(output_dir, f'patch_{patch_idx}_activation_distribution.png')
    plt.savefig(output_filename)
    print(f"Saved plot to '{output_filename}'")
    plt.close(fig) # Close the figure to free up memory

def main():
    """
    Main function to parse arguments and orchestrate the plotting.
    """
    parser = argparse.ArgumentParser(
        description="Visualize mean kernel activations from a statistics file."
    )
    parser.add_argument(
        "stats_file",
        type=str,
        help="Path to the activation_stats.json file."
    )
    parser.add_argument(
        "--patches",
        nargs='+',
        type=int,
        required=True,
        help="A list of patch indices to generate plots for (e.g., --patches 10 50 120)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Directory to save the output plots."
    )

    args = parser.parse_args()

    # --- Load Data ---
    try:
        with open(args.stats_file, 'r') as f:
            stats_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{args.stats_file}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{args.stats_file}'.")
        return

    # --- Create Output Directory ---
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Ensured output directory exists: '{args.output_dir}'")

    # --- Generate Plots ---
    for patch_idx in args.patches:
        plot_patch_distribution(stats_data, patch_idx, args.output_dir)

if __name__ == '__main__':
    main()
