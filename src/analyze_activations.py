import json
import argparse
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import config  

def load_and_process_data(filepath):
    """
    Loads the per-image activation records and processes them into a
    nested dictionary, aggregated by KERNEL (across all patches).
    
    Args:
        filepath (str): Path to the 'image_activation_records_...json' file.
        
    Returns:
        dict: A nested dict: data[class_label][kernel_idx] = [activations]
    """
    print(f"Loading data from {filepath}...")
    with open(filepath, 'r') as f:
        all_image_records = json.load(f)

    if not all_image_records:
        raise ValueError("Loaded data is empty. Cannot perform analysis.")

    # Infer num_kernels from the first record
    try:
        num_kernels = len(all_image_records[0]['activations'][0]['kernel_activations'])
        print(f"Detected {num_kernels} kernels.")
    except (IndexError, KeyError) as e:
        print(f"Error parsing first record: {e}")
        # Read from config as a fallback
        num_kernels = config.NUM_KERNELS
        print(f"Assuming {num_kernels} kernels based on config.")

    # --- UPDATED: Read NUM_CLASSES from config ---
    num_classes = config.NUM_CLASSES
    print(f"Processing for {num_classes} classes.")

    # Initialize the nested data structure
    # data = {class_label: {kernel_idx: [values]}}
    data = {
        c_idx: {k_idx: [] for k_idx in range(num_kernels)} for c_idx in range(num_classes)
    }

    # Populate the data structure
    for image_record in all_image_records:
        class_label = image_record.get('class_label')
        
        # Check if label is valid
        if class_label is None or not (0 <= class_label < num_classes):
            continue 
            
        for patch_record in image_record['activations']:
            for kernel_idx, activation_val in enumerate(patch_record['kernel_activations']):
                if kernel_idx >= num_kernels:
                    continue # Safety check
                
                # Append to the kernel's list for the correct class
                data[class_label][kernel_idx].append(activation_val)
    
    print("Data processed successfully.")
    return data, num_kernels, num_classes

def calculate_significance(processed_data, num_kernels, num_classes):
    """
    Performs a One-Way ANOVA for each KERNEL (aggregated across patches) to find
    significant differences between class means.
    
    Args:
        processed_data (dict): The nested dict from load_and_process_data.
        num_kernels (int): The number of kernels.
        num_classes (int): The number of classes.
        
    Returns:
        list: A sorted list of tuples, ranked by significance:
              [(kernel_idx, f_stat, p_value, mean_per_class), ...]
    """
    print("Calculating statistical significance for all kernels (aggregated across patches)...")
    results = []
    
    for kernel_idx in range(num_kernels):
        
        # Collect activation lists for this kernel from all classes
        class_activations = []
        mean_per_class = []
        
        for c_idx in range(num_classes):
            activations = processed_data[c_idx][kernel_idx]
            # Need at least 2 samples in each group for ANOVA
            if len(activations) < 2:
                class_activations = None # Flag as invalid
                break
            
            class_activations.append(activations)
            mean_per_class.append(np.mean(activations))

        if class_activations is None:
            continue # Skip this kernel, not enough data in one or more groups
            
        # --- UPDATED: Perform One-Way ANOVA ---
        # This tests if there is a significant difference between ANY of the class means
        f_stat, p_value = stats.f_oneway(*class_activations)
        
        if np.isnan(f_stat) or np.isnan(p_value):
            continue
            
        results.append((kernel_idx, f_stat, p_value, mean_per_class))

    # --- UPDATED: Sort results by the F-statistic, descending. ---
    # High F-stat = large variance between groups vs. within groups
    results.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"ANOVA tests completed. Found {len(results)} valid combinations.")
    return results

def plot_kernel_distributions(all_results, processed_data, num_kernels, num_classes):
    """
    Generates a grid of distribution plots for ALL kernels.
    
    Args:
        all_results (list): The list of tuples from calculate_significance.
        processed_data (dict): The nested data dict.
        num_kernels (int): The number of kernels (e.g., 8).
        num_classes (int): The number of classes.
    """
    if not all_results:
        print("No results to plot.")
        return

    print("Generating all kernel distribution plots...")
    
    rows = (num_kernels + 1) // 2
    cols = 2
    if num_kernels == 1:
        rows, cols = 1, 1
        
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 5), squeeze=False)
    axes_flat = axes.flatten()
    
    # Get a color map for plotting multiple classes
    colors = plt.cm.get_cmap('tab10', num_classes)
    
    stats_map = {res[0]: res[1:] for res in all_results} 

    for kernel_idx in range(num_kernels):
        ax = axes_flat[kernel_idx]
        
        # --- UPDATED: Plot KDE for all classes ---
        for c_idx in range(num_classes):
            activations = processed_data[c_idx][kernel_idx]
            if activations:
                sns.kdeplot(
                    activations, 
                    ax=ax, 
                    label=f'Class {c_idx}', 
                    fill=True, 
                    color=colors(c_idx), 
                    alpha=0.5
                )
        
        # Add titles and labels
        if kernel_idx in stats_map:
            f_stat, p_value, mean_per_class = stats_map[kernel_idx]
            # Format means for title
            means_str = ", ".join([f"C{i}: {m:.2f}" for i, m in enumerate(mean_per_class)])
            title = (
                f"Kernel {kernel_idx}\n"
                f"Means: [{means_str}]\n"
                f"F-statistic: {f_stat:.2f}, p-value: {p_value:.1e}"
            )
        else:
            title = f"Kernel {kernel_idx}\n(Not enough data for ANOVA)"
        
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Max Activation Value")
        ax.set_ylabel("Density")
        ax.legend()
        
    # Hide any unused subplots
    for i in range(num_kernels, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout(pad=2.0)
    
    os.makedirs('plots', exist_ok=True)
    output_filename = "plots/all_kernel_distributions.png"
    plt.savefig(output_filename)
    plt.close(fig)
    
    print(f"\nSuccessfully saved combined plot to '{output_filename}'")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze activation records to find kernel-level differences."
    )
    # --- THIS LINE IS NOW FIXED ---
    parser.add_argument(
        "json_file", 
        type=str, 
        help="Path to the 'image_activation_records_...json' file."
    )
    args = parser.parse_args()
    
    try:
        # 1. Load and process data
        data, num_kernels, num_classes = load_and_process_data(config.ACTIVATION_OUTPUT_DIR + args.json_file)
        
        # 2. Calculate significance
        all_results = calculate_significance(data, num_kernels, num_classes)
        
        if not all_results:
            print("Could not find any significant results.")
            return

        # 3. Plot the distributions for all kernels
        plot_kernel_distributions(all_results, data, num_kernels, num_classes)
        
        # --- UPDATED: Print ANOVA results ---
        print("\n--- Significance Stats per Kernel (Aggregated) ---")
        print(f"{'Kernel':<6} | {'F-stat':<10} | {'p-value':<12} | {'Class Means':<30}")
        print("-" * 65)
        # Sort by kernel_idx for a clean print
        all_results.sort(key=lambda x: x[0])
        for (k, f, pval, means) in all_results:
            means_str = ", ".join([f"{m:.2f}" for m in means])
            print(f"{k:<6} | {f:<10.3f} | {pval:<12.3e} | {means_str:<30}")

    except FileNotFoundError:
        print(f"Error: File not found at {args.json_file}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.json_file}. File might be corrupt or empty.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

#/data2/masino_lab/abamini/Face2pheno/image_activations/image_activation_records_age_classification_multi-class_20251110_123708.json