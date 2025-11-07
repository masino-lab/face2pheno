import matplotlib.pyplot as plt
import numpy as np
import config
from sklearn.metrics import (
    roc_curve, auc, 
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, accuracy_score
)
import os # <-- ADDED to handle file paths

def plot_curves(y_true, y_scores, save_path="."):
    """
    Calculates, plots, and saves evaluation curves and reports.
    - For 'binary': Plots ROC and Precision-Recall curves.
    - For 'multi-class': Plots a Confusion Matrix and prints a classification report.

    Args:
        y_true (list or np.array): True class labels.
        y_scores (list or np.array): 
            - For binary: Probability estimates of the positive class.
            - For multi-class: Raw probability arrays (shape [n_samples, n_classes]).
        save_path (str): The directory where the plot images will be saved.
    """
    
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # --- Create save_path directory if it doesn't exist ---
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created directory: {save_path}")
    
    if config.CLASSIFICATION_TYPE == 'binary':
        print("Generating Binary Classification Plots (ROC, PR)...")
        # --- ROC Curve ---
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        roc_filename = os.path.join(save_path, "roc_curve.png") # <-- UPDATED
        plt.savefig(roc_filename)
        print(f"ROC curve saved to {roc_filename} (AUC = {roc_auc:.2f})")
        plt.close()

        # --- Precision-Recall Curve ---
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True)
        
        pr_filename = os.path.join(save_path, "pr_curve.png") # <-- UPDATED
        plt.savefig(pr_filename)
        print(f"Precision-Recall curve saved to {pr_filename} (Average Precision = {pr_auc:.2f})")
        plt.close()

    elif config.CLASSIFICATION_TYPE == 'multi-class':
        print("Generating Multi-Class Classification Report and Plot...")
        
        # Get predicted labels from scores
        y_pred = np.argmax(y_scores, axis=1)
        
        # Generate class labels (e.g., ["0", "1", "2", "3"])
        class_labels = [str(i) for i in range(config.NUM_CLASSES)]
        class_indices = np.arange(config.NUM_CLASSES)

        # --- UPDATED: Calculate, print, AND SAVE classification report ---
        print("\n--- Classification Report ---")
        accuracy = accuracy_score(y_true, y_pred)
        
        # Start building the report string
        report_str = f"Overall Accuracy: {accuracy:.4f}\n\n"
        
        report_dict_str = classification_report(
            y_true, 
            y_pred, 
            labels=class_indices, 
            target_names=class_labels
        )
        
        report_str += report_dict_str # Add the full report
        
        print(report_str) # Print the full report to the console
        
        # Save the report string to a text file
        report_filename = os.path.join(save_path, "classification_report.txt")
        try:
            with open(report_filename, 'w') as f:
                f.write(report_str)
            print(f"Classification report saved to {report_filename}")
        except IOError as e:
            print(f"Error saving classification report: {e}")
        # --- End of updated section ---

        # Calculate the confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=class_indices)
        
        # Plot the confusion matrix
        print("Generating Confusion Matrix plot...")
        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(ax=ax, cmap=plt.cm.Blues)
        
        plt.title('Confusion Matrix')
        cm_filename = os.path.join(save_path, "confusion_matrix.png") # <-- UPDATED
        plt.savefig(cm_filename)
        print(f"Confusion Matrix saved to {cm_filename}")
        plt.close()


