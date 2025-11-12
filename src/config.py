import torch

# --- Data Paths ---
BASE_DATA_PATH = "/home/abamini/face2pheno/data/"
# This CSV MUST have the 'supernode_group' column
TRIANGLE_LIST_PATH = f"{BASE_DATA_PATH}processed/triangles_list_super_node.csv" 
ADJACENT_TRIANGLES_PATH = f"{BASE_DATA_PATH}processed/adjacent_triangles.csv"
LANDMARK_PIXELS_DIR = f"{BASE_DATA_PATH}processed/landmark_pixel_coordinates/"
IMAGE_DIR = '/data2/masino_lab/abamini/Face2pheno/utkface'
ACTIVATION_OUTPUT_DIR = '/data2/masino_lab/abamini/Face2pheno/image_activations/'

# --- Model & Training Parameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
EPOCHS = 20
NUM_KERNELS = 8
TEST_SPLIT_RATIO = 0.3 # Use 30% of the data for testing
RANDOM_SEED = 42 # For reproducible train/test splits

# --- Feature Extraction Parameters ---
PATCH_SIZE = 32
IMAGE_CHANNELS = 3 # 3 for RGB

# --- Task Definition ---
TASK_TYPE = 'age_classification'
CLASSIFICATION_TYPE = 'multi-class'
AGE_THRESHOLDS = [12, 25, 40, 60] # Example for multi-class

# --- Graph Structure Parameters ---
NUM_TRIANGLES = 854 # The total number of triangles in the original mesh
NUM_SUPER_NODES = 10 # The number of groups 
USE_SUPER_NODES = 1 # 1 to use super-nodes, 0 otherwise

# --- Auto-calculated Parameters ---
if CLASSIFICATION_TYPE == 'binary':
    NUM_CLASSES = 2
elif CLASSIFICATION_TYPE == 'multi-class':
    NUM_CLASSES = len(AGE_THRESHOLDS) + 1
else:
    print(f"Warning: CLASSIFICATION_TYPE '{CLASSIFICATION_TYPE}' not recognized. Defaulting to 2 classes.")
    NUM_CLASSES = 2

# This tells the model how many nodes to expect
if USE_SUPER_NODES:
    MODEL_NUM_NODES = NUM_SUPER_NODES
else:
    MODEL_NUM_NODES = NUM_TRIANGLES