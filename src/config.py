import torch

# --- Data Paths ---
BASE_DATA_PATH = "/home/abamini/face2pheno/data/"
TRIANGLE_LIST_PATH = f"{BASE_DATA_PATH}processed/triangles_list.csv"
ADJACENT_TRIANGLES_PATH = f"{BASE_DATA_PATH}processed/adjacent_triangles.csv"
LANDMARK_PIXELS_DIR = f"{BASE_DATA_PATH}processed/landmark_pixel_coordinates_test/"
IMAGE_DIR = '/data2/masino_lab/abamini/Face2pheno/utkface'
ACTIVATION_OUTPUT_DIR = '/data2/masino_lab/abamini/Face2pheno/image_activations/'

# --- Model & Training Parameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
EPOCHS = 25
NUM_KERNELS = 16
NUM_NODES = 854
TEST_SPLIT_RATIO = 0.3 # Use 30% of the data for testing
RANDOM_SEED = 42 # For reproducible train/test splits

# --- Feature Extraction Parameters ---
PATCH_SIZE = 32
IMAGE_CHANNELS = 3 # 3 for RGB

# Define the overall task
# Options: 'age_classification', 'emotion_detection', etc.
TASK_TYPE = 'age_classification'

# Define the classification type
# Options: 'binary', 'multi-class'
CLASSIFICATION_TYPE = 'multi-class'

# Define the age thresholds for classification.
# This will be used by your dataset loader to create labels.
# - For 'binary': A single value [T] creates classes 0 (< T) and 1 (>= T).
# - For 'multi-class': [T1, T2, T3] creates 4 classes:
#   Class 0: < T1
#   Class 1: T1 to < T2
#   Class 2: T2 to < T3
#   Class 3: >= T3
# AGE_THRESHOLDS = [12] # Example for binary
AGE_THRESHOLDS = [12, 25, 40, 60] # Example for multi-class

# --- Auto-calculated Number of Classes ---
if CLASSIFICATION_TYPE == 'binary':
    NUM_CLASSES = 2
elif CLASSIFICATION_TYPE == 'multi-class':
    NUM_CLASSES = len(AGE_THRESHOLDS) + 1
else:
    # Default or error case
    print(f"Warning: CLASSIFICATION_TYPE '{CLASSIFICATION_TYPE}' not recognized. Defaulting to 2 classes.")
    NUM_CLASSES = 2


