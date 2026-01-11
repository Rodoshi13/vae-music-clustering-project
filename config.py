import os
import torch

# PROJECT PATHS
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
FORMLA_DIR = os.path.join(DATA_DIR, 'formla')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# FEATURE DIMENSIONS
AUDIO_FEATURE_DIM = 128
LYRICS_EMBEDDING_DIM = 300
COMBINED_FEATURE_DIM = AUDIO_FEATURE_DIM + LYRICS_EMBEDDING_DIM

# VAE CONFIGURATION - Change for each task
LATENT_DIM_EASY = 20
LATENT_DIM_MEDIUM = 30
LATENT_DIM_HARD = 40
HIDDEN_DIM = 256

# TRAINING
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# CLUSTERING
N_CLUSTERS = 10
RANDOM_STATE = 42
TEST_SPLIT = 0.2

# LANGUAGES
LANGUAGES = ['en', 'pt-br', 'es']

# CREATE DIRECTORIES IF THEY DON'T EXIST
os.makedirs(FORMLA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("✓ Config loaded")
print(f"✓ Using device: {DEVICE}")
