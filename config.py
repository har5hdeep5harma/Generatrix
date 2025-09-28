import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 25
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
LATENT_DIM = 64 
HYPERNET_HIDDEN_DIM = 128
IMG_CHANNELS = 3
IMG_SIZE = 32
NUM_CLASSES = 10
