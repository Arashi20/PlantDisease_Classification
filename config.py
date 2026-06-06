import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(_BASE_DIR, "data/raw/PlantVillage")
BATCH_SIZE = 32
NUM_EPOCHS = 3  # The dataset has a lot of images, and since I dont have a dedicated GPU, 3 epochs is fine.
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = os.path.join(_BASE_DIR, "models/best_model.pth")