import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(_BASE_DIR, "data/raw/PlantVillage")
BATCH_SIZE = 32
NUM_EPOCHS = 3  # The dataset has a lot of images, and since I dont have a dedicated GPU, 3 epochs is fine.
LEARNING_RATE = 1e-4



# Switch between the heavy model and the lightweight model.
ARCHITECTURE = "mobilenet_v3_small"  # Currently set to lightweight
MODEL_SAVE_PATH = f"models/{ARCHITECTURE}_best_model.pth"