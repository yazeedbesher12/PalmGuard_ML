from dataclasses import dataclass

@dataclass(frozen=True)
class AudioConfig:
    sr: int = 16000
    seg_s: float = 2.0
    hop_s: float = 1.0
    n_mels: int = 64
    fmin: float = 50.0
    fmax: float = 3500.0

DATASET_DIR = "data/dataset"
LABELS_CSV = f"{DATASET_DIR}/labels.csv"
TREE_LOCATIONS_CSV = f"{DATASET_DIR}/tree_locations.csv"

MODEL_DIR = "models"
BEST_MODEL_PATH = f"{MODEL_DIR}/best_cnn.pt"

OUTPUTS_DIR = "outputs"
