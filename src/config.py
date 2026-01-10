from pathlib import Path


# Common Paths
DATA_DIR = Path("data")
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = Path("models")
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# IDs
START_ID = 8000
END_ID = 9000

# File Paths
RAW_FILE = PROCESSED_DATA_DIR / f"cdep_{START_ID}_{END_ID}.json"
CLEAN_FILE = PROCESSED_DATA_DIR / f"clean_dataset_final_{START_ID}_{END_ID}.json"

# Models
EXTRACTIVE_MODEL_PATH = CHECKPOINTS_DIR / "matchsum_ro_epoch_5.pt"
ABSTRACTIVE_MODEL_PATH = MODELS_DIR / "mt5_finetuned"
BASE_BERT_MODEL = "dumitrescustefan/bert-base-romanian-cased-v1"

# Evaluation Defaults
TEST_SIZE_DEFAULT = 64
