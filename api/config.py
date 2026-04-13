from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # GolfSwingAnalysis/
SCRIPTS_DIR = BASE_DIR / "scripts"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

MODEL_PATH = MODELS_DIR / "swing_classifier.pkl"
FEATURES_CSV = DATA_DIR / "processed" / "swing_features.csv"
DB_PATH = BASE_DIR / "api" / "swing_history.db"
