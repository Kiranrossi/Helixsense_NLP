from pathlib import Path
from setfit import SetFitModel
import os

# Base directories and paths
# This points to: /Users/kiranguruv/Helixsense_NLP/app
BASE_DIR = Path("/Users/kiranguruv/Helixsense_NLP/app")
MODELS_DIR = BASE_DIR / "models"

def load_setfit_model():
    """Load saved SetFit model from the setfit_model directory."""
    path = MODELS_DIR / "setfit_model"
    print(f"Loading from: {path}")
    if not path.exists():
        print(f"SetFit model directory not found at {path}")
        return
    try:
        model = SetFitModel.from_pretrained(str(path))
        print("Successfully loaded model")
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    load_setfit_model()
