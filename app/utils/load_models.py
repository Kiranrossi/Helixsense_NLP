from pathlib import Path

import pandas as pd
import joblib
from setfit import SetFitModel

# Base directory = app/ (where app.py and data.xlsx live)
BASE_DIR = Path(__file__).resolve().parent.parent  # utils -> app
DATA_PATH = BASE_DIR / "data.xlsx"
MODELS_DIR = BASE_DIR / "models"



def load_dataset() -> pd.DataFrame:
    """Load the main dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"data.xlsx not found at {DATA_PATH}")
    df = pd.read_excel(DATA_PATH)
    return df


def load_tfidf_vectorizer():
    path = MODELS_DIR / "vectorizer.pkl"   # your file
    if not path.exists():
        raise FileNotFoundError(f"TF-IDF vectorizer not found at {path}")
    return joblib.load(path)

def load_tfidf_model():
    path = MODELS_DIR / "tfidf_logreg.pkl"  # your file
    if not path.exists():
        raise FileNotFoundError(f"TF-IDF model not found at {path}")
    return joblib.load(path)


def load_setfit_model():
    """Load saved SetFit model from the setfit_model directory."""
    path = MODELS_DIR / "setfit_model"
    if not path.exists():
        raise FileNotFoundError(f"SetFit model directory not found at {path}")
    model = SetFitModel.from_pretrained(str(path))
    return model


def _ensure_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure there is at least one recognised label column.

    If none of the expected label names are present, create a temporary
    PredictedCategory = "Unknown" column so that baseline / SetFit /
    comparison / report tabs can all run end-to-end. [web:28][web:35]
    """
    label_candidates = [
        "label",
        "Label",
        "class",
        "Class",
        "category",
        "Category",
        "target",
        "PredictedCategory",
    ]
    existing = next((c for c in label_candidates if c in df.columns), None)
    if existing is None:
        df = df.copy()
        df["PredictedCategory"] = "Unknown"
    else:
        df[existing] = df[existing].astype(str)
    return df


def load_all_models_and_data():
    """
    Convenience function used by Streamlit.

    Returns: (df, tfidf_vectorizer, tfidf_model, setfit_model)
    """
    df = load_dataset()
    df = _ensure_label_column(df)

    tfidf_vectorizer = load_tfidf_vectorizer()
    tfidf_model = load_tfidf_model()
    setfit_model = load_setfit_model()
    return df, tfidf_vectorizer, tfidf_model, setfit_model
