from pathlib import Path

import pandas as pd
import joblib
from setfit import SetFitModel

# Base directories and paths
# This points to: /Users/kiranguruv/Helixsense_NLP/app
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data.xlsx"
MODELS_DIR = BASE_DIR / "models"


def load_dataset() -> pd.DataFrame:
    """Load the main dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    return pd.read_excel(DATA_PATH)


def load_tfidf_vectorizer():
    """Load saved TF-IDF vectorizer."""
    path = MODELS_DIR / "vectorizer.pkl"
    if not path.exists():
        raise FileNotFoundError(f"TF-IDF vectorizer not found at {path}")
    return joblib.load(path)


def load_tfidf_model():
    """Load saved Logistic Regression baseline model."""
    path = MODELS_DIR / "tfidf_logreg.pkl"
    if not path.exists():
        raise FileNotFoundError(f"TF-IDF Logistic Regression model not found at {path}")
    return joblib.load(path)


def load_setfit_model():
    """Load saved SetFit model from the setfit_model directory."""
    path = MODELS_DIR / "setfit_model"
    if not path.exists():
        raise FileNotFoundError(f"SetFit model directory not found at {path}")
    return SetFitModel.from_pretrained(str(path), local_files_only=True)


def _ensure_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure there is at least one recognised label column.
    If none of the expected label names are present, create PredictedCategory.
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
