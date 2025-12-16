from pathlib import Path
# Mocking SetFitModel to verify the fix logic without installing the heavy library
# We want to ensure that we are passing local_files_only=True
class SetFitModel:
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        print(f"Loading from: {path}")
        print(f"kwargs: {kwargs}")
        if not kwargs.get("local_files_only"):
            raise ValueError("Expected local_files_only=True")
        return "Model Loaded"

import sys
# Mock the module for the test
sys.modules["setfit"] = type(sys)("setfit")
sys.modules["setfit"].SetFitModel = SetFitModel

# Now import the util
sys.path.append("/Users/kiranguruv/Helixsense_NLP/app")
from utils.load_models import load_setfit_model

if __name__ == "__main__":
    try:
        model = load_setfit_model()
        print("Success!")
    except Exception as e:
        print(f"Failed: {e}")
