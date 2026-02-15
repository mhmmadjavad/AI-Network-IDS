import joblib
import os

def save_artifact(obj, filename):
    """
    Utility function to persist models and scalers to the models/ directory.
    """
    directory = 'models'
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    target_path = os.path.join(directory, filename)
    joblib.dump(obj, target_path)
    print(f"✅ Artifact saved: {target_path}")