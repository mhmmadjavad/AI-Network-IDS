import joblib
import pandas as pd
import numpy as np
import warnings
import os

# Suppress serialization warnings for cleaner terminal output
warnings.filterwarnings("ignore", category=UserWarning)

class IDSPredictor:
    def __init__(self, model_path='models/'):
        print("🔍 Initializing Inference Engine...")
        try:
            self.xgb_model = joblib.load(os.path.join(model_path, 'xgboost_model.pkl'))
            self.lr_model = joblib.load(os.path.join(model_path, 'logistic_model.pkl'))
            self.scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
            print("✅ Models and Scaler loaded successfully.")
        except Exception as e:
            print(f"❌ Initialization failed: {e}")

    def run_inference(self, data_dict):
        # NSL-KDD Numeric Feature Mapping (38 features expected by scaler)
        # Mapping key features to their respective indices in the training set
        input_vector = np.zeros((1, 38))
        
        feature_map = {
            'duration': 0,
            'src_bytes': 4,
            'count': 22,
            'serror_rate': 24,
            'dst_host_count': 31,
            'dst_host_srv_count': 32,
            'logged_in': 11
        }

        for key, value in data_dict.items():
            if key in feature_map:
                input_vector[0, feature_map[key]] = value

        # Standardizing input
        scaled_vec = self.scaler.transform(input_vector)
        
        # Logistic Regression Prediction
        lr_prob = self.lr_model.predict_proba(scaled_vec)[0][1]
        
        # XGBoost Prediction with dynamic feature alignment (for 122-dim models)
        try:
            xgb_prob = self.xgb_model.predict_proba(scaled_vec)[0][1]
        except ValueError:
            # Pad the vector if XGBoost was trained on one-hot encoded dataset (122 features)
            aligned_vec = np.zeros((1, 122))
            aligned_vec[0, :38] = scaled_vec
            xgb_prob = self.xgb_model.predict_proba(aligned_vec)[0][1]
        
        return {
            "Logistic": {"label": "ATTACK" if lr_prob > 0.5 else "NORMAL", "score": lr_prob},
            "XGBoost": {"label": "ATTACK" if xgb_prob > 0.5 else "NORMAL", "score": xgb_prob}
        }

def display_results(test_name, results):
    header = "{:<20} | {:<10} | {:<12}"
    print(f"\n--- {test_name} ---")
    print(header.format("ALGORITHM", "DETECTION", "PROBABILITY"))
    print("-" * 50)
    for model, res in results.items():
        print(header.format(model, res['label'], f"{res['score']*100:.2f}%"))

if __name__ == "__main__":
    predictor = IDSPredictor()

    # Case 1: High-intensity DoS Attack Simulation
    dos_attack = {
        'duration': 0, 
        'src_bytes': 0, 
        'count': 511, 
        'serror_rate': 1.0, 
        'dst_host_count': 255
    }
    
    # Case 2: Sophisticated Probe/Scan Simulation
    probe_attack = {
        'duration': 0, 
        'src_bytes': 0, 
        'count': 511, 
        'serror_rate': 1.0, 
        'dst_host_count': 255, 
        'dst_host_srv_count': 1, 
        'logged_in': 0
    }

    display_results("SCENARIO: DOS ATTACK", predictor.run_inference(dos_attack))
    display_results("SCENARIO: PROBE/SCAN ATTACK", predictor.run_inference(probe_attack))