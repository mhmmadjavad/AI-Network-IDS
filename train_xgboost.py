import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import os
from load_data import load_raw_train 
from encode_data import preprocess_and_encode 

def train_network_security_model():
    """
    Main pipeline for training the XGBoost classifier for Network IDS.
    """
    print("Step 1: Data Acquisition and Preprocessing...")
    
    # Loading raw KDD dataset using custom loader
    raw_df = load_raw_train() 
    
    # Feature engineering and encoding (resulting in aligned features)
    X, y = preprocess_and_encode(raw_df)
    
    print(f"Input features shape: {X.shape}")
    
    # Initializing XGBoost Classifier with optimized hyperparameters
    print("Step 2: Training Gradient Boosted Trees...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        eval_metric='logloss',
        random_state=42,
        tree_method='hist' # Optimized for medium-to-large datasets
    )

    # Training on the processed dataset
    model.fit(X, y)

    # Persistence: Saving the trained model to the models directory
    if not os.path.exists('models'):
        os.makedirs('models')
        
    joblib.dump(model, 'models/xgboost_model.pkl')
    print("✅ Model serialized and saved to models/xgboost_model.pkl")

    # Feature Importance Visualization
    print("Step 3: Analyzing Feature Importance...")
    plt.figure(figsize=(12, 8))
    
    # Using XGBoost's built-in importance plotting
    ax = plt.gca()
    xgb.plot_importance(model, max_num_features=15, ax=ax, grid=False, importance_type='weight')
    
    plt.title("Key Network Features Driving IDS Decisions")
    
    # Ensure labels aren't cut off
    plt.tight_layout()
    
    if not os.path.exists('evaluation'):
        os.makedirs('evaluation')
        
    plt.savefig('evaluation/feature_importance.png', dpi=300)
    print("📊 Evaluation charts exported to evaluation/ folder.")

if __name__ == "__main__":
    train_network_security_model()