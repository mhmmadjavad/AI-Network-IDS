import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Custom modules for data ingestion
from load_data import load_raw_train
from encode_data import preprocess_and_encode

def execute_training_pipeline():
    """
    Standard training pipeline for the baseline Logistic Regression model.
    Includes scaling, stratified splitting, and performance evaluation.
    """
    print("🚀 Initializing Baseline Training Pipeline...")
    
    # Data Acquisition
    raw_df = load_raw_train()
    X, y = preprocess_and_encode(raw_df)
    
    # Stratified split to maintain class balance in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature Scaling (Crucial for convergence in Logistic Regression)
    print("⚖️ Normalizing features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model Initialization with increased iterations for high-dimensional data
    model = LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced')
    
    print("🧠 Training Logistic Regression Model...")
    model.fit(X_train_scaled, y_train)
    
    # Model Evaluation
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*30)
    print(f"📊 Performance Metrics:")
    print(f"Overall Accuracy: {acc:.4f}")
    print("="*30)
    
    print("\n[Confusion Matrix]")
    print(confusion_matrix(y_test, y_pred))
    
    print("\n[Detailed Classification Report]")
    print(classification_report(y_test, y_pred))
    
    # Artifact Persistence
    if not os.path.exists('models'):
        os.makedirs('models')
        
    joblib.dump(model, 'models/logistic_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("\n✅ Model and Scaler successfully exported to models/ directory.")
    return model, scaler

if __name__ == "__main__":
    execute_training_pipeline()