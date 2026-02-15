# Network Intrusion Detection System (IDS) using AI 🛡️

A robust Machine Learning pipeline designed to detect and classify network intrusions using the NSL-KDD dataset. This project compares **Logistic Regression** (Baseline) with **XGBoost** (Advanced) to provide a high-accuracy security solution.

## 🚀 Overview
This system analyzes network traffic patterns to distinguish between 'Normal' behavior and 'Malicious' attacks (DoS, Probe, etc.). It features a complete pipeline from raw data ingestion to real-time inference.



## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **ML Frameworks:** Scikit-Learn, XGBoost
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Persistence:** Joblib

## 📁 Project Structure
- `models/`: Contains serialized `.pkl` files for the Scaler and ML models.
- `evaluation/`: Exported performance charts and feature importance plots.
- `DATA/raw/`: Storage for the NSL-KDD dataset files.
- `predict.py`: The main inference engine for testing scenarios.
- `train_xgboost.py`: Training script for the Gradient Boosting model.
- `train_test.py`: Training script for the Logistic Regression baseline.
- `load_data.py` & `encode_data.py`: Modular preprocessing scripts.



## 📊 Performance & Insights
The system achieves high-precision detection by focusing on key network features:
- **Accuracy:** ~97.2% 
- **Key Indicators:** `src_bytes`, `count`, `serror_rate`.

XGBoost provides superior stability in complex scenarios, while Logistic Regression offers rapid, linear decision-making for standard threats.



## ⚙️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/AI-IDS.git](https://github.com/YourUsername/AI-IDS.git)
   cd AI-IDS




##Install dependencies:
pip install -r requirements.txt

##Train the models:
python train_xgboost.py
python train_test.py


##Run Inference/Tests:
python predict.py

📝 License
This project is for educational purposes as part of an AI Network Security study.
