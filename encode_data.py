import pandas as pd

def preprocess_and_encode(df):
    """
    Performs feature selection and label encoding.
    Converts categorical network traffic features into numerical format.
    """
    # Dropping the difficulty column as it's not a network feature
    if 'difficulty' in df.columns:
        df = df.drop('difficulty', axis=1)

    # Binary classification: 'normal' vs everything else (attack)
    y = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    
    # Feature selection: Extracting only numeric columns for the baseline/scaler
    # This aligns with the 38-feature vector used in our inference engine
    X = df.select_dtypes(include=['number'])
    
    return X, y