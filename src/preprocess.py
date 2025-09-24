import pandas as pd
from sklearn.model_selection import train_test_split
import os

def preprocess_data():
    """Loads raw data, preprocesses it, and saves train/test splits."""
    RAW_DATA_PATH = "data/raw/hour.csv"
    PROCESSED_DATA_PATH = "data/processed"

    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)

    # Load data
    df = pd.read_csv(RAW_DATA_PATH)

    # Drop unnecessary columns
    df = df.drop(columns=['instant', 'dteday', 'casual', 'registered'])

    # Define features and target
    X = df.drop('cnt', axis=1)
    y = df['cnt']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save processed data
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(os.path.join(PROCESSED_DATA_PATH, "train.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DATA_PATH, "test.csv"), index=False)

    print("Data preprocessing complete. Train and test sets saved.")

if __name__ == "__main__":
    preprocess_data()