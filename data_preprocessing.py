import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import load_config

def load_data():
    cfg = load_config()
    data_cfg = cfg["data"]

    df = pd.read_csv(data_cfg["features_path"])
    target_col = data_cfg["target_column"]

    y = df[target_col]
    X = df.drop(columns=[target_col])

    return X, y

def train_val_split(X, y):
    cfg = load_config()
    split_cfg = cfg["split"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
        stratify=y
    )
    return X_train, X_val, y_train, y_val

def scale_features(X_train, X_val):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled, scaler
