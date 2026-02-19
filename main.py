import json
from pathlib import Path

from utils import load_config, make_dirs
from data_preprocessing import load_data, train_val_split, scale_features
from gwo_optimizer import GWOXGBoost
from xgboost_model import build_xgb_model
from model_evaluation import evaluate_model

def main():
    make_dirs()
    cfg = load_config()

    print("Loading data...")
    X, y = load_data()

    print("Train/validation split...")
    X_train, X_val, y_train, y_val = train_val_split(X, y)

    print("Scaling features...")
    X_train_scaled, X_val_scaled, scaler = scale_features(X_train, X_val)

    print("Running Gray Wolf Optimization for XGBoost hyperparameters...")
    gwo = GWOXGBoost(X_train_scaled, y_train, X_val_scaled, y_val)
    best_params, best_auc = gwo.optimize()

    print("Best params:", best_params)
    print(f"Best validation AUC: {best_auc:.4f}")

    print("Training final model with best params...")
    model = build_xgb_model(best_params)
    model.fit(X_train_scaled, y_train)

    print("Evaluating final model...")
    metrics = evaluate_model(model, X_val_scaled, y_val)

    Path("results").mkdir(exist_ok=True, parents=True)
    model.save_model("results/best_model.json")
    with open("results/best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    print("Saved best model and params to results/")
    print("Evaluation metrics:", metrics)

if __name__ == "__main__":
    main()
