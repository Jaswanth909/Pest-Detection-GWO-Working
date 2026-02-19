from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import json

def evaluate_model(model, X_val, y_val, save_path: str = "results/evaluation_metrics.json"):
    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "auc_roc": float(roc_auc_score(y_val, y_proba)),
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred)),
        "recall": float(recall_score(y_val, y_pred)),
    }

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
