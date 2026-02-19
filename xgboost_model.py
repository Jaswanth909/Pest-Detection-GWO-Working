from xgboost import XGBClassifier
from utils import load_config

def build_xgb_model(params=None):
    cfg = load_config()
    model_cfg = cfg["model"]

    base_params = {
        "objective": model_cfg.get("objective", "binary:logistic"),
        "eval_metric": model_cfg.get("eval_metric", "auc"),
        "n_estimators": model_cfg.get("n_estimators", 200),
        "use_label_encoder": False
    }

    if params is not None:
        base_params.update(params)

    model = XGBClassifier(**base_params)
    return model
