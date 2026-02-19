import numpy as np

from utils import load_config, ensure_int, clip
from xgboost_model import build_xgb_model
from sklearn.metrics import roc_auc_score

class GWOXGBoost:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        cfg = load_config()
        gwo_cfg = cfg["gwo"]
        self.n_wolves = gwo_cfg["n_wolves"]
        self.max_iterations = gwo_cfg["max_iterations"]

        bounds_cfg = gwo_cfg["param_bounds"]
        self.param_names = list(bounds_cfg.keys())
        self.bounds = np.array([bounds_cfg[name] for name in self.param_names], dtype=float)
        self.dim = len(self.param_names)

    def _decode(self, position):
        params = {}
        for i, name in enumerate(self.param_names):
            low, high = self.bounds[i]
            val = clip(position[i], low, high)
            if name == "max_depth":
                val = ensure_int(val)
            params[name] = val
        return params

    def _fitness(self, position):
        params = self._decode(position)
        model = build_xgb_model(params)
        model.fit(self.X_train, self.y_train)
        y_proba = model.predict_proba(self.X_val)[:, 1]
        auc = roc_auc_score(self.y_val, y_proba)
        return auc

    def optimize(self):
        lb = self.bounds[:, 0]
        ub = self.bounds[:, 1]

        wolves = np.random.uniform(lb, ub, size=(self.n_wolves, self.dim))

        alpha_pos = np.zeros(self.dim)
        alpha_score = -np.inf
        beta_pos = np.zeros(self.dim)
        beta_score = -np.inf
        delta_pos = np.zeros(self.dim)
        delta_score = -np.inf

        for t in range(self.max_iterations):
            a = 2 - t * (2 / self.max_iterations)

            for i in range(self.n_wolves):
                score = self._fitness(wolves[i])

                if score > alpha_score:
                    delta_score, delta_pos = beta_score, beta_pos.copy()
                    beta_score, beta_pos = alpha_score, alpha_pos.copy()
                    alpha_score, alpha_pos = score, wolves[i].copy()
                elif score > beta_score:
                    delta_score, delta_pos = beta_score, beta_pos.copy()
                    beta_score, beta_pos = score, wolves[i].copy()
                elif score > delta_score:
                    delta_score, delta_pos = score, wolves[i].copy()

            for i in range(self.n_wolves):
                for d in range(self.dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a * r1 - 1
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha_pos[d] - wolves[i, d])
                    X1 = alpha_pos[d] - A1 * D_alpha

                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - 1
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta_pos[d] - wolves[i, d])
                    X2 = beta_pos[d] - A2 * D_beta

                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a * r1 - 1
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta_pos[d] - wolves[i, d])
                    X3 = delta_pos[d] - A3 * D_delta

                    wolves[i, d] = (X1 + X2 + X3) / 3.0
                    wolves[i, d] = clip(wolves[i, d], lb[d], ub[d])

            print(f"Iteration {t+1}/{self.max_iterations}, best AUC: {alpha_score:.4f}")

        best_params = self._decode(alpha_pos)
        return best_params, alpha_score
