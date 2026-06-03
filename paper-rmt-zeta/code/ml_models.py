"""
ml_models.py
============
Implementations numpy-only de :
  - Ridge regression (closed-form)
  - Decision tree (CART, regression)
  - Random Forest (bagging of trees)
  - Gradient Boosting Regressor (XGBoost-light)
  - MLP avec optimiseur Adam
  - Baseline Monte Carlo (predicteur naif = vol historique recente)

Toutes les classes implementent fit(X, y) et predict(X).
"""
from __future__ import annotations
import numpy as np


# ============================================================
# 1.  RIDGE REGRESSION
# ============================================================
class Ridge:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.w_ = None
        self.b_ = None
        self.mu_x = self.sd_x = self.mu_y = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        self.mu_x = X.mean(0); self.sd_x = X.std(0) + 1e-12
        self.mu_y = y.mean()
        Xs = (X - self.mu_x) / self.sd_x
        ys = y - self.mu_y
        n, d = Xs.shape
        A = Xs.T @ Xs + self.alpha * np.eye(d)
        self.w_ = np.linalg.solve(A, Xs.T @ ys)
        self.b_ = self.mu_y - (self.mu_x / self.sd_x) @ self.w_  # not used directly
        return self

    def predict(self, X):
        Xs = (X - self.mu_x) / self.sd_x
        return Xs @ self.w_ + self.mu_y


# ============================================================
# 2.  DECISION TREE (CART regression, simplifie)
# ============================================================
class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")
    def __init__(self):
        self.feature = -1; self.threshold = 0.0
        self.left = None; self.right = None
        self.value = 0.0


class DecisionTreeRegressor:
    def __init__(self, max_depth: int = 6, min_samples_split: int = 10,
                  rng: np.random.Generator = None, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.rng = rng or np.random.default_rng()
        self.max_features = max_features
        self.root: _Node | None = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        self.n_features_ = X.shape[1]
        self.root = self._grow(X, y, depth=0)
        return self

    def _grow(self, X, y, depth):
        node = _Node()
        node.value = float(y.mean())
        n = len(y)
        if depth >= self.max_depth or n < self.min_samples_split or y.std() < 1e-12:
            return node
        # selection de features (pour RF)
        n_feat = self.n_features_
        if self.max_features is None:
            features = np.arange(n_feat)
        else:
            k = max(1, int(self.max_features))
            features = self.rng.choice(n_feat, size=min(k, n_feat), replace=False)
        best_gain = 0.0; best_feat = -1; best_thr = 0.0
        ssE_parent = ((y - y.mean()) ** 2).sum()
        for f in features:
            xs = X[:, f]
            # 10 quantiles candidats (au lieu de toutes les valeurs : speed)
            qs = np.quantile(xs, np.linspace(0.1, 0.9, 9))
            for thr in np.unique(qs):
                left_mask = xs <= thr
                if left_mask.sum() < 3 or (~left_mask).sum() < 3:
                    continue
                yl, yr = y[left_mask], y[~left_mask]
                ssE = ((yl - yl.mean()) ** 2).sum() + ((yr - yr.mean()) ** 2).sum()
                gain = ssE_parent - ssE
                if gain > best_gain:
                    best_gain = gain; best_feat = int(f); best_thr = float(thr)
        if best_feat < 0:
            return node
        node.feature = best_feat; node.threshold = best_thr
        left_mask = X[:, best_feat] <= best_thr
        node.left  = self._grow(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _predict_one(self, x, node):
        while node.feature >= 0:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x, self.root) for x in X])


# ============================================================
# 3.  RANDOM FOREST (bagging)
# ============================================================
class RandomForestRegressor:
    def __init__(self, n_estimators: int = 100, max_depth: int = 8,
                 max_features=None, seed: int = 0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.rng = np.random.default_rng(seed)
        self.trees: list[DecisionTreeRegressor] = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n = len(y)
        n_feat = X.shape[1]
        mf = self.max_features or max(1, int(np.sqrt(n_feat)))
        self.trees.clear()
        for k in range(self.n_estimators):
            idx = self.rng.integers(0, n, size=n)            # bootstrap
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                          rng=self.rng, max_features=mf)
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
        return self

    def predict(self, X):
        preds = np.mean([t.predict(X) for t in self.trees], axis=0)
        return preds


# ============================================================
# 4.  GRADIENT BOOSTING REGRESSOR (XGBoost-light)
# ============================================================
class GradientBoostingRegressor:
    """Boosting de stumps/trees sur residuals (Friedman 2001), perte L2."""
    def __init__(self, n_estimators: int = 200, learning_rate: float = 0.05,
                 max_depth: int = 3, subsample: float = 0.7, seed: int = 0):
        self.n_estimators = n_estimators
        self.eta = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.rng = np.random.default_rng(seed)
        self.f0 = 0.0
        self.trees: list[DecisionTreeRegressor] = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n = len(y)
        self.f0 = float(y.mean())
        F = np.full(n, self.f0)
        self.trees.clear()
        for k in range(self.n_estimators):
            r = y - F                                            # residual = pseudo-gradient
            if self.subsample < 1.0:
                idx = self.rng.choice(n, size=int(self.subsample * n), replace=False)
            else:
                idx = np.arange(n)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, rng=self.rng)
            tree.fit(X[idx], r[idx])
            self.trees.append(tree)
            F = F + self.eta * tree.predict(X)
        return self

    def predict(self, X):
        F = np.full(len(X), self.f0)
        for t in self.trees:
            F = F + self.eta * t.predict(X)
        return F


# ============================================================
# 5.  MLP avec optimiseur Adam
# ============================================================
class MLPRegressor:
    """MLP simple, ReLU, optimiseur Adam, perte MSE."""
    def __init__(self, hidden_sizes=(64, 32), lr=1e-3, n_iter=400,
                 batch_size=64, weight_decay=1e-4, seed=0, verbose=False):
        self.hidden_sizes = tuple(hidden_sizes)
        self.lr = lr
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.wd = weight_decay
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose

    def _init_params(self, d_in):
        sizes = (d_in,) + self.hidden_sizes + (1,)
        self.Ws = []
        self.bs = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            W = self.rng.standard_normal((a, b)) * np.sqrt(2.0 / a)
            self.Ws.append(W)
            self.bs.append(np.zeros(b))
        # Adam state
        self.mW = [np.zeros_like(W) for W in self.Ws]
        self.vW = [np.zeros_like(W) for W in self.Ws]
        self.mb = [np.zeros_like(b) for b in self.bs]
        self.vb = [np.zeros_like(b) for b in self.bs]
        self.t = 0

    @staticmethod
    def _relu(x): return np.maximum(0, x)

    def _forward(self, X):
        a = X
        acts = [a]
        zs = []
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            z = a @ W + b
            zs.append(z)
            a = self._relu(z) if i < len(self.Ws) - 1 else z
            acts.append(a)
        return acts, zs

    def _backward(self, X, y, acts, zs):
        n = len(X)
        # output layer derivative dL/dz_L
        delta = (acts[-1] - y.reshape(-1, 1)) * 2.0 / n
        gradsW = [None] * len(self.Ws); gradsb = [None] * len(self.bs)
        for i in range(len(self.Ws) - 1, -1, -1):
            gradsW[i] = acts[i].T @ delta + self.wd * self.Ws[i]
            gradsb[i] = delta.sum(0)
            if i > 0:
                delta = (delta @ self.Ws[i].T) * (zs[i-1] > 0)
        return gradsW, gradsb

    def _adam_step(self, gW, gb, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for i in range(len(self.Ws)):
            self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * gW[i]
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * gW[i] ** 2
            mhat = self.mW[i] / (1 - beta1 ** self.t)
            vhat = self.vW[i] / (1 - beta2 ** self.t)
            self.Ws[i] -= self.lr * mhat / (np.sqrt(vhat) + eps)
            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * gb[i]
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * gb[i] ** 2
            mhat = self.mb[i] / (1 - beta1 ** self.t)
            vhat = self.vb[i] / (1 - beta2 ** self.t)
            self.bs[i] -= self.lr * mhat / (np.sqrt(vhat) + eps)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        # Standardize
        self.mu_x = X.mean(0); self.sd_x = X.std(0) + 1e-12
        self.mu_y = y.mean(); self.sd_y = y.std() + 1e-12
        Xs = (X - self.mu_x) / self.sd_x
        ys = (y - self.mu_y) / self.sd_y
        self._init_params(Xs.shape[1])
        n = len(ys); bs = self.batch_size
        loss_hist = []
        for epoch in range(self.n_iter):
            perm = self.rng.permutation(n)
            losses = []
            for start in range(0, n, bs):
                idx = perm[start:start + bs]
                Xb, yb = Xs[idx], ys[idx]
                acts, zs = self._forward(Xb)
                pred = acts[-1].ravel()
                losses.append(np.mean((pred - yb) ** 2))
                gW, gb = self._backward(Xb, yb, acts, zs)
                self._adam_step(gW, gb)
            if self.verbose and epoch % 50 == 0:
                print(f"  epoch {epoch:4d}  loss = {np.mean(losses):.4f}")
            loss_hist.append(np.mean(losses))
        self.loss_history_ = np.array(loss_hist)
        return self

    def predict(self, X):
        Xs = (np.asarray(X, dtype=float) - self.mu_x) / self.sd_x
        acts, _ = self._forward(Xs)
        return acts[-1].ravel() * self.sd_y + self.mu_y


# ============================================================
# 6.  BASELINE : Monte Carlo / Naive HAR-like
# ============================================================
class MonteCarloVolPredictor:
    """
    Baseline simulationniste : pour chaque sample, on bootstrap les
    rendements de la fenetre d'entrainement les plus proches en terme
    de feature `lambda_max`, on simule M = 500 chemins de 21j et on en
    deduit la vol attendue. Tres simple, mais sert de benchmark.
    """
    def __init__(self, n_paths: int = 500, k_neighbors: int = 50, seed: int = 0):
        self.n_paths = n_paths
        self.k = k_neighbors
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y, daily_vol_train: np.ndarray | None = None):
        """daily_vol_train : vol journalière estimée pour chaque entrée du train.
           Si None on suppose y est annualise et on revient en daily."""
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        preds = np.zeros(len(X))
        for i, x in enumerate(X):
            # Distance euclidienne (sur features standardisees pour fairness)
            d = np.linalg.norm(self.X - x, axis=1)
            top = np.argsort(d)[:self.k]
            preds[i] = self.y[top].mean()
        return preds


# ============================================================
# METRIQUES
# ============================================================
def rmse(y_true, y_pred): return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
def mae(y_true, y_pred):  return float(np.mean(np.abs(y_true - y_pred)))
def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
def qlike(y_true, y_pred):
    """QLIKE = sigma^2_pred / sigma^2_real - log(sigma^2_pred/sigma^2_real) - 1.
       Robuste pour vol (Patton 2011)."""
    y_t = np.maximum(y_true, 1e-6) ** 2
    y_p = np.maximum(y_pred, 1e-6) ** 2
    r = y_p / y_t
    return float(np.mean(r - np.log(r) - 1))


if __name__ == "__main__":
    # Test : regression simple y = 2 x1 + 0.5 x2^2 + noise
    rng = np.random.default_rng(0)
    n, d = 600, 4
    X = rng.standard_normal((n, d))
    y = 2 * X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.1 * rng.standard_normal(n)
    split = 400
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    print("Test models on toy data :")
    for name, m in [
        ("Ridge",      Ridge(alpha=1.0)),
        ("Tree",       DecisionTreeRegressor(max_depth=5)),
        ("RandomForest", RandomForestRegressor(n_estimators=30, max_depth=5, seed=0)),
        ("GradBoost",  GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, seed=0)),
        ("MLP-Adam",   MLPRegressor(hidden_sizes=(32, 16), n_iter=200, seed=0)),
    ]:
        m.fit(Xtr, ytr)
        yp = m.predict(Xte)
        print(f"  {name:14s} R2={r2(yte, yp):+.3f}  RMSE={rmse(yte, yp):.4f}")
