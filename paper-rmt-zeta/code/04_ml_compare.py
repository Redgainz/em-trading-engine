"""
04_ml_compare.py
================
Comparaison de 6 modeles pour predire vol_future_21 :
  - Constant-mean       (baseline trivial)
  - HAR-like (Ridge sur vol historique seule)
  - Ridge sur features RMT+Zeta
  - RandomForest sur features RMT+Zeta
  - GradientBoosting sur features RMT+Zeta  (analog XGBoost)
  - MLP-Adam sur features RMT+Zeta
  - Monte Carlo / kNN

Split temporel : 75% train (anciennes obs), 25% test (recentes).
Metriques : RMSE, MAE, R^2, QLIKE.
Sortie  : tableau + predictions CSV + figures.
"""
import os, sys, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
from ml_models import (Ridge, RandomForestRegressor, GradientBoostingRegressor,
                        MLPRegressor, MonteCarloVolPredictor,
                        rmse, mae, r2, qlike)

OUT_DIR = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(OUT_DIR, "data")
FIG_DIR  = os.path.join(OUT_DIR, "figures")

df = pd.read_csv(os.path.join(DATA_DIR, "features.csv"), index_col=0, parse_dates=True)
print(f"Loaded features : {df.shape}")

# Choix des features
# (on EXCLUT 'regime' qui est l'info ground-truth, et TW_p qui est NaN/constante)
FEATURES_RMT = [
    "lambda_max", "lambda_2", "lambda_3", "var_market", "sigma2_noise",
    "n_factors", "TW_z", "participation_ratio", "IPR_v1",
    "D_KS_GUE", "D_KS_Poisson", "zeta_score",
]
FEATURES_VOL = ["vol_past_120", "vol_ewma_60"]
FEATURES_ALL = FEATURES_RMT + FEATURES_VOL
TARGET = "vol_future_21"

# Transformation log pour stabilite (vol = sqrt(var) -> log-vol gaussien-like)
df["log_vol_future"] = np.log(df[TARGET].clip(lower=1e-3))
df["log_vol_past"]   = np.log(df["vol_past_120"].clip(lower=1e-3))
df["log_vol_ewma"]   = np.log(df["vol_ewma_60"].clip(lower=1e-3))
FEATURES_VOL_LOG = ["log_vol_past", "log_vol_ewma"]
TARGET_LOG = "log_vol_future"

# Split temporel 75/25
n = len(df)
split = int(0.75 * n)
df_tr = df.iloc[:split]
df_te = df.iloc[split:]
print(f"Train : {len(df_tr)} obs ({df_tr.index[0].date()} -> {df_tr.index[-1].date()})")
print(f"Test  : {len(df_te)} obs ({df_te.index[0].date()} -> {df_te.index[-1].date()})")

# Le tableau principal travaille sur log-vol (predictions back-transformees)
y_tr_log = df_tr[TARGET_LOG].values; y_te_log = df_te[TARGET_LOG].values
y_tr = df_tr[TARGET].values; y_te = df_te[TARGET].values

def expo(z): return np.exp(z)

# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
results = {}
preds = {}

def evaluate(name, y_true, y_pred):
    return dict(
        model=name,
        RMSE=rmse(y_true, y_pred),
        MAE=mae(y_true, y_pred),
        R2=r2(y_true, y_pred),
        QLIKE=qlike(y_true, y_pred),
    )

print("\n=== Entrainement et evaluation (target = log_vol_future) ===\n")

# 1. Constant baseline
y_pred = np.full_like(y_te, np.exp(y_tr_log.mean()))
results["Const-Mean"] = evaluate("Const-Mean", y_te, y_pred)
preds["Const-Mean"] = y_pred

# 2. HAR-like (Ridge avec uniquement vol historique en log)
X_tr_h = df_tr[FEATURES_VOL_LOG].values; X_te_h = df_te[FEATURES_VOL_LOG].values
m = Ridge(alpha=1.0).fit(X_tr_h, y_tr_log)
y_pred = expo(m.predict(X_te_h))
results["HAR-Ridge"] = evaluate("HAR-Ridge", y_te, y_pred)
preds["HAR-Ridge"] = y_pred

# 3. Ridge sur features RMT+Zeta uniquement
X_tr_r = df_tr[FEATURES_RMT].values; X_te_r = df_te[FEATURES_RMT].values
m = Ridge(alpha=2.0).fit(X_tr_r, y_tr_log)
y_pred = expo(m.predict(X_te_r))
results["Ridge-RMT"] = evaluate("Ridge-RMT", y_te, y_pred)
preds["Ridge-RMT"] = y_pred

# 4. Ridge tous features (RMT + vol historique)
FEATURES_ALL_LOG = FEATURES_RMT + FEATURES_VOL_LOG
X_tr_a = df_tr[FEATURES_ALL_LOG].values; X_te_a = df_te[FEATURES_ALL_LOG].values
m = Ridge(alpha=2.0).fit(X_tr_a, y_tr_log)
y_pred = expo(m.predict(X_te_a))
results["Ridge-ALL"] = evaluate("Ridge-ALL", y_te, y_pred)
preds["Ridge-ALL"] = y_pred

# 5. Random Forest (regularise : max_depth=5)
t0 = time.time()
m = RandomForestRegressor(n_estimators=100, max_depth=5, seed=42).fit(X_tr_a, y_tr_log)
y_pred = expo(m.predict(X_te_a))
results["RandomForest"] = evaluate("RandomForest", y_te, y_pred)
preds["RandomForest"] = y_pred
print(f"  RandomForest fit/predict : {time.time()-t0:.1f}s")

# 6. Gradient Boosting (XGBoost-light) regularise
t0 = time.time()
m = GradientBoostingRegressor(n_estimators=200, learning_rate=0.03,
                                max_depth=3, subsample=0.7, seed=42).fit(X_tr_a, y_tr_log)
y_pred = expo(m.predict(X_te_a))
results["GradBoost"] = evaluate("GradBoost", y_te, y_pred)
preds["GradBoost"] = y_pred
print(f"  GradBoost fit/predict : {time.time()-t0:.1f}s")

# 7. MLP-Adam (regularise : taille reduite + weight decay 1e-3)
t0 = time.time()
m = MLPRegressor(hidden_sizes=(32, 16), n_iter=300, lr=5e-4,
                  weight_decay=1e-3, seed=42).fit(X_tr_a, y_tr_log)
y_pred = expo(m.predict(X_te_a))
results["MLP-Adam"] = evaluate("MLP-Adam", y_te, y_pred)
preds["MLP-Adam"] = y_pred
print(f"  MLP-Adam fit/predict : {time.time()-t0:.1f}s")
loss_history_mlp = m.loss_history_

# 8. Monte Carlo kNN (sur features standardisees)
mu_x = X_tr_a.mean(0); sd_x = X_tr_a.std(0) + 1e-12
m = MonteCarloVolPredictor(k_neighbors=30, seed=42).fit(
    (X_tr_a - mu_x) / sd_x, y_tr_log)
y_pred = expo(m.predict((X_te_a - mu_x) / sd_x))
results["MonteCarlo-kNN"] = evaluate("MonteCarlo-kNN", y_te, y_pred)
preds["MonteCarlo-kNN"] = y_pred

# ----------------------------------------------------------------------
# Affichage & sauvegarde
# ----------------------------------------------------------------------
df_res = pd.DataFrame(results).T
df_res = df_res.set_index("model")
df_res = df_res.astype(float).round(4)
print("\n=== TABLEAU FINAL ===")
print(df_res.to_string())

df_res.to_csv(os.path.join(DATA_DIR, "ml_results.csv"))

# Sauve les predictions
df_preds = pd.DataFrame(preds, index=df_te.index)
df_preds["y_true"] = y_te
df_preds.to_csv(os.path.join(DATA_DIR, "ml_predictions.csv"))

# Sauvegarde de l'historique de perte MLP
np.savetxt(os.path.join(DATA_DIR, "mlp_loss_history.txt"), loss_history_mlp)

print("\nSaved: data/ml_results.csv, data/ml_predictions.csv")

# Identification du gagnant
best = df_res["RMSE"].idxmin()
print(f"\n*** MEILLEUR MODELE (RMSE) : {best} ***")
print(f"    RMSE = {df_res.loc[best, 'RMSE']:.4f}")
print(f"    R^2  = {df_res.loc[best, 'R2']:.4f}")
print(f"    QLIKE= {df_res.loc[best, 'QLIKE']:.4f}")

# ----------------------------------------------------------------------
# Analyse conditionnelle : performance par regime
# ----------------------------------------------------------------------
print("\n=== Analyse conditionnelle (par regime sur test set) ===\n")
df_cond = pd.DataFrame(preds, index=df_te.index)
df_cond["y_true"] = y_te
df_cond["regime"] = df_te["regime"].values
regimes_lab = {0: "CALME", 1: "NORMAL", 2: "STRESS", 3: "CRISE", 4: "RECOVERY"}

cond_rows = []
for r in sorted(df_cond["regime"].unique()):
    sub = df_cond[df_cond["regime"] == r]
    n_r = len(sub)
    if n_r < 5:
        continue
    for col in preds.keys():
        cond_rows.append({
            "regime": regimes_lab.get(r, f"R{r}"),
            "n_obs": n_r,
            "model": col,
            "RMSE": rmse(sub["y_true"].values, sub[col].values),
            "MAE":  mae(sub["y_true"].values,  sub[col].values),
            "R2":   r2(sub["y_true"].values,   sub[col].values),
        })
df_cond_res = pd.DataFrame(cond_rows)
print(df_cond_res.pivot(index="model", columns="regime", values="RMSE").round(3).to_string())

df_cond_res.to_csv(os.path.join(DATA_DIR, "ml_results_per_regime.csv"), index=False)

# Gain marginal RMT vs HAR seul (% reduction RMSE)
print("\n=== Gain marginal RMT vs HAR-Ridge seul (% reduction RMSE) ===")
gains = []
for col in preds.keys():
    if col == "HAR-Ridge": continue
    gain = (df_res.loc["HAR-Ridge", "RMSE"] - df_res.loc[col, "RMSE"]) / df_res.loc["HAR-Ridge", "RMSE"] * 100
    gains.append({"model": col, "gain_pct": round(float(gain), 2)})
print(pd.DataFrame(gains).set_index("model").to_string())
