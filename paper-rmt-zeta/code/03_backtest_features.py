"""
03_backtest_features.py
=======================
Extraction des features RMT + Zeta sur fenetre glissante T_w = 120 jours,
target = vol realisee future sur T_f = 21 jours, step = 5 jours.

Sortie : data/features.csv (un row par fenetre)
"""
import os, sys, time
import numpy as np
import pandas as pd

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
from rmt_zeta import rmt_features, realized_vol

OUT_DIR = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(OUT_DIR, "data")

df_R = pd.read_csv(os.path.join(DATA_DIR, "returns.csv"), index_col=0, parse_dates=True)
df_reg = pd.read_csv(os.path.join(DATA_DIR, "regimes.csv"), index_col=0, parse_dates=True)
T, N = df_R.shape
print(f"Loaded: {T} jours x {N} actifs")

T_w = 120     # fenetre features
T_f = 21      # horizon de vol future
STEP = 5      # pas de la fenetre glissante

rows = []
t_start = time.time()
n_steps = (T - T_w - T_f) // STEP

for k, t in enumerate(range(T_w, T - T_f, STEP)):
    R_past   = df_R.iloc[t - T_w : t].values
    R_future = df_R.iloc[t : t + T_f].values
    feats = rmt_features(R_past)
    feats["date"] = df_R.index[t]
    feats["vol_past_120"] = realized_vol(R_past, annualize=True)
    feats["vol_future_21"] = realized_vol(R_future, annualize=True)
    feats["regime"] = int(df_reg["regime"].iloc[t])
    # Vol "garch-like" : EWMA sur 60 jours
    idx = R_past.mean(axis=1)
    weights = 0.94 ** np.arange(T_w)[::-1]
    weights /= weights.sum()
    ew_var = np.sum(weights * (idx - idx.mean()) ** 2)
    feats["vol_ewma_60"] = float(np.sqrt(ew_var * 252))
    rows.append(feats)
    if k % 50 == 0:
        elapsed = time.time() - t_start
        eta = elapsed / max(k, 1) * (n_steps - k)
        print(f"  step {k:4d}/{n_steps} ({df_R.index[t].date()}) — elapsed {elapsed:.1f}s, ETA {eta:.0f}s")

df_feat = pd.DataFrame(rows).set_index("date")
df_feat.to_csv(os.path.join(DATA_DIR, "features.csv"))
print(f"\nSaved features.csv : {df_feat.shape}")
print(f"Total time : {time.time() - t_start:.1f}s")

# Sanity
print("\n=== Stats features ===")
print(df_feat[["lambda_max", "var_market", "n_factors", "TW_z", "zeta_score",
               "vol_past_120", "vol_future_21", "regime"]].describe().round(3))

# Correlation features <-> target
print("\n=== Correlation entre features et vol_future_21 ===")
cor = df_feat.corr()["vol_future_21"].sort_values(ascending=False)
print(cor.to_string())
