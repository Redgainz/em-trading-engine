"""
02_simple_example.py
====================
Exemple pedagogique sur UNE seule fenetre de 120 jours.

Etapes :
 1. Charger les rendements
 2. Choisir une date pivot (e.g. juste avant la crise COVID)
 3. Extraire les features RMT + Zeta sur les 120 jours qui precedent
 4. Calculer la vol realisee sur les 21 jours suivants
 5. Comparer avec une prediction simple : sigma_pred = f(lambda_max / N, vol_implicite_proxy)
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
from rmt_zeta import (correlation_matrix, eigen_spectrum, mp_bounds, mp_pdf,
                       mp_sigma2_fit, rmt_features, realized_vol,
                       wigner_gue_pdf, wigner_poisson_pdf,
                       unfold_spectrum, nn_spacings)

OUT_DIR = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(OUT_DIR, "data")
FIG_DIR  = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200,
    "font.size": 10, "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
})

# Charger
df_R = pd.read_csv(os.path.join(DATA_DIR, "returns.csv"), index_col=0, parse_dates=True)
df_reg = pd.read_csv(os.path.join(DATA_DIR, "regimes.csv"), index_col=0, parse_dates=True)
T, N = df_R.shape
print(f"Loaded: {T} jours x {N} actifs")

# Choisir une date pivot : on cherche un index ou le regime est CALME ou NORMAL
# juste avant qu'il passe en STRESS/CRISE -> permet de tester la valeur
# predictive du framework.
T_w = 120     # fenetre de calcul features
T_f = 21      # horizon de vol future

# Cherche l'evenement le plus dramatique : transition NORMAL -> CRISE (regime 3)
# avec le plus grand saut de vol future / vol passee.
regimes = df_reg["regime"].values
idx_returns_full = df_R.mean(axis=1).values
candidates = []
for t in range(T_w + 1, T - T_f - 1):
    if regimes[t-1] in (0, 1) and regimes[t:t+10].max() >= 3:
        vp = idx_returns_full[t - T_w : t].std()
        vf = idx_returns_full[t : t + T_f].std()
        candidates.append((vf / vp, t))
candidates.sort(reverse=True)
idx_pivot = candidates[0][1]
print(f"\nMeilleur pivot (saut de vol max) : ratio = {candidates[0][0]:.2f}")

date_pivot = df_R.index[idx_pivot]
print(f"\nDate pivot : {date_pivot.date()}  (regime t-1 = {regimes[idx_pivot-1]} -> regime t..t+5 max = {regimes[idx_pivot:idx_pivot+5].max()})")

# Fenetre passee + future
R_past   = df_R.iloc[idx_pivot - T_w : idx_pivot].values
R_future = df_R.iloc[idx_pivot : idx_pivot + T_f].values

# Features RMT + Zeta
feats = rmt_features(R_past)
print("\n=== Features extraites sur les 120j precedant la date pivot ===")
for k, v in feats.items():
    print(f"  {k:25s}: {v:.4f}")

# Vol realisee future
vol_future = realized_vol(R_future, annualize=True)
print(f"\nVol realisee 21j futurs (annualisee) : {vol_future*100:.2f}%")

# Vol realisee passee (benchmark)
vol_past = realized_vol(R_past, annualize=True)
print(f"Vol realisee 120j passes  (annualisee) : {vol_past*100:.2f}%")

# Prediction naive : modele lineaire empirique entre lambda_max et vol future
# (calibre plus tard, ici on illustre)
# Heuristique : sigma_pred = sqrt(lambda_max / N) * sqrt(sigma2_noise) * sigma_base * sqrt(252)
sigma_pred_naive = np.sqrt(feats["lambda_max"] / N) * np.sqrt(feats["sigma2_noise"]) * R_past.std() * np.sqrt(252)
print(f"\nPrediction naive RMT (sans ML) : {sigma_pred_naive*100:.2f}%")

# ----------------------------------------------------------------------
# FIGURES
# ----------------------------------------------------------------------
C = correlation_matrix(R_past)
eig, V = eigen_spectrum(C)
sigma2 = feats["sigma2_noise"]
lam_m, lam_p = mp_bounds(T_w, N, sigma2)

# Fig 1 : spectre eigenvalues + MP fit
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# (a) histogramme spectre vs MP
ax = axes[0]
ax.hist(eig[eig < 5], bins=30, density=True, alpha=0.65, color="#3a86ff",
        edgecolor="white", label=r"Spectre empirique (bulk)")
grid = np.linspace(lam_m, lam_p, 200)
ax.plot(grid, mp_pdf(grid, T_w, N, sigma2), color="#d62828", lw=2,
        label=fr"Marchenko-Pastur ($\sigma^2={sigma2:.2f}$)")
ax.axvline(lam_p, color="#d62828", ls="--", alpha=0.6, label=r"$\lambda_+$")
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"Densit\'e")
ax.set_title(f"(a) Spectre vs Marchenko-Pastur — fenetre 120j")
ax.legend(fontsize=8, loc="upper right")

# (b) all eigenvalues (log y) avec annotations
ax = axes[1]
ax.semilogy(range(1, N+1), eig, "o-", color="#3a86ff", ms=5, lw=1)
ax.axhline(lam_p, color="#d62828", ls="--", label=fr"$\lambda_+$={lam_p:.2f}")
ax.axhline(lam_m, color="#06a77d", ls="--", label=fr"$\lambda_-$={lam_m:.2f}")
# Annotate top 3
for i in range(3):
    ax.annotate(f"$\\lambda_{i+1}$={eig[i]:.1f}", xy=(i+1, eig[i]),
                xytext=(i+1+2, eig[i]*1.5), fontsize=9)
ax.set_xlabel("Rang"); ax.set_ylabel(r"$\lambda$ (log)")
ax.set_title(f"(b) {feats['n_factors']:.0f} facteur(s) au-dessus de $\\lambda_+$")
ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_01_spectrum_MP.png"), bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: figures/fig_01_spectrum_MP.png")

# Fig 2 : distribution des spacings vs GUE et Poisson
fig, ax = plt.subplots(figsize=(7, 4.5))
unf = unfold_spectrum(eig)
s = nn_spacings(unf)
s = s / s.mean()
grid_s = np.linspace(0, 4, 400)
ax.hist(s, bins=20, density=True, alpha=0.55, color="#3a86ff", edgecolor="white",
        label=fr"Spacings empiriques (N$-$1 = {N-1})")
ax.plot(grid_s, wigner_gue_pdf(grid_s), color="#d62828", lw=2.2,
        label=r"GUE / Zeta-Riemann (Wigner surmise)")
ax.plot(grid_s, wigner_poisson_pdf(grid_s), color="#06a77d", lw=2.2, ls="--",
        label=r"Poisson (levels d\'ecorr\'el\'es)")
ax.set_xlabel(r"$s$ (espacement normalis\'e)")
ax.set_ylabel(r"$p(s)$")
ax.set_title(f"Statistique de Montgomery-Odlyzko : D$_{{KS-GUE}}$ = {feats['D_KS_GUE']:.3f}, D$_{{KS-Poisson}}$ = {feats['D_KS_Poisson']:.3f}")
ax.legend(fontsize=9, loc="upper right")
ax.set_xlim(0, 4)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_02_spacings_GUE.png"), bbox_inches="tight")
plt.close(fig)
print("Saved: figures/fig_02_spacings_GUE.png")

# Fig 3 : visualisation contexte temporel
fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)

# Top: cumulative return de l'index equally-weighted
idx_ret = df_R.mean(axis=1)
cum = (1 + idx_ret).cumprod()
ax = axes[0]
ax.plot(df_R.index, cum, color="#1d3557", lw=1.0)
ax.axvspan(df_R.index[idx_pivot - T_w], df_R.index[idx_pivot], color="#3a86ff", alpha=0.18,
           label="Fenetre 120j (features)")
ax.axvspan(df_R.index[idx_pivot], df_R.index[idx_pivot + T_f], color="#d62828", alpha=0.18,
           label="21j futurs (vol cible)")
ax.set_ylabel("Index cumulatif")
ax.legend(loc="upper left", fontsize=9)
ax.set_title(f"Contexte : date pivot = {date_pivot.date()}")

# Bottom: vol roulante 21j
roll_vol = idx_ret.rolling(21).std() * np.sqrt(252)
ax = axes[1]
ax.plot(df_R.index, roll_vol*100, color="#a4161a", lw=1.0)
ax.axvspan(df_R.index[idx_pivot - T_w], df_R.index[idx_pivot], color="#3a86ff", alpha=0.18)
ax.axvspan(df_R.index[idx_pivot], df_R.index[idx_pivot + T_f], color="#d62828", alpha=0.18)
ax.axhline(vol_past*100, color="#3a86ff", ls="--", alpha=0.7, label=f"Vol 120j passe = {vol_past*100:.1f}%")
ax.axhline(vol_future*100, color="#d62828", ls="--", alpha=0.7, label=f"Vol 21j futur = {vol_future*100:.1f}%")
ax.set_ylabel("Vol 21j (%)"); ax.set_xlabel("Date")
ax.legend(loc="upper left", fontsize=9)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_03_context.png"), bbox_inches="tight")
plt.close(fig)
print("Saved: figures/fig_03_context.png")

# Sauve les features pour reference
with open(os.path.join(OUT_DIR, "data", "simple_example_features.txt"), "w") as fh:
    fh.write(f"Date pivot : {date_pivot.date()}\n")
    fh.write(f"Vol future realisee : {vol_future*100:.2f}%\n")
    fh.write(f"Vol passee : {vol_past*100:.2f}%\n")
    fh.write(f"Prediction naive RMT : {sigma_pred_naive*100:.2f}%\n\n")
    fh.write("=== Features ===\n")
    for k, v in feats.items():
        fh.write(f"  {k:25s}: {v:.6f}\n")
print(f"\nSaved: data/simple_example_features.txt")
