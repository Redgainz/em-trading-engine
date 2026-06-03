"""
05_figures_all.py
=================
Genere toutes les figures restantes du paper (4 -> 10).
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

OUT_DIR = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(OUT_DIR, "data")
FIG_DIR  = os.path.join(OUT_DIR, "figures")

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200,
    "font.size": 10, "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
})

# Donnees
df_R = pd.read_csv(os.path.join(DATA_DIR, "returns.csv"), index_col=0, parse_dates=True)
df_reg = pd.read_csv(os.path.join(DATA_DIR, "regimes.csv"), index_col=0, parse_dates=True)
df_feat = pd.read_csv(os.path.join(DATA_DIR, "features.csv"), index_col=0, parse_dates=True)
df_pred = pd.read_csv(os.path.join(DATA_DIR, "ml_predictions.csv"), index_col=0, parse_dates=True)
df_res  = pd.read_csv(os.path.join(DATA_DIR, "ml_results.csv"), index_col=0)
df_cond = pd.read_csv(os.path.join(DATA_DIR, "ml_results_per_regime.csv"))
loss_mlp = np.loadtxt(os.path.join(DATA_DIR, "mlp_loss_history.txt"))

# ============================================================
# Fig 4 : Heatmap de la matrice de correlation moyenne
# ============================================================
df_meta = pd.read_csv(os.path.join(DATA_DIR, "meta.csv"))
order = df_meta.sort_values(["asset_class", "sector"]).index.tolist()
tickers_ord = df_meta.iloc[order]["ticker"].tolist()
R_ord = df_R[tickers_ord]
# Correlation moyenne sur toute la periode
C_full = R_ord.corr().values
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(C_full, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"Correlation $\rho_{ij}$")
# Trait de separation entre asset classes
classes = df_meta.iloc[order]["asset_class"].tolist()
boundaries = [i for i in range(1, len(classes)) if classes[i] != classes[i-1]]
for b in boundaries:
    ax.axhline(b - 0.5, color="black", lw=0.7)
    ax.axvline(b - 0.5, color="black", lw=0.7)
# Labels des classes
prev = 0; cls_pos = []
for b in boundaries + [len(classes)]:
    mid = (prev + b) / 2
    cls_pos.append((mid, classes[prev]))
    prev = b
ax.set_xticks([p for p, _ in cls_pos]); ax.set_yticks([p for p, _ in cls_pos])
ax.set_xticklabels([c for _, c in cls_pos], rotation=0)
ax.set_yticklabels([c for _, c in cls_pos])
ax.set_title("Matrice de correlation moyenne 2010-2025 (60 actifs)")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_04_corr_heatmap.png"), bbox_inches="tight")
plt.close(fig)
print("Saved: fig_04_corr_heatmap.png")

# ============================================================
# Fig 5 : Evolution temporelle des features RMT-Zeta avec regimes
# ============================================================
fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
# Colorier le fond selon le regime
regime_colors = {0: "#d4f1f9", 1: "#ffffff", 2: "#ffd5a8", 3: "#ffadad", 4: "#caffbf"}
regime_names  = {0: "CALME", 1: "NORMAL", 2: "STRESS", 3: "CRISE", 4: "RECOVERY"}

# On utilise les regimes alignes sur l'index des features
reg_aligned = df_reg.reindex(df_feat.index)["regime"].fillna(method="ffill").astype(int).values

def shade_regimes(ax):
    """Ombre le fond suivant le regime."""
    dts = df_feat.index
    cur = reg_aligned[0]; start = 0
    for i in range(1, len(reg_aligned)):
        if reg_aligned[i] != cur:
            ax.axvspan(dts[start], dts[i], color=regime_colors[cur], alpha=0.55, zorder=0)
            cur = reg_aligned[i]; start = i
    ax.axvspan(dts[start], dts[-1], color=regime_colors[cur], alpha=0.55, zorder=0)

for ax in axes:
    shade_regimes(ax)

axes[0].plot(df_feat.index, df_feat["lambda_max"], color="#1d3557", lw=1.1)
axes[0].set_ylabel(r"$\lambda_{\max}$")
axes[0].set_title("Top eigenvalue (mode marche) — concentration de la variance")

axes[1].plot(df_feat.index, df_feat["var_market"]*100, color="#1d3557", lw=1.1)
axes[1].set_ylabel(r"% var. expliquee")
axes[1].set_title(r"Variance expliquee par le mode marche : $\lambda_{\max}/N$")

axes[2].plot(df_feat.index, df_feat["zeta_score"], color="#a4161a", lw=1.1)
axes[2].axhline(0, color="black", lw=0.5, ls="--")
axes[2].set_ylabel("zeta_score")
axes[2].set_title(r"\textbf{Zeta-score} : $\log(D_{\text{KS-Poisson}}/D_{\text{KS-GUE}})$ — universalite GUE")

axes[3].plot(df_feat.index, df_feat["vol_future_21"]*100, color="#7209b7", lw=1.1)
axes[3].set_ylabel(r"Vol 21j fwd (\%)")
axes[3].set_xlabel("Date")
axes[3].set_title(r"Cible : volatilite realisee future 21j annualisee (\%)")

# Legende des regimes (un seul rectangle par regime)
from matplotlib.patches import Patch
handles = [Patch(facecolor=regime_colors[r], edgecolor="black", alpha=0.7, label=regime_names[r])
            for r in [0,1,2,3,4]]
axes[0].legend(handles=handles, loc="upper left", fontsize=8, ncol=5)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_05_features_timeseries.png"), bbox_inches="tight")
plt.close(fig)
print("Saved: fig_05_features_timeseries.png")

# ============================================================
# Fig 6 : Scatter plots features RMT vs vol future avec correlation
# ============================================================
features_show = [("lambda_max", r"$\lambda_{\max}$"),
                  ("var_market", r"$\lambda_{\max}/N$"),
                  ("D_KS_GUE",   r"$D_{KS-GUE}$"),
                  ("zeta_score", r"zeta-score"),
                  ("participation_ratio", "PR (mode marche)"),
                  ("sigma2_noise", r"$\sigma^2_{noise}$")]

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, (col, label) in zip(axes.flat, features_show):
    x = df_feat[col].values
    y = df_feat["vol_future_21"].values * 100
    # Couleurs selon regime
    cols_pts = [regime_colors[reg_aligned[i]] for i in range(len(df_feat))]
    ax.scatter(x, y, s=8, c=cols_pts, edgecolor="#444", lw=0.2, alpha=0.85)
    # Trend line
    z = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, np.polyval(z, xs), color="#d62828", lw=1.2)
    rho = np.corrcoef(x, y)[0, 1]
    ax.set_title(fr"{label} vs Vol future (correlation = {rho:+.2f})", fontsize=10)
    ax.set_xlabel(label)
    ax.set_ylabel("Vol future 21j (\%)")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_06_scatter_features.png"), bbox_inches="tight")
plt.close(fig)
print("Saved: fig_06_scatter_features.png")

# ============================================================
# Fig 7 : Predictions vs realised - 4 modeles
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
models_show = ["HAR-Ridge", "Ridge-ALL", "GradBoost", "MLP-Adam"]
for ax, name in zip(axes.flat, models_show):
    ax.plot(df_pred.index, df_pred["y_true"]*100, color="#1d3557", lw=1.4,
            label="Vol realisee (oracle)")
    ax.plot(df_pred.index, df_pred[name]*100, color="#d62828", lw=1.0, alpha=0.85,
            label=f"Prediction {name}")
    rmse_val = float(df_res.loc[name, "RMSE"])
    r2_val   = float(df_res.loc[name, "R2"])
    ax.set_title(fr"{name} — RMSE = {rmse_val:.4f}, $R^2$ = {r2_val:+.3f}")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylabel("Vol 21j (\%)")
axes[1, 0].set_xlabel("Date"); axes[1, 1].set_xlabel("Date")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_07_predictions_vs_real.png"), bbox_inches="tight")
plt.close(fig)
print("Saved: fig_07_predictions_vs_real.png")

# ============================================================
# Fig 8 : Bar plot RMSE par modele
# ============================================================
fig, ax = plt.subplots(figsize=(9, 4.5))
models_ord = df_res.sort_values("RMSE").index.tolist()
colors_bars = ["#06a77d" if m == "Ridge-ALL" else
                "#3a86ff" if m in ["Ridge-RMT", "Ridge-ALL"] else
                "#a4161a" if m in ["Const-Mean", "HAR-Ridge"] else
                "#6c757d" for m in models_ord]
bars = ax.bar(range(len(models_ord)), df_res.loc[models_ord, "RMSE"],
              color=colors_bars, edgecolor="black", lw=0.5)
for i, (m, b) in enumerate(zip(models_ord, bars)):
    ax.text(i, b.get_height() + 0.002, f"{b.get_height():.4f}",
             ha="center", va="bottom", fontsize=8)
ax.axhline(df_res.loc["HAR-Ridge", "RMSE"], color="#a4161a", ls="--", lw=1,
            label="Baseline HAR-Ridge")
ax.set_xticks(range(len(models_ord)))
ax.set_xticklabels(models_ord, rotation=30, ha="right")
ax.set_ylabel("RMSE (vol annualisee)")
ax.set_title("Comparaison des modeles : RMSE sur l'ensemble de test (193 obs)")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_08_rmse_bar.png"), bbox_inches="tight")
plt.close(fig)
print("Saved: fig_08_rmse_bar.png")

# ============================================================
# Fig 9 : RMSE par regime (analyse conditionnelle)
# ============================================================
piv = df_cond.pivot(index="regime", columns="model", values="RMSE")
fig, ax = plt.subplots(figsize=(11, 5))
models_pick = ["HAR-Ridge", "Ridge-ALL", "GradBoost", "MLP-Adam", "RandomForest"]
piv_show = piv[models_pick]
piv_show = piv_show.reindex(["CALME", "NORMAL", "STRESS", "CRISE", "RECOVERY"])
x = np.arange(len(piv_show.index))
w = 0.16
colors = ["#a4161a", "#06a77d", "#3a86ff", "#7209b7", "#fb8500"]
for i, m in enumerate(models_pick):
    if m not in piv_show.columns: continue
    ax.bar(x + (i - 2) * w, piv_show[m].values, width=w,
            label=m, color=colors[i], edgecolor="black", lw=0.4)
ax.set_xticks(x); ax.set_xticklabels(piv_show.index)
ax.set_ylabel("RMSE conditionnel au regime")
ax.set_title("Performance conditionnelle au regime de marche (test set)")
ax.legend(ncol=5, fontsize=9, loc="upper left")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_09_rmse_by_regime.png"), bbox_inches="tight")
plt.close(fig)
print("Saved: fig_09_rmse_by_regime.png")

# ============================================================
# Fig 10 : Convergence MLP-Adam
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(loss_mlp, color="#1d3557", lw=1)
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (log-vol)")
ax.set_yscale("log")
ax.set_title("Convergence de l'optimiseur Adam — MLP (32, 16)")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_10_mlp_convergence.png"), bbox_inches="tight")
plt.close(fig)
print("Saved: fig_10_mlp_convergence.png")

print("\n=== Toutes les figures generees ===")
