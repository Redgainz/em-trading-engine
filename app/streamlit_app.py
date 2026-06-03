"""
Streamlit App — RMT-Zeta Volatility Engine
==========================================
Lance l'app en local :   streamlit run app/streamlit_app.py
Deploiement gratuit :    https://streamlit.io/cloud  (connecte ton repo GitHub)

L'app est multi-pages avec sidebar :
  1. Vue d'ensemble
  2. Explorateur de spectres
  3. Universalite GUE / Zeta de Riemann
  4. Comparaison de modeles
  5. Predictions hors echantillon
  6. Analyse par regime
"""
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Permet d'importer rmt_zeta + ml_models depuis paper-rmt-zeta/code/
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "paper-rmt-zeta", "code"))

from rmt_zeta import (correlation_matrix, eigen_spectrum, mp_bounds, mp_pdf,
                       mp_sigma2_fit, rmt_features, realized_vol,
                       wigner_gue_pdf, wigner_poisson_pdf,
                       unfold_spectrum, nn_spacings, zeta_universality_distance)

DATA = os.path.join(ROOT, "paper-rmt-zeta", "data")

# ------------------------------------------------------------
# Config et style
# ------------------------------------------------------------
st.set_page_config(
    page_title="RMT-Zeta Volatility Engine",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main { padding-top: 1rem; }
h1 { color: #1d3557; }
h2 { color: #1d3557; border-bottom: 2px solid #d62828; padding-bottom: 0.3rem; }
.stat-card {
    background: white; padding: 1rem; border-radius: 8px;
    border-left: 4px solid #d62828; margin-bottom: 1rem;
}
.stMetric { background: #f7f8fa; padding: 0.8rem; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Cache des donnees
# ------------------------------------------------------------
@st.cache_data
def load_data():
    df_R = pd.read_csv(os.path.join(DATA, "returns.csv"), index_col=0, parse_dates=True)
    df_reg = pd.read_csv(os.path.join(DATA, "regimes.csv"), index_col=0, parse_dates=True)
    df_feat = pd.read_csv(os.path.join(DATA, "features.csv"), index_col=0, parse_dates=True)
    df_pred = pd.read_csv(os.path.join(DATA, "ml_predictions.csv"), index_col=0, parse_dates=True)
    df_res = pd.read_csv(os.path.join(DATA, "ml_results.csv"), index_col=0)
    df_reg_res = pd.read_csv(os.path.join(DATA, "ml_results_per_regime.csv"))
    return df_R, df_reg, df_feat, df_pred, df_res, df_reg_res

df_R, df_reg, df_feat, df_pred, df_res, df_reg_res = load_data()

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.title("RMT-Zeta Engine")
st.sidebar.markdown("""
**Reda Mikou** — EDHEC MSc DAAI

Apprentissage Profond Robuste au Tail Risk.

Side paper : Universalite GUE & Zeta de Riemann pour la prediction multi-asset.
""")

page = st.sidebar.radio(
    "Navigation",
    ["1. Vue d'ensemble",
     "2. Explorateur de spectres",
     "3. Universalite GUE / Zeta",
     "4. Comparaison de modeles",
     "5. Predictions hors echantillon",
     "6. Analyse par regime"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("[Repo GitHub](https://github.com/) &middot; [Paper PDF](../paper-rmt-zeta/Reda_Mikou_Paper_RMT_Zeta.pdf)")

REGIME_NAMES = {0: "CALME", 1: "NORMAL", 2: "STRESS", 3: "CRISE", 4: "RECOVERY"}
REGIME_COLORS = {0: "#d4f1f9", 1: "#ffffff", 2: "#ffd5a8", 3: "#ffadad", 4: "#caffbf"}

# ============================================================
# PAGE 1 : VUE D'ENSEMBLE
# ============================================================
if page.startswith("1."):
    st.title("Universalite GUE, RMT et fonction zeta de Riemann")
    st.markdown("**Dashboard interactif pour la prediction de volatilite multi-asset**")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Series multi-asset", "60")
    col2.metric("Jours simules", "4000", "16 ans")
    col3.metric("Features RMT/Zeta", "13")
    col4.metric("Fenetres analysees", "772")

    st.markdown("---")

    st.header("Problematique")
    st.markdown("""
    La prediction de volatilite reste un probleme central de l'econometrie financiere.
    Les modeles autoregressifs heterogenes (HAR, Corsi 2009) demeurent la baseline difficile a battre.
    Mais sur des univers multi-asset, ils ignorent l'**information structurelle** contenue dans la
    matrice de correlation entre actifs.

    Ce projet teste l'hypothese suivante : **la deviation du spectre empirique a l'universalite GUE**
    (= la distribution attendue pour les eigenvalues d'une matrice hermitienne aleatoire, equivalente
    a la pair correlation des zeros de la fonction zeta de Riemann) est un signal predictif de volatilite,
    complementaire a la vol historique.
    """)

    st.header("Resultats hors echantillon")
    sorted_res = df_res.sort_values("RMSE")
    st.dataframe(
        sorted_res.style
            .format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "R2": "{:+.3f}", "QLIKE": "{:.4f}"})
            .background_gradient(subset=["RMSE"], cmap="RdYlGn_r"),
        use_container_width=True
    )
    best_model = sorted_res.index[0]
    har_rmse = float(df_res.loc["HAR-Ridge", "RMSE"])
    best_rmse = float(sorted_res.iloc[0]["RMSE"])
    gain = (har_rmse - best_rmse) / har_rmse * 100
    st.success(f"**Meilleur modele : {best_model}** — RMSE {best_rmse:.4f} ({gain:+.2f}% vs HAR-Ridge baseline {har_rmse:.4f})")

# ============================================================
# PAGE 2 : EXPLORATEUR DE SPECTRES
# ============================================================
elif page.startswith("2."):
    st.title("Explorateur de spectres")
    st.markdown("Selectionne une date pour calculer le spectre des 120 jours precedents en live.")

    available_dates = df_feat.index.tolist()
    selected_date = st.select_slider(
        "Date pivot",
        options=available_dates,
        value=available_dates[len(available_dates) // 2],
        format_func=lambda d: d.strftime("%Y-%m-%d")
    )

    # Recalcul live a partir de returns.csv
    t = df_R.index.searchsorted(selected_date)
    T_w = 120
    if t < T_w:
        st.warning("Pas assez d'historique avant cette date")
    else:
        R_past = df_R.iloc[t - T_w:t].values
        C = correlation_matrix(R_past)
        eig, V = eigen_spectrum(C)
        sigma2 = mp_sigma2_fit(eig, T_w, R_past.shape[1])
        lam_m, lam_p = mp_bounds(T_w, R_past.shape[1], sigma2)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("lambda_max", f"{eig[0]:.2f}")
        col2.metric("Var marche", f"{eig[0] / R_past.shape[1] * 100:.1f}%")
        col3.metric("Facteurs > lambda+", int(np.sum(eig > lam_p)))
        col4.metric("sigma2 bruit", f"{sigma2:.3f}")

        # Spectrum plot
        col_left, col_right = st.columns(2)
        with col_left:
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            ax.hist(eig[eig < 5], bins=25, density=True, alpha=0.65,
                     color="#3a86ff", edgecolor="white")
            grid = np.linspace(lam_m, lam_p, 200)
            ax.plot(grid, mp_pdf(grid, T_w, R_past.shape[1], sigma2),
                     color="#d62828", lw=2, label=f"MP fit (sigma2={sigma2:.2f})")
            ax.axvline(lam_p, color="#d62828", ls="--", alpha=0.6)
            ax.set_xlabel("lambda"); ax.set_ylabel("Densite")
            ax.set_title("Bulk du spectre vs Marchenko-Pastur")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        with col_right:
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            ax.semilogy(range(1, len(eig) + 1), eig, "o-", color="#3a86ff", ms=4)
            ax.axhline(lam_p, color="#d62828", ls="--", label=f"lambda+={lam_p:.2f}")
            ax.axhline(lam_m, color="#06a77d", ls="--", label=f"lambda-={lam_m:.2f}")
            ax.set_xlabel("Rang"); ax.set_ylabel("lambda (log)")
            ax.set_title("Tous les eigenvalues")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)

# ============================================================
# PAGE 3 : UNIVERSALITE GUE / ZETA
# ============================================================
elif page.startswith("3."):
    st.title("Universalite GUE & fonction zeta de Riemann")
    st.markdown("""
    La **conjecture de Montgomery-Odlyzko** etablit que les espacements normalises des zeros
    non-triviaux de la fonction zeta suivent la distribution de Wigner pour l'ensemble GUE.
    Nous testons cette propriete sur les eigenvalues empiriques de la matrice de correlation
    pour detecter les regimes de marche anormaux.
    """)

    available_dates = df_feat.index.tolist()
    selected_date = st.select_slider(
        "Date pivot",
        options=available_dates,
        value=available_dates[len(available_dates) // 2],
        format_func=lambda d: d.strftime("%Y-%m-%d"),
        key="dateGUE"
    )

    t = df_R.index.searchsorted(selected_date)
    T_w = 120
    if t < T_w:
        st.warning("Pas assez d'historique")
    else:
        R_past = df_R.iloc[t - T_w:t].values
        C = correlation_matrix(R_past)
        eig, _ = eigen_spectrum(C)
        z = zeta_universality_distance(eig)

        col1, col2, col3 = st.columns(3)
        col1.metric("D_KS_GUE", f"{z['D_KS_GUE']:.3f}",
                     help="Distance Kolmogorov-Smirnov a la surmise de Wigner-GUE")
        col2.metric("D_KS_Poisson", f"{z['D_KS_Poisson']:.3f}",
                     help="Distance KS a la distribution Poisson (levels decorreles)")
        col3.metric("Zeta-score", f"{z['zeta_score']:+.3f}",
                     help="log(D_Poisson / D_GUE). > 0 = proche GUE")

        unf = unfold_spectrum(eig)
        s = nn_spacings(unf); s = s / s.mean()
        grid = np.linspace(0, 4, 400)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(s, bins=18, density=True, alpha=0.55, color="#3a86ff",
                 edgecolor="white", label="Spacings empiriques")
        ax.plot(grid, wigner_gue_pdf(grid), color="#d62828", lw=2.2,
                 label="GUE / Zeta-Riemann (Wigner)")
        ax.plot(grid, wigner_poisson_pdf(grid), color="#06a77d", lw=2.2, ls="--",
                 label="Poisson (decorreles)")
        ax.set_xlabel("s (espacement normalise)")
        ax.set_ylabel("p(s)")
        ax.set_xlim(0, 4)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_title(f"Spacings unfolded vs Wigner-GUE — {selected_date.date()}")
        st.pyplot(fig)

        if z['zeta_score'] > 0:
            st.info(f"Zeta-score positif : le spectre est plus proche de l'universalite GUE "
                     f"que de Poisson. Compatible avec un bruit gaussien dominant.")
        else:
            st.warning(f"Zeta-score negatif : deviation vers Poisson. "
                        f"Signature d'une structure non-aleatoire (potentiellement contagion / crise).")

# ============================================================
# PAGE 4 : COMPARAISON DE MODELES
# ============================================================
elif page.startswith("4."):
    st.title("Comparaison de 6 modeles ML")
    st.markdown("Tous les modeles entraines sur le **meme split temporel** (75% train, 25% test).")

    col1, col2 = st.columns([2, 1])
    with col1:
        sorted_res = df_res.sort_values("RMSE")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = ["#06a77d" if m == "Ridge-ALL" else
                   "#a4161a" if m == "HAR-Ridge" else
                   "#adb5bd" if m == "Const-Mean" else "#3a86ff"
                   for m in sorted_res.index]
        bars = ax.bar(range(len(sorted_res)), sorted_res["RMSE"], color=colors, edgecolor="black", lw=0.5)
        ax.axhline(df_res.loc["HAR-Ridge", "RMSE"], color="#a4161a", ls="--",
                    label="HAR baseline")
        for i, (m, b) in enumerate(zip(sorted_res.index, bars)):
            ax.text(i, b.get_height() + 0.002, f"{b.get_height():.4f}",
                     ha="center", fontsize=8)
        ax.set_xticks(range(len(sorted_res)))
        ax.set_xticklabels(sorted_res.index, rotation=30, ha="right")
        ax.set_ylabel("RMSE")
        ax.set_title("RMSE par modele (193 obs hors echantillon)")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        st.pyplot(fig)

    with col2:
        st.markdown("**Tableau ordonne par RMSE :**")
        st.dataframe(
            sorted_res.style.format({"RMSE": "{:.4f}", "MAE": "{:.4f}",
                                       "R2": "{:+.3f}", "QLIKE": "{:.4f}"}),
            use_container_width=True
        )

# ============================================================
# PAGE 5 : PREDICTIONS HORS ECHANTILLON
# ============================================================
elif page.startswith("5."):
    st.title("Predictions hors echantillon")
    st.markdown("Compare jusqu'a 3 modeles a la vol realisee oracle.")

    available_models = [c for c in df_pred.columns if c != "y_true"]
    selected = st.multiselect(
        "Modeles a afficher",
        available_models,
        default=["HAR-Ridge", "Ridge-ALL", "MLP-Adam"]
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_pred.index, df_pred["y_true"] * 100,
             color="#1d3557", lw=2, label="Vol realisee (oracle)")
    for m in selected:
        ax.plot(df_pred.index, df_pred[m] * 100, lw=1.2, alpha=0.85, label=m)
    ax.set_ylabel("Vol annualisee (%)"); ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("Predictions vs realise (test set 2021-2025)")
    st.pyplot(fig)

# ============================================================
# PAGE 6 : ANALYSE PAR REGIME
# ============================================================
elif page.startswith("6."):
    st.title("Analyse par regime de marche")
    st.markdown("""
    **Message central du paper** : la valeur ajoutee des features RMT/Zeta n'est pas uniforme
    dans le temps. En regime **CRISE**, MLP-Adam reduit le RMSE de 4.4% par rapport a HAR seul.
    """)

    piv = df_reg_res.pivot(index="regime", columns="model", values="RMSE")
    piv = piv.reindex(["CALME", "NORMAL", "STRESS", "CRISE", "RECOVERY"])

    selected_models = st.multiselect(
        "Modeles a afficher",
        piv.columns.tolist(),
        default=["HAR-Ridge", "Ridge-ALL", "GradBoost", "MLP-Adam", "RandomForest"]
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(piv.index))
    w = 0.8 / len(selected_models)
    for i, m in enumerate(selected_models):
        ax.bar(x + (i - len(selected_models) / 2 + 0.5) * w,
                piv[m].values, width=w, label=m, edgecolor="black", lw=0.3)
    ax.set_xticks(x); ax.set_xticklabels(piv.index)
    ax.set_ylabel("RMSE"); ax.legend(ncol=3, fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("RMSE conditionnel par regime de marche")
    st.pyplot(fig)

    # Heatmap des gains vs HAR
    st.subheader("Gain marginal (%) versus HAR-Ridge")
    har = piv["HAR-Ridge"]
    gains = pd.DataFrame({m: (har - piv[m]) / har * 100 for m in piv.columns if m != "HAR-Ridge"})
    st.dataframe(
        gains.style.format("{:+.2f}%").background_gradient(cmap="RdYlGn", axis=None),
        use_container_width=True
    )
