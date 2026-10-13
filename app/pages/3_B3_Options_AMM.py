"""
B3 Options AMM — market-making haute frequence sur options actions bresiliennes
===============================================================================
Lance une seance du backtest en direct (donnees reelles PETR4 / VALE3 en cache)
avec la version de modele et les parametres choisis, puis affiche les KPI du desk :
edge, PnL retention, markouts, attribution greeks, qualite de quoting.
"""
import os
import sys
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
AMM_DIR = os.path.join(ROOT, "b3-options-amm")
sys.path.insert(0, AMM_DIR)

from amm import calibration, data  # noqa: E402
from amm.kpi import MARKOUT_HORIZONS_S, day_summary  # noqa: E402
from amm.params import FlowParams, HedgeParams  # noqa: E402
from amm.runner import run_detailed  # noqa: E402
from amm.setup_day import load_ticker_context  # noqa: E402

st.set_page_config(page_title="B3 Options AMM", page_icon=":zap:", layout="wide")
BLUE, ORANGE, AQUA, INK_2 = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"


@st.cache_resource
def context(ticker: str):
    return load_ticker_context(ticker)


@st.cache_data
def precomputed(name: str) -> pd.DataFrame | None:
    path = os.path.join(data.DATA_DIR, name)
    return pd.read_csv(path, index_col=0) if os.path.exists(path) else None


@st.cache_data(show_spinner="Simulation de la seance (~2 s)...")
def simulate(ticker: str, day_iso: str, version: str, overrides: tuple):
    cfg = calibration.all_versions()[version]
    o = dict(overrides)
    strategy = replace(cfg.strategy, gamma=o["gamma"], kappa=o["kappa"], flow_fade=o["flow_fade"],
                       as_widening=o["as_widening"], clip_contracts=o["clip"])
    hedge = HedgeParams(band_mode="fixed", band_shares=o["band"]) if o["band_override"] else cfg.hedge
    cfg = replace(cfg, strategy=strategy, hedge=hedge)
    day = pd.Timestamp(day_iso).date()
    res, market = run_detailed(ticker, day, cfg, FlowParams())
    return res, market, day_summary(res, market)


# ------------------------------------------------------------------ Sidebar
st.sidebar.title("B3 Options AMM")
tickers = data.load_market_meta()["tickers"]
ticker = st.sidebar.selectbox("Sous-jacent", tickers)
ctx = context(ticker)
day = st.sidebar.selectbox("Seance", list(ctx.days)[::-1], format_func=lambda d: d.strftime("%d/%m/%Y"))
versions = calibration.all_versions()
version = st.sidebar.selectbox("Version du modele", list(versions), index=len(versions) - 1)
base = versions[version]
st.sidebar.markdown("**Parametres**")
gamma = st.sidebar.select_slider("Aversion au risque gamma", options=[1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2],
                                 value=min([1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2], key=lambda g: abs(g - base.strategy.gamma)))
kappa = st.sidebar.slider("Elasticite du flux kappa", 0.8, 2.5, float(round(base.strategy.kappa, 2)), 0.05)
flow_fade = st.sidebar.slider("Fade sur flux (R$ spot / contrat-delta)", 0.0, 0.01, float(base.strategy.flow_fade), 0.001)
as_widening = st.sidebar.slider("Elargissement adverse selection phi", 0.0, 1.0, float(base.strategy.as_widening), 0.05)
clip = st.sidebar.select_slider("Clip (contrats)", options=[5, 10, 20], value=base.strategy.clip_contracts)
band_override = st.sidebar.checkbox("Forcer une bande de hedge fixe", value=False)
band = st.sidebar.slider("Bande de hedge (actions)", 200, 5000, 1000, 100, disabled=not band_override)

overrides = (("gamma", gamma), ("kappa", kappa), ("flow_fade", flow_fade), ("as_widening", as_widening),
             ("clip", clip), ("band", float(band)), ("band_override", band_override))
res, market, kpi = simulate(ticker, day.isoformat(), version, overrides)

# ------------------------------------------------------------------ Header
st.title("Market-making HFT sur options actions B3")
st.caption(f"{ticker} - seance du {day:%d/%m/%Y} - {market.book.n} options cotees sur "
           f"{len(set(market.book.expiries))} echeances - spot reel Yahoo 1 min, nappe calibree sur la chaine B3 reelle")
c = st.columns(6)
c[0].metric("PnL net (MTM)", f"R$ {kpi['pnl_mtm']:,.0f}")
c[1].metric("Edge capture", f"R$ {kpi['edge']:,.0f}")
c[2].metric("PnL retention", f"{kpi['retention']:.0%}" if np.isfinite(kpi["retention"]) else "-")
c[3].metric("Fills / contrats", f"{kpi['fills']:,} / {kpi['contracts']:,.0f}")
c[4].metric("Uptime deux-sens", f"{kpi['uptime']:.0%}")
c[5].metric("Max drawdown intraday", f"R$ {kpi['intraday_max_dd']:,.0f}")

tab_session, tab_kpi, tab_desk = st.tabs(["Seance", "KPI & markouts", "Iterations v0 -> v5"])

with tab_session:
    s, f = res.samples, res.fills
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1.3, 1]})
    axes[0].plot(np.arange(market.n_seconds) / 60, market.S, color=INK_2, linewidth=0.9)
    for side, color, label in ((1, BLUE, "achats"), (-1, ORANGE, "ventes")):
        sub = f[f["we_side"] == side]
        axes[0].scatter(sub["t"] / 60, sub["S"], s=7, color=color, label=label, linewidths=0)
    axes[0].set_ylabel("spot (R$)")
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].plot(s["t"] / 60, s["pnl_mid"], color=BLUE, label="PnL net (MTM)")
    axes[1].set_ylabel("R$")
    axes[1].legend(frameon=False, loc="upper left")
    axes[2].plot(s["t"] / 60, s["vega_volpt"], color=AQUA)
    axes[2].axhline(0, color=INK_2, linewidth=0.7)
    axes[2].set_ylabel("vega R$/pt")
    axes[2].set_xlabel("minutes depuis 10:00")
    for ax in axes:
        ax.grid(alpha=0.3)
    st.pyplot(fig)
    book = pd.DataFrame({"ticker": market.book.tickers, "strike": market.book.strikes,
                         "type": np.where(market.book.is_call, "CALL", "PUT"), "echeance": market.book.expiries,
                         "trades/jour (modele)": np.round(market.book.lambda_per_s * 2 * 24900).astype(int),
                         "position fin (contrats)": res.sample_q[-1].astype(int)})
    st.dataframe(book, width="stretch", hide_index=True)

with tab_kpi:
    left, right = st.columns(2)
    attribution = pd.Series({"edge capture": kpi["edge"], "frais B3": -kpi["fees"], "spread hedge": -kpi["hedge_cost"],
                             "delta": kpi["attr_delta"], "gamma": kpi["attr_gamma"], "vega": kpi["attr_vega"],
                             "theta": kpi["attr_theta"], "residu + timing": kpi["attr_residual"] + kpi["attr_timing"],
                             "PnL net": kpi["pnl_mtm"]}, name="R$")
    left.subheader("Attribution du PnL")
    left.dataframe(attribution.round(0).to_frame(), width="stretch")
    contracts = max(kpi["contracts"], 1)
    mk = pd.DataFrame({"horizon": ["fill", *[f"{h}s" for h in MARKOUT_HORIZONS_S]],
                       "brut R$/contrat": [kpi["edge"] / contracts, *[kpi[f"mk_{h}s"] / contracts for h in MARKOUT_HORIZONS_S]],
                       "delta-couvert R$/contrat": [kpi["edge"] / contracts, *[kpi[f"mkh_{h}s"] / contracts for h in MARKOUT_HORIZONS_S]]})
    right.subheader("Markouts")
    right.line_chart(mk.set_index("horizon"), color=[ORANGE, BLUE])
    right.dataframe(mk.round(3), width="stretch", hide_index=True)
    st.markdown(f"Spread cote moyen **{kpi['spread_ticks']:.2f} ticks** vs ecran **{kpi['screen_spread_ticks']:.2f}** - "
                f"temps au meilleur prix **{kpi['at_best']:.0%}** - part informee des fills **{kpi['informed_share']:.0%}** - "
                f"{kpi['hedges']} ordres de hedge ({kpi['shares_hedged']:,.0f} actions)")

with tab_desk:
    summary = precomputed("results_summary.csv")
    if summary is None:
        st.info("Lance `python b3-options-amm/scripts/04_backtest_iterations.py` pour generer le tableau.")
    else:
        cols = ["pnl_mtm", "edge", "retention", "sharpe_ann", "hit_ratio", "max_dd", "fills",
                "edge_per_contract", "mk60_per_contract", "attr_delta"]
        st.subheader("Desk PETR4 + VALE3, toutes seances (common random numbers)")
        st.dataframe(summary[cols].style.format({"retention": "{:.0%}", "hit_ratio": "{:.0%}", "sharpe_ann": "{:.1f}",
                                                 "edge_per_contract": "{:.2f}", "mk60_per_contract": "{:.2f}"},
                                                precision=0), width="stretch")
    for name in ("fig_03_iterations.png", "fig_05_markouts.png"):
        path = os.path.join(AMM_DIR, "figures", name)
        if os.path.exists(path):
            st.image(path, width="stretch")
