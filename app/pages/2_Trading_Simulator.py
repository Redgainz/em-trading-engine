"""
Trading Simulator
=================
Page Streamlit du simulateur de portefeuille mois par mois.

Layout :
  - Sidebar : controles globaux (capital initial, reset, settings)
  - Header : date courante, valeur portfolio, P&L, regime detect
  - LEFT (3) : Market overview (prix, vol, regime, GARCH/EWMA)
  - CENTER (3) : Position builder (selection instrument + quantite + execute)
  - RIGHT (3) : Analytics (Greeks portfolio, SABR surface, VaR/CVaR)
  - BOTTOM : P&L history, positions table, transactions
"""
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ajoute lib/ au path
HERE = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(APP_DIR)
sys.path.insert(0, APP_DIR)

from lib.market_state import (MarketSimulator, INSTRUMENT_UNIVERSE,
                                OPTION_TO_UNDERLYING, FOREIGN_RATES,
                                SABR_PARAMS, VOL_BASE)
from lib.portfolio import Portfolio, Position
from lib.quant_engine import (bs_greeks, sabr_implied_vol, historical_var,
                                historical_cvar, nelson_siegel_curve)

DATA_DIR = os.path.join(ROOT, "paper-rmt-zeta", "data")

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Trading Simulator",
                    page_icon=":bar_chart:", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 100%; }
[data-testid="stMetricValue"] { font-size: 1.4rem; }
[data-testid="stMetricLabel"] { font-size: 0.75rem; }
.small-table table { font-size: 0.85rem; }
div.stMetric { background: #f7f8fa; padding: 0.5rem; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

REGIME_NAMES = {0: "CALME", 1: "NORMAL", 2: "CRISE"}
REGIME_COLORS = {0: "#06a77d", 1: "#fb8500", 2: "#a4161a"}

# ============================================================
# INIT SESSION STATE
# ============================================================
if "simulator" not in st.session_state:
    st.session_state.simulator = MarketSimulator(
        data_dir=DATA_DIR, start_date="2024-01-01", seed=int(np.random.randint(0, 1e6))
    )
    st.session_state.portfolio = Portfolio(initial_cash=1_000_000.0)
    st.session_state.month_count = 0
    st.session_state.current_state = st.session_state.simulator.get_market_state()
    # Snapshot mois 0
    st.session_state.portfolio.snapshot(st.session_state.current_state,
                                          st.session_state.current_state["date"])

sim: MarketSimulator = st.session_state.simulator
pf: Portfolio = st.session_state.portfolio
state = st.session_state.current_state

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Trading Simulator")
st.sidebar.markdown("Construis un portefeuille mois par mois sur 2024-2025.")

if st.sidebar.button("Reset complet (nouveau seed)"):
    seed = int(np.random.randint(0, 1e6))
    st.session_state.simulator = MarketSimulator(
        data_dir=DATA_DIR, start_date="2024-01-01", seed=seed
    )
    st.session_state.portfolio = Portfolio(initial_cash=1_000_000.0)
    st.session_state.month_count = 0
    st.session_state.current_state = st.session_state.simulator.get_market_state()
    st.session_state.portfolio.snapshot(st.session_state.current_state,
                                           st.session_state.current_state["date"])
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Mois ecoules** : {st.session_state.month_count}")
st.sidebar.markdown(f"**Date** : {state['date'].date()}")
st.sidebar.markdown(f"**Regime** : {REGIME_NAMES[state['regime_idx']]}")

if st.sidebar.button("Avancer 1 mois", type="primary"):
    st.session_state.current_state = sim.step_one_month()
    # Decrement options maturity de 1 mois et settle si expire
    pf.age_positions(1.0 / 12.0,
                      st.session_state.current_state["spot"],
                      st.session_state.current_state["date"])
    pf.snapshot(st.session_state.current_state,
                 st.session_state.current_state["date"])
    st.session_state.month_count += 1
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Cout de transaction** : 5 bps")
st.sidebar.markdown("**Capital initial** : $1,000,000")

# ============================================================
# HEADER : METRICS
# ============================================================
mtm = pf.mark_to_market(state, state["date"])
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Date", state["date"].strftime("%Y-%m"))
col2.metric("Cash", f"${pf.cash:,.0f}")
col3.metric("MTM positions", f"${mtm['mtm_positions']:,.0f}")
col4.metric("Total value", f"${mtm['total_value']:,.0f}",
             f"${mtm['pnl_total']:+,.0f}")
col5.metric("Positions", f"{len(pf.positions)}")

# ============================================================
# MAIN LAYOUT : 3 COLUMNS
# ============================================================
col_left, col_center, col_right = st.columns([1.1, 1.2, 1.2])

# ------------------------------------------------------------
# LEFT : MARKET OVERVIEW
# ------------------------------------------------------------
with col_left:
    st.subheader("Marche")

    # Regime indicator
    regime_idx = state["regime_idx"]
    st.markdown(
        f"<div style='background:{REGIME_COLORS[regime_idx]}; padding:0.4rem; "
        f"border-radius:5px; color:white; font-weight:600; text-align:center;'>"
        f"REGIME : {REGIME_NAMES[regime_idx]}</div>", unsafe_allow_html=True
    )
    regime_probs = state["regime_probs"]
    st.caption(f"Probas regime (next month) : CALME={regime_probs[0]:.0%} "
                f"| NORMAL={regime_probs[1]:.0%} | CRISE={regime_probs[2]:.0%}")

    st.markdown("**Prix spot actuels**")
    spot_rows = []
    for ticker, asset_class, label, _ in INSTRUMENT_UNIVERSE:
        if asset_class == "OPTION_FX":
            continue
        spot_rows.append({
            "Ticker": ticker, "Class": asset_class,
            "Prix": f"{state['spot'][ticker]:.4f}" if asset_class == "FX"
                     else f"{state['spot'][ticker]:.2f}",
        })
    st.dataframe(pd.DataFrame(spot_rows), use_container_width=True,
                  hide_index=True, height=240)

    # Prix options
    st.markdown("**Prix options**")
    opt_rows = []
    for opt_ticker, (under, kind, T) in OPTION_TO_UNDERLYING.items():
        opt_rows.append({
            "Option": opt_ticker.replace("_ATM", ""),
            "Spot": state["spot"][opt_ticker],
            "IV": f"{state['iv'][opt_ticker]*100:.1f}%",
        })
    st.dataframe(pd.DataFrame(opt_rows), use_container_width=True,
                  hide_index=True, height=200)

    # Mini chart : selected instrument
    plot_ticker = st.selectbox("Visualiser historique",
                                 [t for t, c, _, _ in INSTRUMENT_UNIVERSE if c != "OPTION_FX"],
                                 key="plot_ticker")
    hist = sim.history.get(plot_ticker, [])
    if len(hist) > 10:
        fig, ax = plt.subplots(figsize=(5, 2.2))
        ax.plot(hist[-252:], color="#1d3557", lw=1.1)
        ax.set_title(f"{plot_ticker} (252j)", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        st.pyplot(fig, use_container_width=True)

# ------------------------------------------------------------
# CENTER : POSITION BUILDER
# ------------------------------------------------------------
with col_center:
    st.subheader("Position builder")
    st.markdown("Selectionne un instrument, une action et une quantite.")

    with st.form("trade_form"):
        all_tickers = [t for t, c, _, _ in INSTRUMENT_UNIVERSE]
        selected = st.selectbox("Instrument", all_tickers,
                                 format_func=lambda t: f"{t}  ({next(c for tt,c,_,_ in INSTRUMENT_UNIVERSE if tt==t)})")
        # Get info
        asset_class = next(c for tt, c, _, _ in INSTRUMENT_UNIVERSE if tt == selected)
        cur_price = state["spot"][selected]
        st.caption(f"Prix actuel : **{cur_price:.4f}**" if asset_class == "FX"
                    else f"Prix actuel : **{cur_price:.2f}**")

        col_a, col_b = st.columns(2)
        with col_a:
            action = st.radio("Action", ["BUY", "SELL"], horizontal=True)
        with col_b:
            qty = st.number_input("Quantite", min_value=1.0, step=100.0, value=1000.0)

        notional = qty * cur_price
        st.caption(f"Notional : **${notional:,.0f}**  ({notional/pf.cash*100:.1f}% du cash)")

        submit = st.form_submit_button("Executer", type="primary")
        if submit:
            try:
                if action == "BUY":
                    if asset_class == "OPTION_FX":
                        under, kind, T = OPTION_TO_UNDERLYING[selected]
                        K = state["spot"][under] * np.exp(
                            (state["r_d"] - state["r_f"][under.split("/")[0]]) * T
                        )
                        pf.buy(selected, asset_class, qty, cur_price, state["date"],
                                strike=K, maturity_years=T, option_kind=kind,
                                underlying_ticker=under)
                    else:
                        pf.buy(selected, asset_class, qty, cur_price, state["date"])
                    st.success(f"Achete {qty:.0f} de {selected} a {cur_price:.4f}")
                else:
                    if asset_class == "OPTION_FX":
                        under, kind, T = OPTION_TO_UNDERLYING[selected]
                        K = state["spot"][under] * np.exp(
                            (state["r_d"] - state["r_f"][under.split("/")[0]]) * T
                        )
                        pf.sell(selected, asset_class, qty, cur_price, state["date"],
                                 strike=K, maturity_years=T, option_kind=kind,
                                 underlying_ticker=under)
                    else:
                        pf.sell(selected, asset_class, qty, cur_price, state["date"])
                    st.success(f"Vendu {qty:.0f} de {selected} a {cur_price:.4f}")
                st.rerun()
            except ValueError as e:
                st.error(str(e))

    # Quick close all
    if st.button("Fermer toutes les positions"):
        pf.close_all({pos.instrument: state["spot"][pos.instrument]
                       for pos in pf.positions if pos.instrument in state["spot"]},
                       state["date"])
        st.rerun()

# ------------------------------------------------------------
# RIGHT : ANALYTICS PANEL
# ------------------------------------------------------------
with col_right:
    st.subheader("Analytics & Risk")

    # Greeks aggreges
    greeks = pf.aggregate_greeks(state)
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Delta $", f"{greeks['delta_usd']:,.0f}")
    g2.metric("Gamma", f"{greeks['gamma_usd']:.1f}")
    g3.metric("Vega", f"{greeks['vega_usd']:,.0f}")
    g4.metric("Theta/j", f"{greeks['theta_usd']:,.0f}")

    # VaR/CVaR portfolio
    var, cvar = pf.compute_var_cvar(level=0.95)
    v1, v2 = st.columns(2)
    v1.metric("VaR 95% (monthly)", f"${var:,.0f}")
    v2.metric("CVaR 95% (monthly)", f"${cvar:,.0f}")

    # Analytics instrument
    selected_analytics = st.selectbox(
        "Analyse instrument",
        [t for t, c, _, _ in INSTRUMENT_UNIVERSE if c != "OPTION_FX"],
        key="ana_ticker"
    )
    a = sim.analytics_for(selected_analytics)
    if a:
        st.markdown("**Vol estimes**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Realisee 252j", f"{a['vol_realized_252d']*100:.1f}%")
        c2.metric("EWMA 0.94", f"{a['vol_ewma']*100:.1f}%")
        c3.metric("GARCH(1,1)", f"{a['garch']['sigma_t1']*np.sqrt(252)*100:.1f}%")

        # GARCH forecast path
        st.markdown("**Forecast GARCH 21j (vol annualisee)**")
        fig, ax = plt.subplots(figsize=(5, 1.8))
        ax.plot(a["garch"]["sigma_path"] * np.sqrt(252) * 100, color="#a4161a", lw=1.5)
        ax.axhline(a["garch"]["sigma_long_run"] * np.sqrt(252) * 100,
                    color="#1d3557", ls="--", alpha=0.6, label="Long run")
        ax.set_xlabel("jours"); ax.set_ylabel("vol (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(labelsize=7)
        st.pyplot(fig, use_container_width=True)

        # SABR surface si FX EM
        if "sabr_surface" in a:
            st.markdown(f"**SABR vol surface — {selected_analytics} (T=3M)**")
            ss = a["sabr_surface"]
            fig, ax = plt.subplots(figsize=(5, 1.8))
            ax.plot(ss["moneyness"], ss["vols"] * 100, "o-", color="#06a77d", lw=1.4, ms=4)
            ax.axvline(1.0, color="black", ls="--", alpha=0.4)
            ax.set_xlabel("Moneyness K/F"); ax.set_ylabel("Vol IV (%)")
            ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
            st.pyplot(fig, use_container_width=True)

    # Nelson-Siegel yield curve (synthetique)
    st.markdown("**Courbe NS US (synthetique)**")
    taus = np.array([0.25, 0.5, 1, 2, 5, 7, 10, 30])
    yields = nelson_siegel_curve(taus, beta0=0.045, beta1=-0.015,
                                   beta2=-0.005, lambda_=0.5) * 100
    fig, ax = plt.subplots(figsize=(5, 1.6))
    ax.plot(taus, yields, "o-", color="#1d3557", lw=1.4, ms=4)
    ax.set_xlabel("Maturite (annees)"); ax.set_ylabel("Taux (%)")
    ax.set_xscale("log")
    ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
    st.pyplot(fig, use_container_width=True)

# ============================================================
# BOTTOM SECTION : P&L HISTORY + POSITIONS + TRANSACTIONS
# ============================================================
st.markdown("---")
b_left, b_right = st.columns([1.4, 1])

with b_left:
    st.subheader("P&L history")
    if len(pf.pnl_history) > 1:
        df_pnl = pd.DataFrame(pf.pnl_history)
        fig, ax = plt.subplots(figsize=(10, 3.0))
        ax.plot(df_pnl["date"], df_pnl["total_value"], color="#1d3557", lw=2,
                 label="Total value")
        ax.axhline(pf.initial_cash, color="#a4161a", ls="--", alpha=0.7, label="Initial")
        ax.fill_between(df_pnl["date"], pf.initial_cash, df_pnl["total_value"],
                          where=df_pnl["total_value"] >= pf.initial_cash,
                          color="#06a77d", alpha=0.2)
        ax.fill_between(df_pnl["date"], pf.initial_cash, df_pnl["total_value"],
                          where=df_pnl["total_value"] < pf.initial_cash,
                          color="#a4161a", alpha=0.2)
        ax.set_ylabel("USD"); ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
        st.pyplot(fig, use_container_width=True)

with b_right:
    st.subheader("Positions actuelles")
    if pf.positions:
        rows = []
        for pos in pf.positions:
            cur_p = state["spot"].get(pos.instrument, pos.entry_price)
            unrealized = pos.quantity * (cur_p - pos.entry_price)
            rows.append({
                "Instrument": pos.instrument,
                "Qty": f"{pos.quantity:+.0f}",
                "Entry": f"{pos.entry_price:.4f}",
                "Now": f"{cur_p:.4f}",
                "P&L": f"${unrealized:+,.0f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                      hide_index=True, height=200)
    else:
        st.info("Aucune position ouverte.")

st.subheader("Transactions")
if pf.transaction_history:
    df_tx = pd.DataFrame(pf.transaction_history[-15:])
    st.dataframe(df_tx, use_container_width=True, hide_index=True, height=200)
else:
    st.caption("Aucune transaction.")
