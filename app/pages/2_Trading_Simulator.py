"""
Trading Simulator — avec evenements mensuels client (v3)
=========================================================
Layout :
  - Sidebar : controles globaux + bouton Avancer 1 mois
  - Header : metrics portfolio
  - Section EVENEMENTS DU MOIS : RFQ a accepter/refuser + news
  - 3 colonnes : Marche / Position builder / Analytics
  - Bottom : P&L history, positions, transactions
"""
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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
from lib.events import (generate_monthly_events, execute_rfq_acceptance,
                          MarketEvent)

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
div.stMetric { background: #f7f8fa; padding: 0.5rem; border-radius: 6px; }
.event-card {
    background: #fdf6ef; border-left: 4px solid #fb8500;
    padding: 0.8rem 1rem; margin-bottom: 0.6rem; border-radius: 4px;
}
.event-card-news {
    background: #f0f3f7; border-left: 4px solid #3a86ff;
    padding: 0.8rem 1rem; margin-bottom: 0.6rem; border-radius: 4px;
}
.event-card-option {
    background: #f1f8f4; border-left: 4px solid #06a77d;
    padding: 0.8rem 1rem; margin-bottom: 0.6rem; border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

REGIME_NAMES = {0: "CALME", 1: "NORMAL", 2: "CRISE"}
REGIME_COLORS = {0: "#06a77d", 1: "#fb8500", 2: "#a4161a"}

# ============================================================
# INIT SESSION STATE
# ============================================================
if "simulator" not in st.session_state:
    st.session_state.simulator = MarketSimulator(
        data_dir=DATA_DIR, start_date="2024-01-01",
        seed=int(np.random.randint(0, 1e6))
    )
    st.session_state.portfolio = Portfolio(initial_cash=1_000_000.0)
    st.session_state.month_count = 0
    st.session_state.current_state = st.session_state.simulator.get_market_state()
    st.session_state.portfolio.snapshot(st.session_state.current_state,
                                          st.session_state.current_state["date"])
    # Generer les evenements du mois initial
    st.session_state.event_rng = np.random.default_rng(
        int(np.random.randint(0, 1e6))
    )
    st.session_state.pending_events = generate_monthly_events(
        st.session_state.event_rng,
        st.session_state.current_state,
        st.session_state.current_state["regime_idx"]
    )
    st.session_state.events_processed = []   # historique d'events

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
    st.session_state.event_rng = np.random.default_rng(seed)
    st.session_state.pending_events = generate_monthly_events(
        st.session_state.event_rng, st.session_state.current_state,
        st.session_state.current_state["regime_idx"]
    )
    st.session_state.events_processed = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Mois écoulés** : {st.session_state.month_count}")
st.sidebar.markdown(f"**Date** : {state['date'].date()}")
st.sidebar.markdown(f"**Régime** : {REGIME_NAMES[state['regime_idx']]}")
st.sidebar.markdown(f"**Événements en attente** : {len(st.session_state.pending_events)}")

if st.sidebar.button("Avancer 1 mois", type="primary"):
    st.session_state.current_state = sim.step_one_month()
    pf.age_positions(1.0 / 12.0,
                       st.session_state.current_state["spot"],
                       st.session_state.current_state["date"])
    pf.snapshot(st.session_state.current_state,
                  st.session_state.current_state["date"])
    # Generer les nouveaux evenements
    st.session_state.pending_events = generate_monthly_events(
        st.session_state.event_rng,
        st.session_state.current_state,
        st.session_state.current_state["regime_idx"]
    )
    st.session_state.month_count += 1
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Coût de transaction** : 5 bps")
st.sidebar.markdown("**Capital initial** : $1,000,000")

# ============================================================
# HEADER : METRICS
# ============================================================
mtm = pf.mark_to_market(state, state["date"])
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Date", state["date"].strftime("%Y-%m"))
col2.metric("Cash", f"${pf.cash:,.0f}")
col3.metric("MTM positions", f"${mtm['mtm_positions']:,.0f}")
col4.metric("Total value", f"${mtm['total_value']:,.0f}",
             f"${mtm['pnl_total']:+,.0f}")
col5.metric("Positions", f"{len(pf.positions)}")
col6.metric("Régime", REGIME_NAMES[state["regime_idx"]])

# ============================================================
# SECTION EVENEMENTS DU MOIS
# ============================================================
st.markdown("---")
n_evts = len(st.session_state.pending_events)
n_actionable = sum(1 for e in st.session_state.pending_events if e.is_actionable)

st.header(f"📅  Événements du mois")
if n_evts > 0:
    st.caption(f"Régime actuel : **{REGIME_NAMES[state['regime_idx']]}**  ·  "
                f"{n_actionable} RFQ à traiter, {n_evts - n_actionable} news à acquitter")
else:
    st.caption(f"Régime actuel : **{REGIME_NAMES[state['regime_idx']]}**  ·  aucun événement.")

if not st.session_state.pending_events:
    st.info("✓ Tous les événements du mois ont été traités. Clique **Avancer 1 mois** pour passer au suivant.")
else:
    for idx, evt in enumerate(list(st.session_state.pending_events)):
        # Container natif Streamlit avec bordure
        with st.container(border=True):
            if evt.kind == "NEWS":
                cols = st.columns([8, 2])
                with cols[0]:
                    st.markdown(f"### 📰  News flash")
                    st.markdown(f"**{evt.description.replace('📰  ', '')}**")
                    if evt.news_currency:
                        impact = "haussier" if evt.news_drift_impact > 0 else "baissier"
                        st.caption(f"Impact attendu : {impact} sur **{evt.news_currency}** "
                                    f"(drift {evt.news_drift_impact*100:+.1f}% / mois)")
                with cols[1]:
                    st.write("")  # spacer
                    if st.button("Acquitter", key=f"news_{evt.event_id}_{st.session_state.month_count}",
                                  use_container_width=True):
                        st.session_state.pending_events.remove(evt)
                        st.session_state.events_processed.append(("NEWS_ACK", evt, state["date"]))
                        st.rerun()

            elif evt.kind == "RFQ_SPOT":
                cols = st.columns([6, 4])
                with cols[0]:
                    st.markdown(f"### 💱  RFQ Spot — {evt.client_name}")
                    st.markdown(f"**{evt.description}**")

                    info_cols = st.columns(4)
                    info_cols[0].metric("Sous-jacent", evt.instrument)
                    info_cols[1].metric("Côté client", evt.client_side)
                    info_cols[2].metric("Notional", f"${evt.notional_usd/1e6:.1f}M")
                    info_cols[3].metric("Spread", f"{evt.spread_bps:.1f} bps")

                with cols[1]:
                    st.markdown("**Si tu acceptes :**")
                    st.markdown(f"→ Tu **{evt.mm_side}** {evt.quantity:,.0f} {evt.instrument.split('/')[0]} "
                                  f"@ **{evt.client_price:.5f}**")
                    st.markdown(f"→ Mid de marché : {evt.market_mid:.5f}")
                    st.success(f"💰 Gain de spread : **+${evt.expected_pnl_spread:,.0f}**")

                    btn_cols = st.columns(2)
                    if btn_cols[0].button("✓ Accepter",
                                            key=f"acc_{evt.event_id}_{st.session_state.month_count}",
                                            type="primary", use_container_width=True):
                        try:
                            execute_rfq_acceptance(pf, evt, state["date"])
                            st.session_state.pending_events.remove(evt)
                            st.session_state.events_processed.append(("ACCEPTED", evt, state["date"]))
                            st.rerun()
                        except Exception as e:
                            st.error(f"Erreur : {e}")
                    if btn_cols[1].button("✗ Refuser",
                                            key=f"rej_{evt.event_id}_{st.session_state.month_count}",
                                            use_container_width=True):
                        st.session_state.pending_events.remove(evt)
                        st.session_state.events_processed.append(("REJECTED", evt, state["date"]))
                        st.rerun()

            elif evt.kind == "RFQ_OPTION":
                cols = st.columns([6, 4])
                with cols[0]:
                    st.markdown(f"### 📊  RFQ Option — {evt.client_name}")
                    st.markdown(f"**{evt.description}**")

                    info_cols = st.columns(4)
                    info_cols[0].metric("Option", evt.instrument.replace("_ATM", ""))
                    info_cols[1].metric("Notional", f"${evt.notional_usd/1e6:.1f}M")
                    info_cols[2].metric("IV", f"{evt.implied_vol*100:.1f}%")
                    info_cols[3].metric("Markup", f"{evt.spread_bps:.1f} bps")

                with cols[1]:
                    st.markdown("**Si tu acceptes :**")
                    st.markdown(f"→ Tu **VEND l'option** ({evt.quantity:,.0f} unités)")
                    st.markdown(f"→ Prime encaissée : **{evt.client_price:.5f}** par unité")
                    st.markdown(f"→ Mid théorique : {evt.market_mid:.5f}")
                    st.success(f"💰 Markup encaissé : **+${evt.expected_pnl_spread:,.0f}**")
                    st.warning("⚠️ Tu prends un risque vega + gamma sur la durée de vie de l'option")

                    btn_cols = st.columns(2)
                    if btn_cols[0].button("✓ Accepter",
                                            key=f"acc_{evt.event_id}_{st.session_state.month_count}",
                                            type="primary", use_container_width=True):
                        try:
                            execute_rfq_acceptance(pf, evt, state["date"])
                            st.session_state.pending_events.remove(evt)
                            st.session_state.events_processed.append(("ACCEPTED", evt, state["date"]))
                            st.rerun()
                        except Exception as e:
                            st.error(f"Erreur : {e}")
                    if btn_cols[1].button("✗ Refuser",
                                            key=f"rej_{evt.event_id}_{st.session_state.month_count}",
                                            use_container_width=True):
                        st.session_state.pending_events.remove(evt)
                        st.session_state.events_processed.append(("REJECTED", evt, state["date"]))
                        st.rerun()

# ============================================================
# MAIN LAYOUT : 3 COLUMNS
# ============================================================
st.markdown("---")
col_left, col_center, col_right = st.columns([1.1, 1.2, 1.2])

with col_left:
    st.subheader("Marché")
    regime_idx = state["regime_idx"]
    st.markdown(
        f"<div style='background:{REGIME_COLORS[regime_idx]}; padding:0.4rem; "
        f"border-radius:5px; color:white; font-weight:600; text-align:center;'>"
        f"RÉGIME : {REGIME_NAMES[regime_idx]}</div>", unsafe_allow_html=True
    )
    regime_probs = state["regime_probs"]
    st.caption(f"Probas régime (next month) : CALME={regime_probs[0]:.0%} "
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

with col_center:
    st.subheader("Position builder")
    st.markdown("Sélectionne un instrument, une action et une quantité.")

    with st.form("trade_form"):
        all_tickers = [t for t, c, _, _ in INSTRUMENT_UNIVERSE]
        selected = st.selectbox("Instrument", all_tickers,
                                 format_func=lambda t: f"{t}  ({next(c for tt,c,_,_ in INSTRUMENT_UNIVERSE if tt==t)})")
        asset_class = next(c for tt, c, _, _ in INSTRUMENT_UNIVERSE if tt == selected)
        cur_price = state["spot"][selected]
        st.caption(f"Prix actuel : **{cur_price:.4f}**" if asset_class == "FX"
                    else f"Prix actuel : **{cur_price:.2f}**")

        col_a, col_b = st.columns(2)
        with col_a:
            action = st.radio("Action", ["BUY", "SELL"], horizontal=True)
        with col_b:
            qty = st.number_input("Quantité", min_value=1.0, step=100.0, value=1000.0)

        notional = qty * cur_price
        st.caption(f"Notional : **${notional:,.0f}**  ({notional/pf.cash*100:.1f}% du cash)")

        submit = st.form_submit_button("Exécuter", type="primary")
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
                    st.success(f"Acheté {qty:.0f} de {selected} à {cur_price:.4f}")
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
                    st.success(f"Vendu {qty:.0f} de {selected} à {cur_price:.4f}")
                st.rerun()
            except ValueError as e:
                st.error(str(e))

    if st.button("Fermer toutes les positions"):
        pf.close_all({pos.instrument: state["spot"][pos.instrument]
                       for pos in pf.positions if pos.instrument in state["spot"]},
                       state["date"])
        st.rerun()

with col_right:
    st.subheader("Analytics & Risk")
    greeks = pf.aggregate_greeks(state)
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Delta $", f"{greeks['delta_usd']:,.0f}")
    g2.metric("Gamma", f"{greeks['gamma_usd']:.1f}")
    g3.metric("Vega", f"{greeks['vega_usd']:,.0f}")
    g4.metric("Theta/j", f"{greeks['theta_usd']:,.0f}")

    var, cvar = pf.compute_var_cvar(level=0.95)
    v1, v2 = st.columns(2)
    v1.metric("VaR 95% (mensuel)", f"${var:,.0f}")
    v2.metric("CVaR 95% (mensuel)", f"${cvar:,.0f}")

    selected_analytics = st.selectbox(
        "Analyse instrument",
        [t for t, c, _, _ in INSTRUMENT_UNIVERSE if c != "OPTION_FX"],
        key="ana_ticker"
    )
    a = sim.analytics_for(selected_analytics)
    if a:
        st.markdown("**Vols estimées**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Réalisée 252j", f"{a['vol_realized_252d']*100:.1f}%")
        c2.metric("EWMA 0.94", f"{a['vol_ewma']*100:.1f}%")
        c3.metric("GARCH(1,1)", f"{a['garch']['sigma_t1']*np.sqrt(252)*100:.1f}%")

        st.markdown("**Forecast GARCH 21j (vol annualisée)**")
        fig, ax = plt.subplots(figsize=(5, 1.8))
        ax.plot(a["garch"]["sigma_path"] * np.sqrt(252) * 100,
                 color="#a4161a", lw=1.5)
        ax.axhline(a["garch"]["sigma_long_run"] * np.sqrt(252) * 100,
                    color="#1d3557", ls="--", alpha=0.6, label="Long run")
        ax.set_xlabel("jours"); ax.set_ylabel("vol (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(labelsize=7)
        st.pyplot(fig, use_container_width=True)

        if "sabr_surface" in a:
            st.markdown(f"**SABR vol surface — {selected_analytics} (T=3M)**")
            ss = a["sabr_surface"]
            fig, ax = plt.subplots(figsize=(5, 1.8))
            ax.plot(ss["moneyness"], ss["vols"] * 100, "o-",
                     color="#06a77d", lw=1.4, ms=4)
            ax.axvline(1.0, color="black", ls="--", alpha=0.4)
            ax.set_xlabel("Moneyness K/F"); ax.set_ylabel("Vol IV (%)")
            ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
            st.pyplot(fig, use_container_width=True)

    st.markdown("**Courbe NS US (synthétique)**")
    taus = np.array([0.25, 0.5, 1, 2, 5, 7, 10, 30])
    yields = nelson_siegel_curve(taus, beta0=0.045, beta1=-0.015,
                                   beta2=-0.005, lambda_=0.5) * 100
    fig, ax = plt.subplots(figsize=(5, 1.6))
    ax.plot(taus, yields, "o-", color="#1d3557", lw=1.4, ms=4)
    ax.set_xlabel("Maturité (années)"); ax.set_ylabel("Taux (%)")
    ax.set_xscale("log")
    ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
    st.pyplot(fig, use_container_width=True)

# ============================================================
# BOTTOM SECTION
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
        ax.axhline(pf.initial_cash, color="#a4161a", ls="--", alpha=0.7,
                    label="Initial")
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

st.subheader("Transactions récentes")
if pf.transaction_history:
    df_tx = pd.DataFrame(pf.transaction_history[-15:])
    st.dataframe(df_tx, use_container_width=True, hide_index=True, height=200)
else:
    st.caption("Aucune transaction.")

# Stats events processed
if st.session_state.events_processed:
    st.subheader("Historique des décisions sur RFQ")
    rows = []
    for status, evt, date in st.session_state.events_processed[-15:]:
        rows.append({
            "Date": date.strftime("%Y-%m"),
            "Type": evt.kind,
            "Décision": status,
            "Client": evt.client_name,
            "Instrument": evt.instrument or "—",
            "Gain spread potentiel": f"${evt.expected_pnl_spread:,.0f}" if evt.is_actionable else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                  hide_index=True, height=200)
