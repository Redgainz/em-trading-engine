"""
kpi.py
======
KPI d'un desk HFT options, calcules a partir du log d'execution et du marche.

  - Edge capture  : sum we_side * (mid_t - prix) * taille * 100   (vs mid ecran a l'execution)
  - Markouts      : PnL d'un fill a t + h vs le mid ecran, brut et delta-couvert
                    (h = 1 s ... 5 min) -> mesure directe de l'adverse selection
  - Attribution   : PnL d'inventaire decompose en delta / gamma / vega / theta / residu
                    (greeks vrais aux points d'echantillonnage, 10 s)
  - PnL retention : PnL net / edge capture  (part de l'edge theorique qui reste au desk)
  - Qualite de quoting : uptime deux-sens, temps au meilleur prix, spread cote vs ecran
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .conventions import BUSINESS_DAYS_YEAR, CONTRACT_SIZE, SESSION_SECONDS
from .engine import DayResult
from .market import DayMarket
from .pricing import bs_greeks

MARKOUT_HORIZONS_S = (1, 5, 10, 30, 60, 120, 300)


def fill_markouts(res: DayResult, m: DayMarket, horizons=MARKOUT_HORIZONS_S) -> pd.DataFrame:
    """Une ligne par fill, colonnes mk_{h}s (brut) et mkh_{h}s (delta-couvert), en R$."""
    f = res.fills
    if f.empty:
        return f.copy()
    t, i = f["t"].to_numpy(), f["i"].to_numpy()
    side, px, size = f["we_side"].to_numpy(), f["price"].to_numpy(), f["size"].to_numpy()
    mid = m.comp_mid
    out = {"edge": side * (f["mid"].to_numpy() - px) * size * CONTRACT_SIZE}
    for h in horizons:
        tm = np.minimum(t + h, m.n_seconds - 1)
        raw = side * (mid[tm, i] - px) * size * CONTRACT_SIZE
        hedge = side * f["delta"].to_numpy() * (m.S[tm] - f["S"].to_numpy()) * size * CONTRACT_SIZE
        out[f"mk_{h}s"] = raw
        out[f"mkh_{h}s"] = raw - hedge
    return f.assign(**out)


def greek_attribution(res: DayResult, m: DayMarket) -> dict[str, float]:
    """Decomposition du PnL d'inventaire entre echantillons (positions figees sur 10 s)."""
    s = res.samples
    t = s["t"].to_numpy()
    q = res.sample_q
    if len(t) < 2:
        return {k: 0.0 for k in ("inventory", "delta", "gamma", "vega", "theta", "residual")}
    t0, t1 = t[:-1], t[1:]
    S0, S1 = m.S[t0], m.S[t1]
    g = bs_greeks(S0[:, None], m.book.strikes[None, :], m.T[t0], m.rate, m.sigma[t0], m.book.is_call[None, :])
    qc = q[:-1] * CONTRACT_SIZE
    stock = s["stock"].to_numpy()[:-1]
    dS = S1 - S0
    mid = m.comp_mid
    inventory = np.sum(qc * (mid[t1] - mid[t0]), axis=1) + stock * dS
    delta = (np.sum(qc * g["delta"], axis=1) + stock) * dS
    gamma = 0.5 * np.sum(qc * g["gamma"], axis=1) * dS ** 2
    vega = np.sum(qc * g["vega"] * (m.sigma[t1] - m.sigma[t0]), axis=1)
    dt_years = (t1 - t0) / SESSION_SECONDS / BUSINESS_DAYS_YEAR
    theta = np.sum(qc * g["theta"], axis=1) * dt_years
    parts = {"inventory": inventory.sum(), "delta": delta.sum(), "gamma": gamma.sum(),
             "vega": vega.sum(), "theta": theta.sum()}
    parts["residual"] = parts["inventory"] - (parts["delta"] + parts["gamma"] + parts["vega"] + parts["theta"])
    return {k: float(v) for k, v in parts.items()}


def max_drawdown(pnl: np.ndarray) -> float:
    if len(pnl) == 0:
        return 0.0
    return float(np.max(np.maximum.accumulate(pnl) - pnl))


def day_summary(res: DayResult, m: DayMarket) -> dict:
    tot, s = res.totals, res.samples
    mk = fill_markouts(res, m)
    attr = greek_attribution(res, m)
    edge = float(mk["edge"].sum()) if not mk.empty else 0.0
    contracts = float(res.fills["size"].sum()) if not res.fills.empty else 0.0
    mtm = tot["pnl_before_liquidation"]
    row = {
        "ticker": res.ticker, "day": res.day, "strategy": res.strategy,
        "pnl_mtm": mtm, "pnl_flat": tot["net_pnl"], "edge": edge,
        "retention": mtm / edge if edge > 0 else np.nan,
        "fills": len(res.fills), "contracts": contracts,
        "informed_share": float(res.fills["informed"].mean()) if len(res.fills) else 0.0,
        "fees": tot["option_fees"] + tot["stock_fees"], "hedge_cost": tot["hedge_spread_cost"],
        "liquidation_cost": tot["liquidation_cost"],
        "hedges": len(res.hedges), "shares_hedged": float(res.hedges["shares"].abs().sum()) if len(res.hedges) else 0.0,
        "uptime": float(s["uptime"].mean()), "at_best": float(s["at_best"].mean()),
        "spread_ticks": float(s["spread_ticks"].mean()), "screen_spread_ticks": float(s["comp_spread_ticks"].mean()),
        "avg_abs_vega": float(s["vega_volpt"].abs().mean()), "max_abs_vega": float(s["vega_volpt"].abs().max()),
        "avg_abs_delta": float(s["delta_shares"].abs().mean()), "avg_gross_contracts": float(s["gross_contracts"].mean()),
        "intraday_max_dd": max_drawdown(s["pnl_mid"].to_numpy()), "killed": bool(tot["killed"]),
    }
    row.update({f"attr_{k}": v for k, v in attr.items()})
    row["attr_timing"] = mtm - (edge + attr["inventory"] - row["fees"] - row["hedge_cost"])
    for h in MARKOUT_HORIZONS_S:
        if mk.empty:
            row[f"mk_{h}s"] = row[f"mkh_{h}s"] = 0.0
        else:
            row[f"mk_{h}s"] = float(mk[f"mk_{h}s"].sum())
            row[f"mkh_{h}s"] = float(mk[f"mkh_{h}s"].sum())
    return row


def aggregate(days: pd.DataFrame, pnl_col: str = "pnl_mtm") -> dict:
    """Statistiques multi-seances d'une strategie."""
    pnl = days[pnl_col].to_numpy()
    std = pnl.std(ddof=1) if len(pnl) > 1 else np.nan
    edge = days["edge"].sum()
    return {
        "days": len(days), "total_pnl": pnl.sum(), "mean_daily_pnl": pnl.mean(),
        "std_daily_pnl": std, "sharpe_ann": pnl.mean() / std * np.sqrt(BUSINESS_DAYS_YEAR) if std and std > 0 else np.nan,
        "hit_ratio": float((pnl > 0).mean()), "max_dd": max_drawdown(np.cumsum(pnl)),
        "total_edge": edge, "retention": pnl.sum() / edge if edge > 0 else np.nan,
        "total_pnl_flat": days["pnl_flat"].sum(), "fills": int(days["fills"].sum()),
        "contracts": days["contracts"].sum(), "uptime": days["uptime"].mean(),
        "avg_abs_vega": days["avg_abs_vega"].mean(), "mkh_60s_per_contract":
            days["mkh_60s"].sum() / max(days["contracts"].sum(), 1.0),
    }
