"""
04_backtest_iterations.py
=========================
Backtest de l'iteration du modele v0 -> v5 sur toutes les seances et tous les
sous-jacents du cache (common random numbers : meme marche pour chaque version).

Sorties :
  - data/results_days.csv     une ligne par (ticker, seance, version) avec tous les KPI
  - data/results_summary.csv  agregat par version (desk = somme des sous-jacents)
  - data/results_oos.csv      meme agregat restreint aux seances de test (hors echantillon)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amm import calibration, data  # noqa: E402
from amm.conventions import BUSINESS_DAYS_YEAR  # noqa: E402
from amm.kpi import max_drawdown  # noqa: E402
from amm.runner import run_grid  # noqa: E402

SUM_COLS = ["pnl_mtm", "pnl_flat", "edge", "fees", "hedge_cost", "liquidation_cost", "fills", "contracts",
            "attr_delta", "attr_gamma", "attr_vega", "attr_theta", "attr_residual", "attr_timing",
            "mk_1s", "mk_10s", "mk_60s", "mk_300s", "mkh_60s"]


def desk_summary(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for config, grp in rows.groupby("config", sort=False):
        daily = grp.groupby("day")[["pnl_mtm", "pnl_flat"]].sum()
        pnl = daily["pnl_mtm"].to_numpy()
        totals = grp[SUM_COLS].sum()
        std = pnl.std(ddof=1)
        out.append({
            "version": config, "days": len(daily), "pnl_mtm": totals["pnl_mtm"], "pnl_flat": totals["pnl_flat"],
            "edge": totals["edge"], "retention": totals["pnl_mtm"] / totals["edge"],
            "mean_daily": pnl.mean(), "std_daily": std,
            "sharpe_ann": pnl.mean() / std * np.sqrt(BUSINESS_DAYS_YEAR) if std > 0 else np.nan,
            "hit_ratio": float((pnl > 0).mean()), "max_dd": max_drawdown(np.cumsum(pnl)),
            "fills": totals["fills"], "contracts": totals["contracts"],
            "edge_per_contract": totals["edge"] / totals["contracts"],
            "mk60_per_contract": totals["mk_60s"] / totals["contracts"],
            "mkh60_per_contract": totals["mkh_60s"] / totals["contracts"],
            "uptime": grp["uptime"].mean(), "avg_abs_vega": grp["avg_abs_vega"].mean(),
            **{c: totals[c] for c in ["fees", "hedge_cost", "attr_delta", "attr_gamma", "attr_vega",
                                      "attr_theta", "attr_residual", "attr_timing"]},
        })
    return pd.DataFrame(out).set_index("version")


def test_days_mask(rows: pd.DataFrame) -> pd.Series:
    cal = os.path.join(data.DATA_DIR, "calibrated_params.json")
    if not os.path.exists(cal):
        return pd.Series(False, index=rows.index)
    import json
    with open(cal) as fh:
        test_days = json.load(fh)["meta"]["test_days"]
    return rows.apply(lambda r: str(r["day"]) in test_days.get(r["ticker"], []), axis=1)


def main() -> None:
    tickers = data.load_market_meta()["tickers"]
    versions = calibration.all_versions()
    print(f"Versions : {list(versions)} | sous-jacents : {tickers}")
    t0 = time.time()
    rows = run_grid(tickers, versions)
    print(f"{len(rows)} seances x versions en {time.time() - t0:.0f}s")
    rows.to_csv(os.path.join(data.DATA_DIR, "results_days.csv"), index=False)

    summary = desk_summary(rows)
    summary.to_csv(os.path.join(data.DATA_DIR, "results_summary.csv"))
    cols = ["pnl_mtm", "pnl_flat", "edge", "retention", "sharpe_ann", "hit_ratio", "max_dd",
            "fills", "edge_per_contract", "mk60_per_contract", "attr_delta", "hedge_cost"]
    pd.set_option("display.width", 220)
    print("\n=== Desk (toutes seances) ===")
    print(summary[cols].round(2).to_string())

    mask = test_days_mask(rows)
    if mask.any():
        oos = desk_summary(rows[mask])
        oos.to_csv(os.path.join(data.DATA_DIR, "results_oos.csv"))
        print("\n=== Hors echantillon (seances de test de la calibration) ===")
        print(oos[cols].round(2).to_string())


if __name__ == "__main__":
    main()
