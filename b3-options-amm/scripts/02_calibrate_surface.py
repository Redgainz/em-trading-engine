"""
02_calibrate_surface.py
=======================
Calibre la nappe SVI sur la chaine B3 reelle du snapshot et ecrit :
  - data/{TICKER}_svi.csv        parametres SVI par echeance + qualite du fit
  - data/{TICKER}_iv_points.csv  points de vol implicite utilises
Controle : densite papillon g(k) >= 0 (pas d'arbitrage statique) sur |k| <= 0.35.
"""
from __future__ import annotations

import os
import sys
from dataclasses import asdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amm import data  # noqa: E402
from amm.setup_day import load_ticker_context  # noqa: E402
from amm.svi import butterfly_density  # noqa: E402

K_GRID = np.linspace(-0.35, 0.35, 141)


def main() -> None:
    for ticker in data.load_market_meta()["tickers"]:
        ctx = load_ticker_context(ticker)
        s = ctx.surface
        print(f"\n== {ticker} | snapshot {s.valuation_date} | spot {s.spot:.2f} | "
              f"r = {s.rate:.4f} | RV20 = {s.rv20_snapshot:.1%} ==")
        rows = []
        for sm in s.smiles:
            g_min = float(butterfly_density(sm.svi, K_GRID).min())
            rows.append({"expiry": sm.expiry, "T": sm.T_snapshot, "atm_vol": sm.atm_vol,
                         "n_points": sm.n_points, "rmse_vol": sm.rmse_vol,
                         "butterfly_g_min": g_min, **asdict(sm.svi)})
            status = "OK" if g_min >= 0 else "ARBITRAGE"
            print(f"  {sm.expiry}  T={sm.T_snapshot:.4f}  ATM={sm.atm_vol:.1%}  "
                  f"n={sm.n_points:3d}  RMSE={sm.rmse_vol * 100:.2f} pt vol  g_min={g_min:.3f} {status}")
        pd.DataFrame(rows).to_csv(os.path.join(data.DATA_DIR, f"{ticker}_svi.csv"), index=False)
        cols = ["ticker", "type", "strike", "expiry", "last", "n_trades", "T", "k", "iv"]
        ctx.iv_points[cols].to_csv(os.path.join(data.DATA_DIR, f"{ticker}_iv_points.csv"), index=False)


if __name__ == "__main__":
    main()
