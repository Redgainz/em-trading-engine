"""
03_calibrate_params.py
======================
Parametrisation du modele en walk-forward :
  1. split chronologique des seances : train (premiere moitie) / test (seconde moitie)
  2. recherche aleatoire de N configurations autour de v4 sur le train
  3. selection par equivalent certain (moyenne - ecart-type du PnL flat quotidien)
  4. evaluation hors echantillon sur le test, comparaison a v4
Sorties : data/calibration_train.csv, data/calibration_test.csv, data/calibrated_params.json

Usage : python scripts/03_calibrate_params.py --n-candidates 40
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amm import calibration, data  # noqa: E402
from amm.runner import run_grid, trading_days  # noqa: E402

TOP_K = 5


def split_days(tickers: list[str]) -> tuple[dict, dict]:
    train, test = {}, {}
    for tk in tickers:
        days = trading_days(tk)
        cut = len(days) // 2
        train[tk], test[tk] = days[:cut], days[cut:]
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-candidates", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    tickers = data.load_market_meta()["tickers"]
    train, test = split_days(tickers)
    first = tickers[0]
    print(f"Train {train[first][0]} -> {train[first][-1]} | Test {test[first][0]} -> {test[first][-1]}")

    grid = calibration.candidate_grid(args.n_candidates, args.seed)
    t0 = time.time()
    train_rows = run_grid(tickers, grid, days=train)
    train_scores = calibration.score(train_rows)
    print(f"\nTrain : {len(grid)} configurations en {time.time() - t0:.0f}s")
    print(train_scores.head(10).round(1).to_string())

    finalists = list(train_scores.index[:TOP_K])
    if calibration.BASE_VERSION not in finalists:
        finalists.append(calibration.BASE_VERSION)
    test_rows = run_grid(tickers, {k: grid[k] for k in finalists}, days=test)
    test_scores = calibration.score(test_rows)
    print("\nTest (hors echantillon) :")
    print(test_scores.round(1).to_string())

    best = train_scores.index[0]
    train_rows.to_csv(os.path.join(data.DATA_DIR, "calibration_train.csv"), index=False)
    test_rows.to_csv(os.path.join(data.DATA_DIR, "calibration_test.csv"), index=False)
    meta = {"selected": best, "train_days": {k: [str(d) for d in v] for k, v in train.items()},
            "test_days": {k: [str(d) for d in v] for k, v in test.items()},
            "train_ce": float(train_scores.loc[best, "ce"]),
            "test_ce": float(test_scores.loc[best, "ce"]),
            "test_ce_base": float(test_scores.loc[calibration.BASE_VERSION, "ce"])}
    calibration.save_calibrated(grid[best], meta)
    print(f"\nRetenu : {best} -> {calibration.CALIBRATED_PATH}")
    print(pd.Series(calibration.config_to_dict(grid[best])["strategy"]).to_string())
    print(pd.Series(calibration.config_to_dict(grid[best])["hedge"]).to_string())


if __name__ == "__main__":
    main()
