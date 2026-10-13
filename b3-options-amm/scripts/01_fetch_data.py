"""
01_fetch_data.py
================
Telecharge les donnees reelles B3 et les ecrit dans b3-options-amm/data/ :

  - {TICKER}_1m.csv     barres 1 minute Yahoo (~20 seances)
  - {TICKER}_daily.csv  2 ans de daily Yahoo (vol realisee)
  - {TICKER}_chain.csv  chaine d'options B3 (opcoes.net.br), echeances mensuelles
  - market_meta.json    Selic (BCB SGS 1178) + date du snapshot

Usage :
    python scripts/01_fetch_data.py                    # PETR4 + VALE3
    python scripts/01_fetch_data.py --tickers BOVA11 ITUB4 --n-monthly 3
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amm import data  # noqa: E402

DEFAULT_TICKERS = ["PETR4", "VALE3"]
MIN_DAYS_TO_EXPIRY = 0


def monthly_expiries(ticker: str, n_monthly: int) -> list[str]:
    listed = data.fetch_expiries(ticker)
    monthly = [e["expiry"] for e in listed if e["monthly"] and e["du"] > MIN_DAYS_TO_EXPIRY]
    return monthly[:n_monthly]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--n-monthly", type=int, default=3,
                        help="nombre d'echeances mensuelles a recuperer")
    args = parser.parse_args()

    selic = data.fetch_selic()
    as_of = dt.date.today().isoformat()
    print(f"Selic (BCB 1178) : {selic:.2%}  |  snapshot {as_of}")

    for ticker in args.tickers:
        print(f"\n== {ticker} ==")
        expiries = monthly_expiries(ticker, args.n_monthly)
        print(f"  echeances mensuelles : {expiries}")
        chain = data.fetch_option_chain(ticker, expiries)
        traded = int((chain["n_trades"] > 0).sum())
        print(f"  chaine : {len(chain)} options, {traded} traitees a la derniere seance")
        intraday = data.fetch_intraday(ticker)
        days = data.trading_days(intraday)
        print(f"  1 minute : {len(intraday)} barres, {len(days)} seances "
              f"({days[0]} -> {days[-1]})")
        daily = data.fetch_daily(ticker)
        print(f"  daily : {len(daily)} seances")
        data.save_snapshot(ticker, intraday, daily, chain, selic, as_of)
    print(f"\nEcrit dans {data.DATA_DIR}")


if __name__ == "__main__":
    main()
