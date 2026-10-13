"""
runner.py
=========
Execution multi-seances / multi-configurations, parallelisee par seance.

Chaque tache (ticker, seance) construit le marche UNE fois puis fait tourner toutes
les configurations dessus : memes trajectoires, memes ordres candidats
(common random numbers) -> differences de PnL entre versions = effet du modele seul.
"""
from __future__ import annotations

import datetime as dt
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from .engine import DayResult, run_day
from .kpi import day_summary
from .market import DayMarket
from .params import Config, FlowParams
from .setup_day import build_market, load_ticker_context

DEFAULT_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))


def _day_task(args: tuple) -> list[dict]:
    ticker, day, configs, flow = args
    market = build_market(ticker, day, flow)
    rows = []
    for label, cfg in configs.items():
        row = day_summary(run_day(market, cfg, flow), market)
        rows.append({**row, "config": label})
    return rows


def trading_days(ticker: str) -> list[dt.date]:
    return list(load_ticker_context(ticker).days)


def run_grid(tickers: list[str], configs: dict[str, Config], flow: FlowParams = FlowParams(),
             days: dict[str, list[dt.date]] | None = None, workers: int = DEFAULT_WORKERS) -> pd.DataFrame:
    """Une ligne par (ticker, seance, config)."""
    tasks = [(tk, d, configs, flow) for tk in tickers for d in (days or {}).get(tk, trading_days(tk))]
    if workers <= 1:
        results = map(_day_task, tasks)
        return pd.DataFrame([r for rows in results for r in rows])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = pool.map(_day_task, tasks, chunksize=1)
        return pd.DataFrame([r for rows in results for r in rows])


def run_detailed(ticker: str, day: dt.date, cfg: Config,
                 flow: FlowParams = FlowParams()) -> tuple[DayResult, DayMarket]:
    """Une seance avec tous les logs (fills, echantillons) pour les figures / l'app."""
    market = build_market(ticker, day, flow)
    return run_day(market, cfg, flow), market


def book_daily_pnl(rows: pd.DataFrame, pnl_col: str = "pnl_mtm") -> pd.DataFrame:
    """PnL quotidien du desk (somme des sous-jacents) par configuration."""
    return rows.groupby(["config", "day"], sort=False)[[pnl_col, "edge", "pnl_flat", "fills", "contracts"]].sum()
