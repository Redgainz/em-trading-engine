"""
setup_day.py
============
Assemble, a partir du cache local, tout ce qu'il faut pour simuler une seance :
nappe calibree (snapshot), modele d'activite, book d'options du jour et DayMarket.
"""
from __future__ import annotations

import datetime as dt
import zlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from . import data
from .conventions import continuous_rate
from .market import DayMarket, build_day_market
from .params import FlowParams
from .surface import VolSurface, calibrate_surface
from .universe import ActivityModel, build_option_book, fit_activity_model


@dataclass(frozen=True)
class TickerContext:
    ticker: str
    surface: VolSurface
    iv_points: pd.DataFrame
    activity: ActivityModel
    chain: pd.DataFrame
    intraday: pd.DataFrame
    daily: pd.DataFrame
    days: tuple[dt.date, ...]
    median_volume: float


@lru_cache(maxsize=8)
def load_ticker_context(ticker: str) -> TickerContext:
    meta = data.load_market_meta()
    rate = continuous_rate(meta["selic"])
    chain = data.load_chain(ticker)
    intraday = data.load_intraday(ticker)
    daily = data.load_daily(ticker)
    valuation = chain["last_date"].max().date()
    spot = float(intraday[intraday.index.date == valuation]["Close"].iloc[-1])
    rv20 = data.realized_vol_before(daily, valuation + dt.timedelta(days=1))
    surface, points = calibrate_surface(ticker, chain, spot, rate, valuation, rv20)
    activity = fit_activity_model(chain, surface)
    days = tuple(data.trading_days(intraday))
    volumes = intraday.groupby(intraday.index.date)["Volume"].sum()
    return TickerContext(ticker=ticker, surface=surface, iv_points=points, activity=activity,
                         chain=chain, intraday=intraday, daily=daily, days=days,
                         median_volume=float(volumes.median()))


def day_seed(ticker: str, day: dt.date, salt: int = 0) -> int:
    """Graine deterministe par (ticker, seance) : stable entre processus."""
    return zlib.crc32(f"{ticker}|{day.isoformat()}|{salt}".encode())


def level_scale_for(ctx: TickerContext, day: dt.date) -> float:
    """Niveau de vol du jour = vol snapshot * RV20(jour) / RV20(snapshot) (sans look-ahead)."""
    rv_day = data.realized_vol_before(ctx.daily, day)
    return float(np.clip(rv_day / ctx.surface.rv20_snapshot, 0.6, 1.6))


def build_market(ticker: str, day: dt.date, flow: FlowParams = FlowParams(), salt: int = 0) -> DayMarket:
    ctx = load_ticker_context(ticker)
    bars_day = ctx.intraday[ctx.intraday.index.date == day]
    if bars_day.empty:
        raise ValueError(f"Pas de barres 1m pour {ticker} le {day}")
    activity_ratio = float(np.clip(bars_day["Volume"].sum() / ctx.median_volume, 0.3, 3.0))
    book = build_option_book(ctx.chain, ctx.surface, ctx.activity, day,
                             float(bars_day["Open"].iloc[0]), activity_ratio)
    return build_day_market(ticker, day, bars_day, ctx.surface, book, flow,
                            level_scale_for(ctx, day), day_seed(ticker, day, salt))
