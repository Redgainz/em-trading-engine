"""
market.py
=========
Simulateur de marche B3 a la seconde pour une seance, ancre sur les donnees reelles.

  Spot      : barres 1 minute Yahoo reelles -> pont brownien a la seconde
              (vol intra-minute = Parkinson de la barre), arrondi au demi-tick.
  Vol vraie : sigma_i(t) = (ATM_term(T) * L_jour + beta_i x_t) * smile_i(z_t) + u_i(t)
              x_t = facteur ATM Ornstein-Uhlenbeck, correle aux chocs spot reels (rho < 0)
              u_i = residu de smile par option (inconnu du MM, appris sur l'ecran)
  Ecran     : concurrents cotant fair + bruit AR(1) +/- h_i, arrondis au tick
  Flux      : - bruit : Poisson, intensite calibree sur la chaine reelle, sensible au prix
                (lambda * exp(-k (delta - h)), plafonnee)
              - informe : connait le fair a t + H et ne traite que s'il a de l'edge
Tout l'alea est tire a la construction (graine par seance) : toutes les versions
de strategie voient exactement le meme marche et les memes ordres candidats
(common random numbers -> comparaisons appariees).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .conventions import SESSION_SECONDS, TICK, BUSINESS_DAYS_YEAR, round_to_tick
from .params import FlowParams
from .pricing import bs_greeks, bs_price
from .surface import VolSurface
from .universe import OptionBook

SECONDS_PER_MINUTE = 60
N_MINUTES = SESSION_SECONDS // SECONDS_PER_MINUTE
PARKINSON_DENOM = 2.0 * np.sqrt(np.log(2.0))
MIN_MINUTE_VOL = 1e-5


@dataclass(frozen=True)
class Events:
    t: np.ndarray            # seconde d'arrivee
    i: np.ndarray            # indice option
    side: np.ndarray         # +1 achat agresseur, -1 vente, 0 = informe (decide a l'arrivee)
    size: np.ndarray         # contrats demandes
    u_accept: np.ndarray     # uniforme pour l'amincissement du flux bruit
    u_queue: np.ndarray      # uniforme pour la priorite dans la file


@dataclass(frozen=True)
class DayMarket:
    ticker: str
    day: dt.date
    rate: float
    level_scale: float
    book: OptionBook
    S: np.ndarray            # (n,) mid action
    T: np.ndarray            # (n, N) maturites
    sigma: np.ndarray        # (n, N) vol vraie
    fair: np.ndarray         # (n, N) fair value vraie (par action)
    half: np.ndarray         # (n, N) demi-spread concurrent non arrondi
    comp_bid: np.ndarray     # (n, N)
    comp_ask: np.ndarray     # (n, N)
    events: Events

    @property
    def n_seconds(self) -> int:
        return len(self.S)

    @property
    def comp_mid(self) -> np.ndarray:
        return 0.5 * (self.comp_bid + self.comp_ask)


def session_minutes(bars_day: pd.DataFrame) -> pd.DataFrame:
    """Reindexe les barres sur la grille 10:00 -> 16:54 (415 minutes), forward-fill."""
    day = bars_day.index[0].normalize()
    grid = day + pd.Timedelta(hours=10) + pd.to_timedelta(np.arange(N_MINUTES), unit="min")
    bars = bars_day.reindex(grid)
    bars["Close"] = bars["Close"].ffill().bfill()
    for col in ("Open", "High", "Low"):
        bars[col] = bars[col].fillna(bars["Close"])
    bars["Volume"] = bars["Volume"].fillna(0.0)
    return bars


def spot_path_from_bars(bars: pd.DataFrame, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Pont brownien a la seconde entre closes consecutifs reels.
    Renvoie (S mid arrondi au demi-tick, chocs normalises par seconde).
    """
    closes = bars["Close"].to_numpy()
    log_c = np.log(np.concatenate([[bars["Open"].iloc[0]], closes]))
    sig_min = np.maximum(np.log(bars["High"] / bars["Low"]).to_numpy() / PARKINSON_DENOM, MIN_MINUTE_VOL)
    sig_sec = sig_min / np.sqrt(SECONDS_PER_MINUTE)

    steps = rng.standard_normal((N_MINUTES, SECONDS_PER_MINUTE)) * sig_sec[:, None]
    walk = np.cumsum(steps, axis=1)
    frac = np.arange(1, SECONDS_PER_MINUTE + 1) / SECONDS_PER_MINUTE
    target = np.diff(log_c)[:, None]
    bridge = walk - frac[None, :] * walk[:, -1:] + frac[None, :] * target
    log_path = (log_c[:-1, None] + bridge).ravel()

    rets = np.diff(np.concatenate([[log_c[0]], log_path]))
    shocks = np.clip(rets / np.repeat(sig_sec, SECONDS_PER_MINUTE), -4.0, 4.0)
    S = np.round(np.exp(log_path) / (TICK / 2)) * (TICK / 2)
    return S, shocks


def vol_factor_paths(shocks: np.ndarray, n_opt: int, flow: FlowParams,
                     rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Facteur ATM x_t (OU correle au spot) et residus u_i(t) (AR(1) par option)."""
    n = len(shocks)
    theta = np.log(2.0) / flow.vol_halflife_s
    nu = flow.vol_of_vol_daily / np.sqrt(SESSION_SECONDS)
    rho = flow.spot_vol_corr
    eps = rng.standard_normal(n)
    drive = nu * (rho * shocks + np.sqrt(1.0 - rho ** 2) * eps)
    x = np.empty(n)
    acc = 0.0
    for t in range(n):
        acc = acc * (1.0 - theta) + drive[t]
        x[t] = acc

    u0 = rng.normal(0.0, flow.idio_vol_std, n_opt)
    nu_u = flow.idio_vol_daily / np.sqrt(SESSION_SECONDS)
    u = u0 + np.cumsum(rng.normal(0.0, nu_u, (n, n_opt)), axis=0)
    return x, u


def _smile_ratios(surface: VolSurface, book: OptionBook, S: np.ndarray,
                  T: np.ndarray, atm: np.ndarray) -> np.ndarray:
    forward = S[:, None] * np.exp(surface.rate * T)
    z = np.log(book.strikes[None, :] / forward) / (atm * np.sqrt(T))
    ratios = np.empty_like(z)
    for j, expiry in enumerate(book.expiries):
        ratios[:, j] = surface.smile_for(expiry).vol_ratio(z[:, j])
    return ratios


def _competitor_quotes(fair, vega, flow: FlowParams, rng):
    n, n_opt = fair.shape
    half = np.maximum(flow.comp_vol_halfspread * vega + flow.comp_rel_halfspread * fair,
                      flow.comp_min_halfspread_ticks * TICK)
    a = flow.comp_noise_autocorr
    innov = rng.standard_normal((n, n_opt)) * np.sqrt(1.0 - a * a)
    noise = np.empty((n, n_opt))
    noise[0] = rng.standard_normal(n_opt)
    for t in range(1, n):
        noise[t] = a * noise[t - 1] + innov[t]
    center = fair + flow.comp_noise * half * noise
    bid = np.maximum(round_to_tick(center - half, "down"), 0.0)
    ask = np.maximum(round_to_tick(center + half, "up"), bid + TICK)
    return half, bid, ask


def _draw_events(book: OptionBook, n: int, flow: FlowParams, rng) -> Events:
    lam_noise = book.lambda_per_s * flow.flow_scale * flow.max_flow_mult
    lam_inf = book.lambda_per_s * flow.flow_scale * flow.informed_ratio
    parts = []
    for side, lam, size_mult in ((1, lam_noise, 1.0), (-1, lam_noise, 1.0), (0, lam_inf, flow.informed_size_mult)):
        t_idx, i_idx = np.nonzero(rng.poisson(lam[None, :], (n, book.n)))
        mean = book.mean_size[i_idx] * size_mult
        size = np.minimum(rng.geometric(1.0 / np.maximum(mean, 1.0)), flow.max_trade_contracts)
        parts.append((t_idx, i_idx, np.full(len(t_idx), side), size))
    t, i, side, size = (np.concatenate(col) for col in zip(*parts))
    order = np.argsort(t, kind="stable")
    m = len(t)
    return Events(t=t[order], i=i[order], side=side[order], size=size[order],
                  u_accept=rng.random(m), u_queue=rng.random(m))


def build_day_market(ticker: str, day: dt.date, bars_day: pd.DataFrame, surface: VolSurface,
                     book: OptionBook, flow: FlowParams, level_scale: float, seed: int) -> DayMarket:
    rng = np.random.default_rng(seed)
    bars = session_minutes(bars_day)
    S, shocks = spot_path_from_bars(bars, rng)
    n = len(S)
    elapsed = np.arange(n) / n
    T = (book.du_after_today[None, :] + (1.0 - elapsed)[:, None]) / BUSINESS_DAYS_YEAR

    x, u = vol_factor_paths(shocks, book.n, flow, rng)
    atm = surface.atm_term(T) * level_scale + book.beta[None, :] * x[:, None]
    atm = np.maximum(atm, 0.05)
    sigma = np.maximum(atm * _smile_ratios(surface, book, S, T, atm) + u, 0.05)

    fair = bs_price(S[:, None], book.strikes[None, :], T, surface.rate, sigma, book.is_call[None, :])
    vega = bs_greeks(S[:, None], book.strikes[None, :], T, surface.rate, sigma, book.is_call[None, :])["vega"]
    half, comp_bid, comp_ask = _competitor_quotes(fair, vega, flow, rng)
    events = _draw_events(book, n, flow, rng)
    return DayMarket(ticker=ticker, day=day, rate=surface.rate, level_scale=level_scale, book=book,
                     S=S, T=T, sigma=sigma, fair=fair, half=half, comp_bid=comp_bid,
                     comp_ask=comp_ask, events=events)
