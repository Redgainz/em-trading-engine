"""
universe.py
===========
Selection du book d'options cote par seance + modele d'intensite du flux
calibre sur l'activite reelle de la chaine B3.

Modele d'activite (GLM de Poisson, lien log, IRLS, sur le snapshot) :
    E[n_trades_i] = exp(b0 + b1 |z_i| + b2 z_i^2 + b3 log T_i + b4 1{call})
    log(contrats / trade) = c0 + c1 |z_i|     (OLS)
Le GLM de Poisson estime l'intensite moyenne sans le biais de Jensen d'un OLS
sur log(1 + n), et integre proprement les options non traitees (n = 0).
ou z = ln(K/F) / (sigma_ATM sqrt T) est la moneyness standardisee.
L'intensite d'une seance passee est rescalee par l'activite du sous-jacent
(volume Yahoo du jour / volume median), observable en temps reel.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .conventions import SESSION_SECONDS, business_days_between, year_fraction
from .surface import VolSurface, term_beta

MIN_DAYS_TO_EXPIRY = 3
MAX_ABS_Z = 2.2
Z_GRID = np.linspace(-MAX_ABS_Z * 0.9, MAX_ABS_Z * 0.9, 11)
MIN_LAMBDA_PER_DAY = 5.0


@dataclass(frozen=True)
class ActivityModel:
    trade_coef: np.ndarray
    size_coef: np.ndarray

    @staticmethod
    def _trade_design(abs_z, T, is_call):
        return np.column_stack([np.ones_like(abs_z), abs_z, abs_z ** 2,
                                np.log(T), is_call.astype(float)])

    def trades_per_day(self, abs_z, T, is_call) -> np.ndarray:
        pred = self._trade_design(abs_z, T, is_call) @ self.trade_coef
        return np.maximum(np.exp(np.minimum(pred, 12.0)), MIN_LAMBDA_PER_DAY)

    def contracts_per_trade(self, abs_z) -> np.ndarray:
        pred = np.column_stack([np.ones_like(abs_z), abs_z]) @ self.size_coef
        return np.clip(np.exp(pred), 1.0, 50.0)


def poisson_glm(X: np.ndarray, y: np.ndarray, n_iter: int = 50, ridge: float = 1e-6) -> np.ndarray:
    """GLM de Poisson (lien log) par moindres carres iterativement reponderes."""
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(max(y.mean(), 1e-3))
    for _ in range(n_iter):
        mu = np.exp(np.clip(X @ beta, -30, 30))
        z = X @ beta + (y - mu) / mu
        XtW = X.T * mu
        step = np.linalg.solve(XtW @ X + ridge * np.eye(X.shape[1]), XtW @ z)
        if np.max(np.abs(step - beta)) < 1e-8:
            return step
        beta = step
    return beta


def standardized_moneyness(surface: VolSurface, S: float, K: np.ndarray, T: np.ndarray) -> np.ndarray:
    atm = surface.atm_term(T)
    forward = S * np.exp(surface.rate * T)
    return np.log(K / forward) / (atm * np.sqrt(T))


def fit_activity_model(chain: pd.DataFrame, surface: VolSurface) -> ActivityModel:
    val = surface.valuation_date
    df = chain.copy()
    df["T"] = [year_fraction(val, e, 1.0) for e in df["expiry"]]
    df = df[df["T"] > 0]
    df["z"] = standardized_moneyness(surface, surface.spot, df["strike"].to_numpy(), df["T"].to_numpy())
    df = df[df["z"].abs() <= 3.0]
    traded_today = df["last_date"].dt.date == val
    n = np.where(traded_today, df["n_trades"], 0)
    abs_z, T, is_call = df["z"].abs().to_numpy(), df["T"].to_numpy(), df["is_call"].to_numpy()
    X = ActivityModel._trade_design(abs_z, T, is_call)
    trade_coef = poisson_glm(X, n.astype(float))

    sized = df[traded_today & (df["n_trades"] > 0) & (df["last"] > 0)]
    contracts = sized["volume_brl"] / (sized["n_trades"] * sized["last"] * 100.0)
    ok = np.isfinite(contracts) & (contracts > 0)
    Xs = np.column_stack([np.ones(ok.sum()), sized["z"].abs().to_numpy()[ok]])
    size_coef, *_ = np.linalg.lstsq(Xs, np.log(contracts[ok].to_numpy()), rcond=None)
    return ActivityModel(trade_coef=trade_coef, size_coef=size_coef)


@dataclass(frozen=True)
class OptionBook:
    """Instruments cotes pendant une seance (tableaux alignes, longueur N)."""
    tickers: tuple[str, ...]
    strikes: np.ndarray
    is_call: np.ndarray
    expiries: tuple[dt.date, ...]
    du_after_today: np.ndarray       # jours ouvres restants apres la seance
    beta: np.ndarray                 # sensibilite au facteur de vol ATM
    lambda_per_s: np.ndarray         # intensite bruit par cote (achat ou vente), par seconde
    mean_size: np.ndarray            # contrats par trade

    @property
    def n(self) -> int:
        return len(self.tickers)


def _pick_strikes(cands: pd.DataFrame, forward: float) -> pd.DataFrame:
    """Une option OTM par strike (call si K >= F), doublons A/E -> la plus traitee."""
    cands = cands[np.where(cands["is_call"], cands["strike"] >= forward, cands["strike"] < forward)]
    cands = cands.sort_values("n_trades", ascending=False).drop_duplicates("strike")
    cands = cands.assign(abs_z=cands["z"].abs()).reset_index(drop=True)
    if cands.empty:
        return cands
    # Strikes repartis sur le smile : le plus proche de chaque point d'une grille en z
    picks = {int(np.argmin(np.abs(cands["z"].to_numpy() - target))) for target in Z_GRID}
    return cands.loc[sorted(picks)]


def build_option_book(chain: pd.DataFrame, surface: VolSurface, activity: ActivityModel,
                      day: dt.date, spot_open: float, activity_ratio: float) -> OptionBook:
    frames = []
    for smile in surface.smiles:
        T_open = year_fraction(day, smile.expiry, 0.0)
        if business_days_between(day, smile.expiry) + 1 < MIN_DAYS_TO_EXPIRY:
            continue
        cands = chain[chain["expiry"] == smile.expiry].copy()
        cands["z"] = standardized_moneyness(surface, spot_open, cands["strike"].to_numpy(),
                                            np.full(len(cands), T_open))
        cands = cands[cands["z"].abs() <= MAX_ABS_Z]
        forward = spot_open * np.exp(surface.rate * T_open)
        frames.append(_pick_strikes(cands, forward).assign(T_open=T_open))
    if not frames:
        raise ValueError(f"Aucune option cotable le {day}")
    book = pd.concat(frames).sort_values(["expiry", "strike"]).reset_index(drop=True)

    abs_z, T, is_call = book["abs_z"].to_numpy(), book["T_open"].to_numpy(), book["is_call"].to_numpy()
    per_day = activity.trades_per_day(abs_z, T, is_call) * activity_ratio
    return OptionBook(
        tickers=tuple(book["ticker"]),
        strikes=book["strike"].to_numpy(dtype=float),
        is_call=is_call.astype(bool),
        expiries=tuple(book["expiry"]),
        du_after_today=np.array([business_days_between(day, e) for e in book["expiry"]],
                                dtype=float),
        beta=term_beta(T),
        lambda_per_s=per_day / 2.0 / SESSION_SECONDS,
        mean_size=activity.contracts_per_trade(abs_z),
    )
