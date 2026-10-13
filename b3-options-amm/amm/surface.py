"""
surface.py
==========
Nappe de volatilite calibree sur la chaine B3 reelle.

1. Points de vol implicite : options OTM traitees a la derniere seance du snapshot,
   inversion Black-Scholes (spot = cloture 1m du jour, r = ln(1 + Selic)).
2. SVI par echeance (svi.fit_svi), pondere par sqrt(nb de trades).
3. Dynamique : le smile est exprime en moneyness standardisee
       z = ln(K/F) / (sigma_ATM sqrt(T))
   (sticky-delta) et le niveau ATM suit une structure par terme interpolee en T.
   Pour une seance passee, le niveau est rescale par RV20(jour) / RV20(snapshot)
   -> ratio IV/RV du snapshot conserve, aucune donnee future utilisee.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .conventions import year_fraction
from .pricing import implied_vol
from .svi import SVIParams, fit_svi

MIN_TRADES = 3
MIN_PREMIUM = 0.03
MAX_ABS_LOG_MONEYNESS = 0.35
TERM_BETA_REF_T = 21 / 252
TERM_BETA_BOUNDS = (0.5, 2.5)


@dataclass(frozen=True)
class ExpirySmile:
    expiry: dt.date
    T_snapshot: float
    forward: float
    atm_vol: float
    svi: SVIParams
    n_points: int
    rmse_vol: float

    def vol_ratio(self, z: np.ndarray) -> np.ndarray:
        """sigma(z) / sigma_ATM au snapshot, pour une moneyness standardisee z."""
        k = np.asarray(z) * self.atm_vol * np.sqrt(self.T_snapshot)
        return self.svi.implied_vol(k, self.T_snapshot) / self.atm_vol


@dataclass(frozen=True)
class VolSurface:
    ticker: str
    valuation_date: dt.date
    spot: float
    rate: float
    smiles: tuple[ExpirySmile, ...]
    rv20_snapshot: float

    def smile_for(self, expiry: dt.date) -> ExpirySmile:
        for smile in self.smiles:
            if smile.expiry == expiry:
                return smile
        raise KeyError(f"Pas de smile calibre pour {expiry}")

    def atm_term(self, T: np.ndarray) -> np.ndarray:
        """Vol ATM interpolee lineairement en variance totale, extrapolation plate."""
        Ts = np.array([s.T_snapshot for s in self.smiles])
        ws = np.array([s.atm_vol ** 2 * s.T_snapshot for s in self.smiles])
        T = np.asarray(T, dtype=float)
        w = np.interp(T, Ts, ws)
        w = np.where(T < Ts[0], self.smiles[0].atm_vol ** 2 * T, w)
        w = np.where(T > Ts[-1], self.smiles[-1].atm_vol ** 2 * T, w)
        return np.sqrt(np.maximum(w, 1e-12) / np.maximum(T, 1e-10))

    def vols(self, S: float, strikes: np.ndarray, T: np.ndarray, expiries: list[dt.date],
             level_scale: float = 1.0) -> np.ndarray:
        """Vol implicite modele pour un panier d'options (une echeance par option)."""
        T = np.asarray(T, dtype=float)
        atm = self.atm_term(T) * level_scale
        forward = S * np.exp(self.rate * T)
        z = np.log(np.asarray(strikes) / forward) / (atm * np.sqrt(T))
        ratios = np.array([self.smile_for(e).vol_ratio(zi) for e, zi in zip(expiries, z)])
        return atm * ratios


def term_beta(T: np.ndarray) -> np.ndarray:
    """Sensibilite d'une vol ATM au facteur de vol (regle de la racine du temps)."""
    beta = np.sqrt(TERM_BETA_REF_T / np.maximum(np.asarray(T, dtype=float), 1e-6))
    return np.clip(beta, *TERM_BETA_BOUNDS)


def implied_vol_points(chain: pd.DataFrame, spot: float, rate: float,
                       valuation: dt.date) -> pd.DataFrame:
    """Points (k, T, IV) exploitables : OTM, traites le jour du snapshot, prix > 3 ticks."""
    df = chain[(chain["last_date"].dt.date == valuation)
               & (chain["n_trades"] >= MIN_TRADES)
               & (chain["last"] >= MIN_PREMIUM)].copy()
    df["T"] = [year_fraction(valuation, e, 1.0) for e in df["expiry"]]
    df = df[df["T"] > 0]
    df["forward"] = spot * np.exp(rate * df["T"])
    df["k"] = np.log(df["strike"] / df["forward"])
    otm = np.where(df["is_call"], df["k"] >= 0, df["k"] < 0)
    df = df[otm & (df["k"].abs() <= MAX_ABS_LOG_MONEYNESS)]
    df["iv"] = implied_vol(df["last"].to_numpy(), spot, df["strike"].to_numpy(),
                           df["T"].to_numpy(), rate, df["is_call"].to_numpy())
    return df.dropna(subset=["iv"]).query("0.05 < iv < 2.5").reset_index(drop=True)


def _fit_expiry(points: pd.DataFrame, expiry: dt.date) -> ExpirySmile | None:
    pts = points[points["expiry"] == expiry]
    if len(pts) < 5:
        return None
    T = float(pts["T"].iloc[0])
    k = pts["k"].to_numpy()
    w = pts["iv"].to_numpy() ** 2 * T
    params = fit_svi(k, w, weights=np.sqrt(pts["n_trades"].to_numpy()))
    fitted = params.implied_vol(k, T)
    atm = float(params.implied_vol(np.array([0.0]), T)[0])
    rmse = float(np.sqrt(np.mean((fitted - pts["iv"].to_numpy()) ** 2)))
    return ExpirySmile(expiry=expiry, T_snapshot=T, forward=float(pts["forward"].iloc[0]),
                       atm_vol=atm, svi=params, n_points=len(pts), rmse_vol=rmse)


def calibrate_surface(ticker: str, chain: pd.DataFrame, spot: float, rate: float,
                      valuation: dt.date, rv20: float) -> tuple[VolSurface, pd.DataFrame]:
    points = implied_vol_points(chain, spot, rate, valuation)
    smiles = [s for e in sorted(points["expiry"].unique()) if (s := _fit_expiry(points, e))]
    if not smiles:
        raise ValueError(f"Aucune echeance calibrable pour {ticker} au {valuation}")
    surface = VolSurface(ticker=ticker, valuation_date=valuation, spot=spot, rate=rate,
                         smiles=tuple(smiles), rv20_snapshot=rv20)
    return surface, points
