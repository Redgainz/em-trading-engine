"""
pricing.py
==========
Black-Scholes vectorise (numpy pur) adapte aux options B3 :
pas de dividende (strikes ajustes des proventos), taux continu r = ln(1 + Selic).

Fonctions :
  - bs_price, bs_greeks : prix et greeks sur tableaux (broadcast numpy)
  - implied_vol : inversion Newton safeguardee par bissection, vectorisee
Conventions :
  - is_call : tableau booleen (True = call)
  - vega par 1.00 de vol (multiplier par 0.01 pour 1 point de vol)
  - theta par annee (diviser par 252 pour un jour ouvre)
"""
from __future__ import annotations

import numpy as np

SQRT_2PI = np.sqrt(2.0 * np.pi)
VOL_MIN, VOL_MAX = 1e-4, 5.0


def norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.square(x)) / SQRT_2PI


def _erfc(x: np.ndarray) -> np.ndarray:
    """erfc Chebyshev (Numerical Recipes 6.2), erreur relative < 1.2e-7 partout."""
    z = np.abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    poly = (-1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418
            + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398
            + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    ans = t * np.exp(-z * z + poly)
    return np.where(x >= 0.0, ans, 2.0 - ans)


def norm_cdf(x: np.ndarray) -> np.ndarray:
    """Precision relative conservee dans les queues (important pour l'IV OTM)."""
    return 0.5 * _erfc(-np.asarray(x, dtype=float) / np.sqrt(2.0))


def _d1_d2(S, K, T, r, sigma):
    T = np.maximum(T, 1e-10)
    sigma = np.maximum(sigma, 1e-8)
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    return d1, d1 - sigma * sqrt_t, sqrt_t


def bs_price(S, K, T, r, sigma, is_call) -> np.ndarray:
    """Prix europeen par action. Gere T <= 0 (payoff intrinseque)."""
    S, K, T, sigma = (np.asarray(a, dtype=float) for a in (S, K, T, sigma))
    is_call = np.asarray(is_call, dtype=bool)
    d1, d2, _ = _d1_d2(S, K, T, r, sigma)
    df = np.exp(-r * np.maximum(T, 0.0))
    call = S * norm_cdf(d1) - K * df * norm_cdf(d2)
    put = K * df * norm_cdf(-d2) - S * norm_cdf(-d1)
    price = np.where(is_call, call, put)
    intrinsic = np.where(is_call, np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
    return np.where(T > 0.0, price, intrinsic)


def bs_greeks(S, K, T, r, sigma, is_call) -> dict[str, np.ndarray]:
    """delta, gamma, vega (par 1.00 vol), theta (par an) — par action."""
    S, K, T, sigma = (np.asarray(a, dtype=float) for a in (S, K, T, sigma))
    is_call = np.asarray(is_call, dtype=bool)
    d1, d2, sqrt_t = _d1_d2(S, K, T, r, sigma)
    pdf = norm_pdf(d1)
    df = np.exp(-r * np.maximum(T, 0.0))
    delta = np.where(is_call, norm_cdf(d1), norm_cdf(d1) - 1.0)
    gamma = pdf / (S * np.maximum(sigma, 1e-8) * sqrt_t)
    vega = S * pdf * sqrt_t
    theta_common = -S * pdf * sigma / (2.0 * sqrt_t)
    theta = np.where(is_call,
                     theta_common - r * K * df * norm_cdf(d2),
                     theta_common + r * K * df * norm_cdf(-d2))
    alive = T > 0.0
    return {
        "delta": np.where(alive, delta, 0.0),
        "gamma": np.where(alive, gamma, 0.0),
        "vega": np.where(alive, vega, 0.0),
        "theta": np.where(alive, theta, 0.0),
    }


def no_arbitrage_bounds(S, K, T, r, is_call) -> tuple[np.ndarray, np.ndarray]:
    df = np.exp(-r * T)
    lower = np.where(is_call, np.maximum(S - K * df, 0.0), np.maximum(K * df - S, 0.0))
    upper = np.where(is_call, S, K * df)
    return lower, upper


def implied_vol(price, S, K, T, r, is_call, n_iter: int = 60, tol: float = 1e-8) -> np.ndarray:
    """
    Vol implicite vectorisee : Newton (pas vega) dans un bracket [lo, hi]
    mis a jour a chaque iteration ; bissection quand Newton sort du bracket.
    Renvoie NaN si le prix viole les bornes d'arbitrage.
    """
    price, S, K, T = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (price, S, K, T)))
    is_call = np.broadcast_to(np.asarray(is_call, dtype=bool), price.shape)
    lower, upper = no_arbitrage_bounds(S, K, T, r, is_call)
    valid = (price > lower + 1e-12) & (price < upper - 1e-12) & (T > 0)

    lo = np.full(price.shape, VOL_MIN)
    hi = np.full(price.shape, VOL_MAX)
    # Point de depart Brenner-Subrahmanyam, borne dans le bracket
    sigma = np.clip(np.sqrt(2.0 * np.pi / np.maximum(T, 1e-10)) * price / S, 0.05, 2.0)

    for _ in range(n_iter):
        model = bs_price(S, K, T, r, sigma, is_call)
        diff = model - price
        if np.all(np.abs(diff[valid]) < tol):
            break
        hi = np.where(diff > 0, sigma, hi)
        lo = np.where(diff <= 0, sigma, lo)
        vega = bs_greeks(S, K, T, r, sigma, is_call)["vega"]
        newton = sigma - diff / np.maximum(vega, 1e-12)
        inside = (newton > lo) & (newton < hi) & (vega > 1e-10)
        sigma = np.where(inside, newton, 0.5 * (lo + hi))

    return np.where(valid, sigma, np.nan)
