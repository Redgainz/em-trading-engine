"""
svi.py
======
Parametrisation SVI "raw" de Gatheral pour le smile d'une echeance :

    w(k) = a + b * ( rho (k - m) + sqrt((k - m)^2 + sigma^2) )

avec w = variance totale implicite (sigma_imp^2 * T) et k = ln(K / F).

Calibration quasi-explicite (De Marco & Martini, Zeliade 2009) :
  - a (m, sigma) fixes, w est lineaire en (a, d, c) avec y = (k - m) / sigma :
        w = a + d y + c sqrt(y^2 + 1),   d = rho b sigma,   c = b sigma
    -> moindres carres ponderes fermes
  - boucle externe Nelder-Mead (numpy pur) sur (m, ln sigma)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

SIGMA_MIN, SIGMA_MAX = 0.02, 2.0


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k: np.ndarray) -> np.ndarray:
        x = np.asarray(k, dtype=float) - self.m
        return self.a + self.b * (self.rho * x + np.sqrt(x * x + self.sigma ** 2))

    def implied_vol(self, k: np.ndarray, T: float) -> np.ndarray:
        if T <= 0:
            raise ValueError("T doit etre > 0")
        return np.sqrt(np.maximum(self.total_variance(k), 1e-10) / T)


def butterfly_density(params: SVIParams, k: np.ndarray) -> np.ndarray:
    """
    g(k) de Gatheral-Jacquier : g >= 0 partout <=> pas d'arbitrage papillon.
    """
    k = np.asarray(k, dtype=float)
    x = k - params.m
    root = np.sqrt(x * x + params.sigma ** 2)
    w = params.total_variance(k)
    w1 = params.b * (params.rho + x / root)
    w2 = params.b * params.sigma ** 2 / root ** 3
    w = np.maximum(w, 1e-12)
    return (1.0 - k * w1 / (2.0 * w)) ** 2 - (w1 ** 2 / 4.0) * (1.0 / w + 0.25) + w2 / 2.0


def nelder_mead(f: Callable[[np.ndarray], float], x0: np.ndarray, step: float = 0.1,
                max_iter: int = 400, tol: float = 1e-10) -> np.ndarray:
    """Nelder-Mead standard (reflexion, expansion, contraction, retrecissement)."""
    n = len(x0)
    simplex = [np.asarray(x0, dtype=float)]
    for i in range(n):
        vertex = np.array(x0, dtype=float)
        vertex[i] += step
        simplex.append(vertex)
    values = [f(v) for v in simplex]

    for _ in range(max_iter):
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        if abs(values[-1] - values[0]) < tol:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        reflected = centroid + (centroid - simplex[-1])
        f_r = f(reflected)
        if f_r < values[0]:
            expanded = centroid + 2.0 * (centroid - simplex[-1])
            f_e = f(expanded)
            simplex[-1], values[-1] = (expanded, f_e) if f_e < f_r else (reflected, f_r)
        elif f_r < values[-2]:
            simplex[-1], values[-1] = reflected, f_r
        else:
            contracted = centroid + 0.5 * (simplex[-1] - centroid)
            f_c = f(contracted)
            if f_c < values[-1]:
                simplex[-1], values[-1] = contracted, f_c
            else:
                best = simplex[0]
                simplex = [best] + [best + 0.5 * (v - best) for v in simplex[1:]]
                values = [values[0]] + [f(v) for v in simplex[1:]]
    return simplex[int(np.argmin(values))]


def _inner_linear_fit(k, w, weights, m, sigma) -> tuple[SVIParams, float]:
    """Moindres carres ponderes en (a, d, c) puis projection sur le domaine admissible."""
    y = (k - m) / sigma
    design = np.column_stack([np.ones_like(y), y, np.sqrt(y * y + 1.0)])
    sw = np.sqrt(weights)
    coef, *_ = np.linalg.lstsq(design * sw[:, None], w * sw, rcond=None)
    a, d, c = coef
    c = float(np.clip(c, 1e-8, 4.0 * sigma))
    d = float(np.clip(d, -c * 0.999, c * 0.999))
    a = float(np.clip(a, -c + 1e-10, np.max(w)))
    params = SVIParams(a=a, b=c / sigma, rho=d / c, m=float(m), sigma=float(sigma))
    resid = params.total_variance(k) - w
    return params, float(np.sum(weights * resid * resid))


def fit_svi(k: np.ndarray, total_var: np.ndarray, weights: np.ndarray | None = None) -> SVIParams:
    """Calibre un smile SVI sur des points (k, w). Requiert au moins 5 points."""
    k = np.asarray(k, dtype=float)
    w = np.asarray(total_var, dtype=float)
    if k.size < 5 or k.size != w.size:
        raise ValueError(f"fit_svi : il faut >= 5 points coherents (recu {k.size})")
    weights = np.ones_like(k) if weights is None else np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    def objective(x: np.ndarray) -> float:
        m, log_sigma = x
        sigma = float(np.clip(np.exp(log_sigma), SIGMA_MIN, SIGMA_MAX))
        return _inner_linear_fit(k, w, weights, m, sigma)[1]

    best_x, best_val = None, np.inf
    for m0 in (k[np.argmin(w)], 0.0):
        for s0 in (np.log(0.05), np.log(0.2)):
            x = nelder_mead(objective, np.array([m0, s0]), step=0.1)
            val = objective(x)
            if val < best_val:
                best_x, best_val = x, val
    sigma = float(np.clip(np.exp(best_x[1]), SIGMA_MIN, SIGMA_MAX))
    return _inner_linear_fit(k, w, weights, best_x[0], sigma)[0]
