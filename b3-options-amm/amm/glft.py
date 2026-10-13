"""
glft.py
=======
Approximations fermees de Gueant-Lehalle-Fernandez-Tapia pour le market-making
multi-actifs (Gueant 2017, "Optimal market making", Applied Mathematical Finance ;
extension options : Baldacci, Bergault & Gueant 2021, Quantitative Finance).

Intensite de remplissage a distance delta du fair : Lambda(delta) = A exp(-k delta).
Fonction valeur quadratique theta(q) ~ -1/2 q' Gamma q, ou Gamma resout l'equation
de Riccati stationnaire
                    Gamma D Gamma = (gamma / 2) Sigma
    D = diag(A_i k_i Delta_i C_i),   C_i = (1 + gamma Delta_i / k_i)^-(1 + k_i / (gamma Delta_i))
=> Gamma = D^-1/2 ( D^1/2 (gamma/2) Sigma D^1/2 )^1/2 D^-1/2

Quotes optimaux (unites : prix par contrat, q en contrats, Delta = taille de clip) :
    delta_bid_i = c1_i + Delta_i Gamma_ii / 2 + (Gamma q)_i
    delta_ask_i = c1_i + Delta_i Gamma_ii / 2 - (Gamma q)_i
    c1_i        = ln(1 + gamma Delta_i / k_i) / (gamma Delta_i)
En mono-actif on retrouve exactement la formule GLFT 2013 :
    Gamma = sqrt( gamma sigma^2 / (2 A Delta k) * (1 + gamma Delta / k)^(1 + k / (gamma Delta)) ).

Sigma (covariance par seconde des prix d'options) est a deux facteurs :
    Sigma = sigma_f^2 e e' + w_d sigma_S^2 d d' + diag(sigma_u^2 nu_i^2)
    e_i = nu_i beta_i   (facteur 1 : niveau de vol ATM, regle racine du temps beta_i)
    d_i = Delta_i       (facteur 2 : spot, pondere par w_d = part du delta non couverte,
                         la couverture a bande laissant un risque directionnel residuel)
"""
from __future__ import annotations

import numpy as np


def glft_c1(gamma: float, k: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Demi-spread statique. Tend vers 1/k quand gamma -> 0."""
    x = gamma * clip / k
    return np.log1p(x) / (gamma * clip)


def glft_c_factor(gamma: float, k: np.ndarray, clip: np.ndarray) -> np.ndarray:
    x = gamma * clip / k
    return np.exp(-(1.0 + 1.0 / x) * np.log1p(x))


def factor_covariance(vega_contract: np.ndarray, beta: np.ndarray, factor_vol_per_s: float,
                      idio_vol_per_s: float, delta_contract: np.ndarray | None = None,
                      spot_vol_per_s: float = 0.0, delta_weight: float = 0.0,
                      independent: bool = False) -> np.ndarray:
    """
    Covariance par seconde (R$^2 par contrat^2). spot_vol_per_s en R$ d'action.
    independent=True ne garde que la diagonale (skew option par option).
    """
    e = vega_contract * beta
    sigma = factor_vol_per_s ** 2 * np.outer(e, e)
    if delta_contract is not None and delta_weight > 0:
        sigma = sigma + delta_weight * spot_vol_per_s ** 2 * np.outer(delta_contract, delta_contract)
    sigma = sigma + np.diag((idio_vol_per_s * vega_contract) ** 2)
    return np.diag(np.diag(sigma)) if independent else sigma


def _sym_sqrt(M: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(0.5 * (M + M.T))
    return (vecs * np.sqrt(np.maximum(vals, 0.0))) @ vecs.T


def riccati_gamma(sigma: np.ndarray, A: np.ndarray, k: np.ndarray, clip: np.ndarray,
                  gamma: float) -> np.ndarray:
    """Solution symetrique semi-definie positive de Gamma D Gamma = (gamma/2) Sigma."""
    d = A * k * clip * glft_c_factor(gamma, k, clip)
    if np.any(d <= 0):
        raise ValueError("D doit etre strictement positif (A, k, clip > 0)")
    d_half = np.sqrt(d)
    inner = _sym_sqrt(0.5 * gamma * (d_half[:, None] * sigma * d_half[None, :]))
    return inner / d_half[:, None] / d_half[None, :]


def single_asset_gamma(sigma2: float, A: float, k: float, clip: float, gamma: float) -> float:
    """Forme fermee GLFT 2013 (sert de test de coherence a riccati_gamma)."""
    x = gamma * clip / k
    return float(np.sqrt(gamma * sigma2 / (2.0 * A * clip * k) * (1.0 + x) ** (1.0 + 1.0 / x)))
