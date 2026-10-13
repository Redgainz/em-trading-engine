"""
hedger.py
=========
Couverture delta du book d'options par l'action (lot de 100).

Bande de non-transaction :
  - "fixed" : |delta| <= band_shares
  - "ww"    : Whalley-Wilmott (1997), H = (3/2 * lambda S Gamma_port^2 / gamma_h)^(1/3)
              avec lambda = cout proportionnel (demi-spread + frais) / S
Hors bande, on revient au bord de la bande ("edge", optimal sous couts proportionnels)
ou a zero ("zero").
"""
from __future__ import annotations

import numpy as np

from .conventions import STOCK_FEE_RATE, STOCK_HALF_SPREAD, STOCK_LOT
from .params import HedgeParams


def hedge_band(params: HedgeParams, S: float, gamma_shares: float) -> float:
    if params.band_mode == "fixed":
        return params.band_shares
    cost_rate = STOCK_HALF_SPREAD / S + STOCK_FEE_RATE
    ww = (1.5 * cost_rate * S * gamma_shares ** 2 / params.ww_risk_aversion) ** (1.0 / 3.0)
    return float(max(ww, params.min_band_shares))


def hedge_order(params: HedgeParams, delta_shares: float, S: float, gamma_shares: float) -> int:
    """Quantite d'actions a traiter (signee, multiple du lot), 0 si dans la bande."""
    band = hedge_band(params, S, gamma_shares)
    if abs(delta_shares) <= band:
        return 0
    target = np.sign(delta_shares) * band if params.hedge_to == "edge" else 0.0
    lots = np.round((target - delta_shares) / STOCK_LOT)
    return int(lots * STOCK_LOT)


def execution_cost(shares: int, S: float) -> tuple[float, float]:
    """(prix d'execution agressif, frais) pour un ordre de couverture."""
    price = S + np.sign(shares) * STOCK_HALF_SPREAD
    return float(price), float(abs(shares) * price * STOCK_FEE_RATE)
