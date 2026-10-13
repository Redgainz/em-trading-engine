"""
params.py
=========
Parametres immuables (frozen dataclasses) du simulateur et de la strategie.

  - FlowParams     : environnement de marche (flux d'ordres, concurrents, dynamique de vol)
  - StrategyParams : modele de quoting (le "nouveau modele" et ses ablations)
  - HedgeParams    : couverture delta en actions
  - RiskParams     : limites pre-trade et kill-switch

VERSIONS documente l'iteration structurelle du modele, v0 -> v4, chaque version
ajoutant une brique a la precedente (ablation cumulative). v5 = v4 avec parametres
calibres en walk-forward (calibration.py).
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class FlowParams:
    flow_scale: float = 1.0             # multiplicateur des intensites calibrees sur la chaine
    informed_ratio: float = 0.50        # intensite informee / intensite bruit
    informed_horizon_s: int = 60        # l'informe connait le fair a t + H
    informed_min_edge_ticks: float = 0.0
    informed_size_mult: float = 2.0
    price_sensitivity: float = 1.5      # k_i = price_sensitivity / h_i (elasticite du flux bruit)
    max_flow_mult: float = 3.0          # plafond de l'intensite quand le marche se resserre
    queue_share: float = 0.35           # part de la file obtenue en joignant le meilleur prix
    comp_vol_halfspread: float = 0.0012  # demi-spread concurrent en vol (0.12 pt de vol)
    comp_rel_halfspread: float = 0.003  # + 0.3% du premium
    comp_min_halfspread_ticks: float = 0.5
    comp_noise: float = 0.20            # bruit du mid concurrent, en fraction de h
    comp_noise_autocorr: float = 0.97   # par seconde (quotes concurrents "stale")
    vol_of_vol_daily: float = 0.012     # ecart-type quotidien du facteur de vol ATM
    vol_halflife_s: float = 6 * 3600.0
    spot_vol_corr: float = -0.6
    idio_vol_std: float = 0.003         # residu de smile statique par option (vol)
    idio_vol_daily: float = 0.002       # diffusion du residu par option
    max_trade_contracts: int = 100


@dataclass(frozen=True)
class StrategyParams:
    name: str = "custom"
    spread_model: str = "glft"          # "fixed" | "glft"
    skew_model: str = "factor"          # "none" | "independent" | "factor"
    gamma: float = 5e-3                 # aversion au risque (1 / R$)
    kappa: float = 1.5                  # elasticite supposee : k_i = kappa / h_observe
    fixed_halfspread_mult: float = 1.0  # spread "fixed" = h_observe * mult
    factor_vol_daily: float = 0.012     # estimation MM de la vol du facteur ATM
    idio_vol_daily: float = 0.002
    delta_risk_weight: float = 0.05     # part du risque spot non couverte (bande de hedge)
    clip_contracts: int = 10
    min_edge_ticks: float = 0.0
    queue_aware: bool = True            # n'ameliore le meilleur concurrent que d'1 tick
    as_widening: float = 0.5            # phi : elargissement par R$ de decay couvert attendu
    as_horizon_s: int = 30
    as_ewma_alpha: float = 0.05
    as_prior_weight: float = 10.0       # shrinkage vers le markout global
    flow_fade: float = 0.002            # R$ de spot par contrat-delta signe recu
    flow_fade_halflife_s: float = 20.0
    fit_factor_alpha: float = 0.3       # filtre du facteur ATM sur les mids ecran
    fit_idio_alpha: float = 0.05
    reprice_every_s: int = 5
    risk_matrix_every_s: int = 60


@dataclass(frozen=True)
class HedgeParams:
    band_mode: str = "fixed"            # "fixed" | "ww" (Whalley-Wilmott)
    band_shares: float = 3000.0
    ww_risk_aversion: float = 1e-4
    min_band_shares: float = 500.0
    hedge_to: str = "edge"              # "edge" (bord de bande) | "zero"


@dataclass(frozen=True)
class RiskParams:
    max_contracts_per_option: int = 200
    max_vega_brl_per_volpt: float = 4000.0
    max_daily_loss_brl: float = 40000.0


@dataclass(frozen=True)
class Config:
    strategy: StrategyParams = StrategyParams()
    hedge: HedgeParams = HedgeParams()
    risk: RiskParams = RiskParams()


_NAIVE = StrategyParams(name="v0_naive", spread_model="fixed", skew_model="none",
                        queue_aware=False, as_widening=0.0, flow_fade=0.0)
_SLOW_HEDGE = HedgeParams(band_mode="fixed", band_shares=3000.0)


def _version(name: str, hedge: HedgeParams = _SLOW_HEDGE, **changes) -> Config:
    return Config(strategy=replace(_NAIVE, name=name, **changes), hedge=hedge)


# Iteration du modele : chaque version ajoute UNE brique a la precedente.
VERSIONS: dict[str, Config] = {
    # v0 : theo +/- demi-spread ecran, pas de gestion d'inventaire, couverture lente
    "v0_naive": Config(strategy=_NAIVE, hedge=_SLOW_HEDGE),
    # v1 : spread optimal GLFT + skew d'inventaire option par option
    "v1_glft": _version("v1_glft", spread_model="glft", skew_model="independent"),
    # v2 : skew dans l'espace des facteurs vol ATM + spot residuel (Riccati multi-actifs)
    "v2_factor": _version("v2_factor", spread_model="glft", skew_model="factor"),
    # v3 : placement conscient de la file (n'ameliore l'ecran que d'1 tick)
    "v3_queue": _version("v3_queue", spread_model="glft", skew_model="factor", queue_aware=True),
    # v4 : fade sur le flux -> le flux options informe sur le spot
    "v4_fade": _version("v4_fade", spread_model="glft", skew_model="factor", queue_aware=True,
                        flow_fade=0.002),
    # v5 : parametres calibres en walk-forward (amm/calibration.py, scripts/04)
}
