"""
calibration.py
==============
Parametrisation du modele par recherche aleatoire, validee en walk-forward.

  - Espace de recherche : aversion au risque gamma, elasticite kappa, taille de clip,
    poids du risque spot residuel, fade sur flux, elargissement AS, bande de hedge.
  - Objectif (equivalent certain) sur les seances d'entrainement :
        CE = moyenne(PnL_flat quotidien du desk) - lambda * ecart-type
    PnL "flat EOD" = inventaire liquide en fin de seance -> penalise le portage.
  - Selection sur train, evaluation sur test (seances posterieures, jamais vues).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from .data import DATA_DIR
from .params import Config, HedgeParams, StrategyParams, VERSIONS

CALIBRATED_PATH = os.path.join(DATA_DIR, "calibrated_params.json")
RISK_PENALTY = 1.0
BASE_VERSION = "v4_fade"


def sample_config(rng: np.random.Generator, label: str) -> Config:
    base = VERSIONS[BASE_VERSION]
    strategy = replace(
        base.strategy, name=label,
        gamma=float(10 ** rng.uniform(-3.0, -1.3)),
        kappa=float(rng.uniform(0.8, 2.5)),
        clip_contracts=int(rng.choice([5, 10, 20])),
        delta_risk_weight=float(rng.choice([0.0, 0.05, 0.2, 0.5])),
        flow_fade=float(rng.choice([0.0, 0.001, 0.002, 0.004, 0.008])),
        as_widening=float(rng.choice([0.0, 0.25, 0.5, 1.0])),
    )
    if rng.random() < 0.5:
        hedge = HedgeParams(band_mode="fixed", band_shares=float(rng.choice([500, 1000, 2000, 3000, 5000])))
    else:
        hedge = HedgeParams(band_mode="ww", ww_risk_aversion=float(10 ** rng.uniform(-5.5, -3.5)),
                            min_band_shares=float(rng.choice([300, 1000])))
    return Config(strategy=strategy, hedge=hedge, risk=base.risk)


def candidate_grid(n_candidates: int, seed: int = 7) -> dict[str, Config]:
    rng = np.random.default_rng(seed)
    configs = {BASE_VERSION: VERSIONS[BASE_VERSION]}
    for k in range(n_candidates):
        label = f"cand_{k:03d}"
        configs[label] = sample_config(rng, label)
    return configs


def score(rows: pd.DataFrame, pnl_col: str = "pnl_flat") -> pd.DataFrame:
    """Score par configuration a partir des lignes (ticker, seance, config)."""
    daily = rows.groupby(["config", "day"])[pnl_col].sum().unstack("day")
    mean, std = daily.mean(axis=1), daily.std(axis=1, ddof=1)
    out = pd.DataFrame({"mean_daily": mean, "std_daily": std,
                        "ce": mean - RISK_PENALTY * std,
                        "sharpe_ann": mean / std * np.sqrt(252)})
    return out.sort_values("ce", ascending=False)


def config_to_dict(cfg: Config) -> dict:
    return {"strategy": asdict(cfg.strategy), "hedge": asdict(cfg.hedge), "risk": asdict(cfg.risk)}


def config_from_dict(payload: dict, name: str | None = None) -> Config:
    strategy = StrategyParams(**payload["strategy"])
    if name:
        strategy = replace(strategy, name=name)
    return Config(strategy=strategy, hedge=HedgeParams(**payload["hedge"]),
                  risk=Config().risk.__class__(**payload["risk"]))


def save_calibrated(cfg: Config, meta: dict) -> None:
    with open(CALIBRATED_PATH, "w") as fh:
        json.dump({"config": config_to_dict(cfg), "meta": meta}, fh, indent=2, default=str)


def load_calibrated(name: str = "v5_calibrated") -> Config | None:
    if not os.path.exists(CALIBRATED_PATH):
        return None
    with open(CALIBRATED_PATH) as fh:
        return config_from_dict(json.load(fh)["config"], name=name)


def all_versions() -> dict[str, Config]:
    """v0 -> v4 (structurels) + v5 calibre en walk-forward s'il existe."""
    versions = dict(VERSIONS)
    calibrated = load_calibrated()
    if calibrated is not None:
        versions["v5_calibrated"] = calibrated
    return versions
