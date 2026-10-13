"""
strategy.py
===========
Le modele de quoting : Factor-GLFT avec apprentissage en ligne.

A chaque reprice (toutes les `reprice_every_s` secondes) :
  1. Fit a l'ecran : residus r_i = (mid_i - theo_i) / vega_i projetes sur le facteur
     ATM (moindres carres ponderes par (vega/h)^2) -> x_hat, puis residus idiosyncratiques
     u_hat_i par EWMA. C'est le "surface fitting" temps reel du desk.
  2. Theo, delta, gamma, vega par Black-Scholes sur la vol modele.
Entre deux reprices, le theo suit le spot par developpement delta-gamma
(c'est ce que fait un moteur HFT : pas de repricing complet a chaque tick).

Quotes (par action) :
    bid_i = theo_i - half_i - skew_i ,  ask_i = theo_i + half_i - skew_i
    half_i = c1_i + clip Gamma_ii / 2 + AS_i + edge_min        (GLFT + adverse selection)
    skew_i = (Gamma q)_i            (inventaire en espace facteurs : vol ATM + spot residuel)
puis contraintes de marche : arrondi au tick, post-only, placement conscient de la file.

AS_i : cout d'adverse selection attendu par fill = decay delta-couvert de l'edge
        d_i = E[edge a l'execution - markout delta-couvert a `as_horizon_s`]
      (EWMA par option, shrinke vers la moyenne globale). Un cout attendu d par fill
      decale le quote optimal de d (Glosten-Milgrom) : half_i += phi * max(d_i, 0).
Flow fade : chaque fill deplace le spot de reference de flow_fade * delta signe,
avec demi-vie flow_fade_halflife_s (le flux options informe sur le sous-jacent).

Etat mutable assume : l'objet vit dans la boucle chaude du backtest.
"""
from __future__ import annotations

import numpy as np

from .conventions import BUSINESS_DAYS_YEAR, CONTRACT_SIZE, SESSION_SECONDS, TICK
from .glft import factor_covariance, glft_c1, riccati_gamma
from .params import RiskParams, StrategyParams
from .pricing import bs_greeks, bs_price
from .surface import VolSurface
from .universe import OptionBook

MAX_RESIDUAL_VOL = 0.03


class OptionQuoter:
    def __init__(self, book: OptionBook, surface: VolSurface, level_scale: float,
                 params: StrategyParams, risk: RiskParams) -> None:
        self.book, self.surface, self.p, self.risk = book, surface, params, risk
        self.level_scale = level_scale
        self.spot_vol_ann = float(surface.atm_term(np.array([21 / BUSINESS_DAYS_YEAR]))[0] * level_scale)
        n = book.n
        self.x_hat = 0.0
        self.u_hat = np.zeros(n)
        self.S0 = np.nan
        self.theo0 = self.delta0 = self.gamma0 = self.vega0 = np.zeros(n)
        self.half_obs = np.full(n, TICK)
        self.half_base = np.full(n, TICK)
        self.gamma_mat = np.zeros((n, n))
        self.skew = np.zeros(n)
        self.as_mark = np.zeros(n)
        self.as_count = np.zeros(n)
        self.as_global, self.as_global_count = 0.0, 0
        self.fade_signal, self.fade_t = 0.0, 0
        self.fade_decay = np.log(2.0) / max(params.flow_fade_halflife_s, 1e-9)
        groups: dict = {}
        for j, expiry in enumerate(book.expiries):
            groups.setdefault(expiry, []).append(j)
        self._groups = [(surface.smile_for(e), np.array(ix)) for e, ix in groups.items()]

    # ------------------------------------------------------------------
    # Vol modele et fit a l'ecran
    # ------------------------------------------------------------------
    def model_vols(self, S: float, T: np.ndarray) -> np.ndarray:
        atm = np.maximum(self.surface.atm_term(T) * self.level_scale + self.book.beta * self.x_hat, 0.05)
        forward = S * np.exp(self.surface.rate * T)
        z = np.log(self.book.strikes / forward) / (atm * np.sqrt(T))
        ratios = np.empty_like(z)
        for smile, ix in self._groups:
            ratios[ix] = smile.vol_ratio(z[ix])
        return np.maximum(atm * ratios + self.u_hat, 0.05)

    def _price(self, S: float, T: np.ndarray):
        sigma = self.model_vols(S, T)
        args = (S, self.book.strikes, T, self.surface.rate, sigma, self.book.is_call)
        return bs_price(*args), bs_greeks(*args)

    def _fit_screen(self, S: float, T: np.ndarray, comp_mid: np.ndarray) -> None:
        theo, greeks = self._price(S, T)
        vega = np.maximum(greeks["vega"], 1e-6)
        resid = np.clip((comp_mid - theo) / vega, -MAX_RESIDUAL_VOL, MAX_RESIDUAL_VOL)
        w = (vega / np.maximum(self.half_obs, TICK / 2)) ** 2
        beta = self.book.beta
        factor = float(np.sum(w * beta * resid) / np.sum(w * beta * beta))
        self.x_hat += self.p.fit_factor_alpha * factor
        self.u_hat = np.clip(self.u_hat + self.p.fit_idio_alpha * (resid - beta * factor),
                             -MAX_RESIDUAL_VOL, MAX_RESIDUAL_VOL)

    def reprice(self, t: int, S: float, T: np.ndarray, comp_bid: np.ndarray,
                comp_ask: np.ndarray, q: np.ndarray) -> None:
        self.half_obs = np.maximum(0.5 * (comp_ask - comp_bid), TICK / 2)
        self._fit_screen(S, T, 0.5 * (comp_bid + comp_ask))
        theo, g = self._price(S, T)
        self.S0, self.theo0 = S, theo
        self.delta0, self.gamma0, self.vega0 = g["delta"], g["gamma"], g["vega"]
        if t % self.p.risk_matrix_every_s < self.p.reprice_every_s:
            self._update_risk_matrix(q)
        self.half_base = self._static_half_spread()

    # ------------------------------------------------------------------
    # GLFT : spread statique + matrice de risque
    # ------------------------------------------------------------------
    def _k_contract(self) -> np.ndarray:
        return self.p.kappa / (self.half_obs * CONTRACT_SIZE)

    def _static_half_spread(self) -> np.ndarray:
        if self.p.spread_model == "fixed":
            half = self.half_obs * self.p.fixed_halfspread_mult
        else:
            clip = np.full(self.book.n, float(self.p.clip_contracts))
            c1 = glft_c1(self.p.gamma, self._k_contract(), clip)
            half = (c1 + 0.5 * clip * np.diag(self.gamma_mat)) / CONTRACT_SIZE
        return half + self.p.min_edge_ticks * TICK

    def _update_risk_matrix(self, q: np.ndarray) -> None:
        if self.p.skew_model == "none":
            self.gamma_mat = np.zeros((self.book.n, self.book.n))
        else:
            per_s = 1.0 / np.sqrt(SESSION_SECONDS)
            sigma = factor_covariance(self.vega0 * CONTRACT_SIZE, self.book.beta,
                                      self.p.factor_vol_daily * per_s, self.p.idio_vol_daily * per_s,
                                      delta_contract=self.delta0 * CONTRACT_SIZE,
                                      spot_vol_per_s=self.S0 * self.spot_vol_ann * per_s / np.sqrt(BUSINESS_DAYS_YEAR),
                                      delta_weight=self.p.delta_risk_weight,
                                      independent=self.p.skew_model == "independent")
            k = self._k_contract()
            A = self.book.lambda_per_s * np.exp(self.p.kappa)
            clip = np.full(self.book.n, float(self.p.clip_contracts))
            self.gamma_mat = riccati_gamma(sigma, A, k, clip, self.p.gamma)
        self.skew = self.gamma_mat @ q / CONTRACT_SIZE

    # ------------------------------------------------------------------
    # Quotes
    # ------------------------------------------------------------------
    def fade_shift(self, t: int) -> float:
        return self.p.flow_fade * self.fade_signal * np.exp(-self.fade_decay * (t - self.fade_t))

    def theo(self, idx, t: int, S: float) -> np.ndarray:
        dS = S - self.S0 + self.fade_shift(t)
        return self.theo0[idx] + self.delta0[idx] * dS + 0.5 * self.gamma0[idx] * dS * dS

    def as_widening(self, idx) -> np.ndarray:
        if self.p.as_widening <= 0:
            return 0.0
        prior = self.p.as_prior_weight
        shrunk = ((self.as_count[idx] * self.as_mark[idx] + prior * self.as_global)
                  / (self.as_count[idx] + prior))
        return self.p.as_widening * np.maximum(shrunk, 0.0) / CONTRACT_SIZE

    def quote(self, idx, t: int, S: float, comp_bid, comp_ask, q, vega_port: float, killed: bool):
        """Quotes pour les options idx. Renvoie (bid, ask, bid_ok, ask_ok)."""
        theo = self.theo(idx, t, S)
        half = self.half_base[idx] + self.as_widening(idx)
        edge = self.p.min_edge_ticks * TICK
        bid = np.minimum(theo - half - self.skew[idx], theo - edge)
        ask = np.maximum(theo + half - self.skew[idx], theo + edge)
        bid = np.minimum(np.floor(bid / TICK + 1e-9) * TICK, comp_ask - TICK)
        ask = np.maximum(np.ceil(ask / TICK - 1e-9) * TICK, comp_bid + TICK)
        if self.p.queue_aware:
            bid = np.where(bid > comp_bid, np.minimum(bid, comp_bid + TICK), bid)
            ask = np.where(ask < comp_ask, np.maximum(ask, comp_ask - TICK), ask)
        clip, qmax = self.p.clip_contracts, self.risk.max_contracts_per_option
        q_i = q[idx]
        bid_ok = (bid >= TICK) & (q_i + clip <= qmax) & (vega_port < self.risk.max_vega_brl_per_volpt)
        ask_ok = (ask > bid) & (q_i - clip >= -qmax) & (vega_port > -self.risk.max_vega_brl_per_volpt)
        if killed:
            bid_ok, ask_ok = bid_ok & (q_i < 0), ask_ok & (q_i > 0)
        return bid, ask, bid_ok, ask_ok

    # ------------------------------------------------------------------
    # Retours d'execution
    # ------------------------------------------------------------------
    def on_fill(self, t: int, i: int, we_side: int, size: int) -> None:
        self.skew = self.skew + self.gamma_mat[:, i] * (we_side * size) / CONTRACT_SIZE
        decayed = self.fade_signal * np.exp(-self.fade_decay * (t - self.fade_t))
        self.fade_signal = decayed - we_side * self.delta0[i] * size
        self.fade_t = t

    def on_markout(self, i: int, decay_per_contract: float) -> None:
        """decay = edge a l'execution - markout couvert (R$ par contrat, > 0 = toxique)."""
        a = self.p.as_ewma_alpha
        first = self.as_count[i] == 0
        self.as_mark[i] = decay_per_contract if first else (1 - a) * self.as_mark[i] + a * decay_per_contract
        self.as_count[i] += 1
        self.as_global = (decay_per_contract if self.as_global_count == 0
                          else (1 - a) * self.as_global + a * decay_per_contract)
        self.as_global_count += 1
