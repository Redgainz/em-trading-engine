"""
market_state.py
===============
Simulateur du marche mois par mois pour le Trading Simulator.

Strategie :
  - Le marche est calibre sur les donnees synthetiques 2010-2023
    (fichier paper-rmt-zeta/data/returns.csv) qui inclut deja
    regimes Markov, jumps en queue, structure factorielle.
  - Pour la simulation 2024-2025, on EXTRAPOLE en continuant a tirer
    des rendements selon la chaine Markov, avec evolution du regime.
  - A chaque mois, on calcule :
      - spot evolue (prix au prochain mois)
      - vol implicite SABR pour chaque devise (avec smile)
      - courbes Nelson-Siegel pour rates
      - regimes Markov (probas)

Univers d'instruments :
  - 3 FX EM spot : BRL/USD, ZAR/USD, TRY/USD
  - 6 options FX ATM (calls + puts, 1M et 3M, sur BRL et ZAR et TRY)
  - 3 actions US : SPY, XLE, XLF
  - 2 commodities : WTI, GOLD
  - 2 bonds : US 10Y (TLT-like), EMBI (EM bonds)
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from datetime import datetime

from .quant_engine import (sabr_implied_vol, garch_forecast, ewma_vol,
                            markov_regimes, nelson_siegel_curve, var_p_fit,
                            var_p_forecast, bs_call, bs_put)


# ============================================================
# UNIVERS DES INSTRUMENTS
# ============================================================
# Format : (ticker, asset_class, label, initial_spot)
INSTRUMENT_UNIVERSE = [
    # FX EM spot (en USD per unit of foreign currency)
    ("BRL/USD", "FX",       "Real bresilien (BRL)",    0.20),
    ("ZAR/USD", "FX",       "Rand sud-africain (ZAR)", 0.054),
    ("TRY/USD", "FX",       "Lire turque (TRY)",       0.033),
    # Equities US
    ("SPY",     "EQUITY",   "S&P 500 ETF (SPY)",      450.0),
    ("XLE",     "EQUITY",   "Energy Select Sector",     85.0),
    ("XLF",     "EQUITY",   "Financial Select Sector",  35.0),
    # Commodities
    ("WTI",     "COMMO",    "WTI Crude Oil",            72.0),
    ("GOLD",    "COMMO",    "Gold (oz)",              2050.0),
    # Bonds
    ("UST10Y",  "BOND",     "US 10Y Treasury (price)", 95.0),
    ("EMBI",    "BOND",     "EM Bonds Index (EMB)",   85.0),
    # FX Options (BRL)
    ("BRL_CALL_1M_ATM", "OPTION_FX", "BRL/USD Call ATM 1M", None),
    ("BRL_PUT_1M_ATM",  "OPTION_FX", "BRL/USD Put ATM 1M",  None),
    ("BRL_CALL_3M_ATM", "OPTION_FX", "BRL/USD Call ATM 3M", None),
    ("BRL_PUT_3M_ATM",  "OPTION_FX", "BRL/USD Put ATM 3M",  None),
    # FX Options (ZAR)
    ("ZAR_CALL_1M_ATM", "OPTION_FX", "ZAR/USD Call ATM 1M", None),
    ("ZAR_PUT_1M_ATM",  "OPTION_FX", "ZAR/USD Put ATM 1M",  None),
    # FX Options (TRY)
    ("TRY_CALL_1M_ATM", "OPTION_FX", "TRY/USD Call ATM 1M", None),
    ("TRY_PUT_1M_ATM",  "OPTION_FX", "TRY/USD Put ATM 1M",  None),
]

# Map options -> underlying currency pair
OPTION_TO_UNDERLYING = {
    "BRL_CALL_1M_ATM": ("BRL/USD", "call", 1/12),
    "BRL_PUT_1M_ATM":  ("BRL/USD", "put",  1/12),
    "BRL_CALL_3M_ATM": ("BRL/USD", "call", 0.25),
    "BRL_PUT_3M_ATM":  ("BRL/USD", "put",  0.25),
    "ZAR_CALL_1M_ATM": ("ZAR/USD", "call", 1/12),
    "ZAR_PUT_1M_ATM":  ("ZAR/USD", "put",  1/12),
    "TRY_CALL_1M_ATM": ("TRY/USD", "call", 1/12),
    "TRY_PUT_1M_ATM":  ("TRY/USD", "put",  1/12),
}

# Vol annualisee de base et drift par instrument (sert si pas de training data)
VOL_BASE = {
    "BRL/USD": 0.22, "ZAR/USD": 0.18, "TRY/USD": 0.30,
    "SPY": 0.18, "XLE": 0.32, "XLF": 0.26,
    "WTI": 0.40, "GOLD": 0.16,
    "UST10Y": 0.08, "EMBI": 0.10,
}
DRIFT = {k: 0.0 for k in VOL_BASE}  # Drift nul par defaut

# Taux interets foreign par devise
FOREIGN_RATES = {
    "BRL": 0.11, "ZAR": 0.08, "TRY": 0.45,  # taux EM realistes
    "USD": 0.045,  # taux US
}

# Parametres SABR (alpha, beta, rho, nu) par devise — calibres avec valeurs realistes
SABR_PARAMS = {
    "BRL/USD": dict(alpha=0.22, beta=1.0, rho=-0.15, nu=0.55),
    "ZAR/USD": dict(alpha=0.18, beta=1.0, rho=-0.10, nu=0.50),
    "TRY/USD": dict(alpha=0.32, beta=1.0, rho=-0.20, nu=0.70),
}


# ============================================================
# MARKET SIMULATOR
# ============================================================
class MarketSimulator:
    """
    Genere l'evolution mensuelle du marche entre `start_date` et `end_date`.

    Methodologie :
      1. Charge le training set (returns.csv 2010-2023) si disponible
      2. Calibre Markov regimes sur ces returns
      3. Pour chaque mois 2024-..., simule un mois de business days (21 jours)
      4. Met a jour spots, vols, taux, regimes
    """

    def __init__(self,
                 data_dir: str | None = None,
                 start_date: str = "2024-01-01",
                 end_date: str = "2025-12-31",
                 seed: int = 42):
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.rng = np.random.default_rng(seed)

        # Try to load training data
        self.training_returns = None
        if data_dir:
            try:
                df = pd.read_csv(os.path.join(data_dir, "returns.csv"),
                                  index_col=0, parse_dates=True)
                self.training_returns = df
            except Exception:
                pass

        # Calibrate Markov sur l'indice equally-weighted
        if self.training_returns is not None:
            idx_returns = self.training_returns.mean(axis=1).values
            self.markov_fit = markov_regimes(idx_returns, n_states=3)
            self.current_regime_idx = int(np.argmax(self.markov_fit["current_regime_probs"]))
            # Cap les sigmas a une valeur realiste (sigma annualisee max 35%)
            sigmas_ann = self.markov_fit["sigmas"] * np.sqrt(252)
            sigmas_ann = np.clip(sigmas_ann, 0.05, 0.35)
            self.markov_fit["sigmas"] = sigmas_ann / np.sqrt(252)
        else:
            # Fallback : random init
            self.markov_fit = dict(
                means=np.array([0.0, 0.0, 0.0]),
                sigmas=np.array([0.005, 0.012, 0.030]),
                transition=np.array([[0.95, 0.04, 0.01],
                                      [0.04, 0.92, 0.04],
                                      [0.02, 0.10, 0.88]]),
                gamma=np.zeros((1, 3)),
                current_regime_probs=np.array([0.6, 0.3, 0.1]),
            )
            self.current_regime_idx = 1

        # State initial
        self.current_date = self.start_date
        self.spot = {t: s for t, ac, lbl, s in INSTRUMENT_UNIVERSE if s is not None}
        # Initialiser quelques historiques (252 jours synthetiques)
        self.history = {t: [s for _ in range(252)] for t, s in self.spot.items()}

    # --------------------------------------------------------
    # Simulation d'un mois (21 jours business)
    # --------------------------------------------------------
    def step_one_month(self) -> dict:
        """
        Avance d'un mois business (21 jours).
        Renvoie le nouvel etat marche.
        """
        n_days = 21
        for d in range(n_days):
            # Tirage de l'etat suivant Markov
            P = self.markov_fit["transition"]
            self.current_regime_idx = int(
                self.rng.choice(3, p=P[self.current_regime_idx])
            )
            sigma_state = self.markov_fit["sigmas"][self.current_regime_idx]
            mu_state = self.markov_fit["means"][self.current_regime_idx]

            # Common shock : scale moderee, df=8 plutot que 5 pour moins de queues
            common_shock = self.rng.standard_t(df=8) * sigma_state * 0.25 + mu_state

            # Appliquer aux instruments selon leur vol de base
            for ticker, asset_class, label, _ in INSTRUMENT_UNIVERSE:
                if asset_class == "OPTION_FX":
                    continue
                vol_ann = VOL_BASE.get(ticker, 0.20)
                vol_daily = vol_ann / np.sqrt(252)
                # Beta vs common market
                if asset_class == "FX":
                    beta_market = 0.20
                elif asset_class == "EQUITY":
                    beta_market = 0.30
                elif asset_class == "COMMO":
                    beta_market = 0.15
                elif asset_class == "BOND":
                    beta_market = -0.10
                else:
                    beta_market = 0.15
                # Idio dominant, scale par vol cible avec petit drift positif (a la longue)
                idio = self.rng.standard_t(df=8) * vol_daily * 0.85
                drift_daily = 0.0001 if asset_class in ("EQUITY", "COMMO") else 0.0
                ret = drift_daily + beta_market * common_shock + idio
                # Cap returns extremes a +/- 6% par jour
                ret = np.clip(ret, -0.06, 0.06)
                if asset_class == "FX":
                    self.spot[ticker] *= np.exp(ret)
                else:
                    self.spot[ticker] *= (1 + ret)

                self.history[ticker].append(self.spot[ticker])
                if len(self.history[ticker]) > 504:
                    self.history[ticker].pop(0)

        self.current_date = self.current_date + pd.DateOffset(months=1)
        return self.get_market_state()

    # --------------------------------------------------------
    # Etat marche complet
    # --------------------------------------------------------
    def get_market_state(self) -> dict:
        """Renvoie l'etat marche pour mark-to-market et analytics."""
        # Vol implicite SABR ATM pour chaque option
        iv = {}
        option_prices = {}
        r_d = FOREIGN_RATES["USD"]
        r_f_dict = {ccy: FOREIGN_RATES[ccy] for ccy in ["BRL", "ZAR", "TRY"]}

        for opt_ticker, (underlying, kind, T) in OPTION_TO_UNDERLYING.items():
            F_currency = underlying.split("/")[0]
            params = SABR_PARAMS.get(underlying, dict(alpha=0.20, beta=1.0, rho=-0.1, nu=0.5))
            S = self.spot[underlying]
            # Forward = S * exp((r_d - r_f) * T)
            r_f = FOREIGN_RATES[F_currency]
            F = S * np.exp((r_d - r_f) * T)
            K = F  # ATM
            implied = sabr_implied_vol(F, K, T, **params)
            iv[opt_ticker] = max(implied, 0.05)
            if kind == "call":
                option_prices[opt_ticker] = bs_call(S, K, T, r_d, r_f, iv[opt_ticker])
            else:
                option_prices[opt_ticker] = bs_put(S, K, T, r_d, r_f, iv[opt_ticker])

        return dict(
            date=self.current_date,
            spot={**self.spot, **option_prices},
            iv=iv,
            r_d=r_d,
            r_f=r_f_dict,
            regime_idx=self.current_regime_idx,
            regime_probs=self._compute_current_regime_probs(),
            sabr_params=SABR_PARAMS,
            vol_base=VOL_BASE,
        )

    def _compute_current_regime_probs(self) -> np.ndarray:
        """Probas regime actuelles : one-hot sur le regime courant + smoothing transitions."""
        P = self.markov_fit["transition"]
        # Start with one-hot on current regime
        post = np.zeros(3); post[self.current_regime_idx] = 1.0
        # Apply 1 transition step to smoothen
        post = P[self.current_regime_idx]
        return post / post.sum()

    # --------------------------------------------------------
    # Analytics computees a la demande sur un instrument
    # --------------------------------------------------------
    def analytics_for(self, ticker: str) -> dict:
        """Renvoie GARCH, EWMA, regression, vol surface SABR pour cet instrument."""
        hist = np.array(self.history.get(ticker, []))
        if len(hist) < 30:
            return {}
        log_ret = np.diff(np.log(hist))
        out = {
            "vol_realized_252d": float(log_ret[-252:].std() * np.sqrt(252)),
            "vol_realized_21d":  float(log_ret[-21:].std() * np.sqrt(252)),
            "vol_ewma":          float(ewma_vol(log_ret[-120:]) * np.sqrt(252)),
            "garch":             garch_forecast(log_ret[-252:]),
        }
        # Pour FX EM : ajouter SABR surface
        if ticker in SABR_PARAMS:
            params = SABR_PARAMS[ticker]
            T = 0.25
            S = self.spot[ticker]
            F = S * np.exp((FOREIGN_RATES["USD"] - FOREIGN_RATES[ticker.split("/")[0]]) * T)
            moneyness = np.linspace(0.85, 1.15, 13)
            vols = np.array([sabr_implied_vol(F, F * m, T, **params) for m in moneyness])
            out["sabr_surface"] = dict(moneyness=moneyness, vols=vols, F=F, T=T)
        return out


# Smoke test
if __name__ == "__main__":
    sim = MarketSimulator(seed=0)
    print(f"Date initiale : {sim.current_date.date()}")
    print(f"Regime initial : {sim.current_regime_idx}")
    print(f"Sigmas regimes : {sim.markov_fit['sigmas']*np.sqrt(252)*100}")
    print(f"BRL initial : {sim.spot['BRL/USD']:.4f}")

    for m in range(3):
        state = sim.step_one_month()
        print(f"\nApres mois {m+1} ({state['date'].date()}):")
        print(f"  BRL = {state['spot']['BRL/USD']:.4f}  ZAR = {state['spot']['ZAR/USD']:.4f}")
        print(f"  SPY = {state['spot']['SPY']:.2f}  Gold = {state['spot']['GOLD']:.2f}")
        print(f"  BRL Call 1M ATM = {state['spot']['BRL_CALL_1M_ATM']:.4f}")
        print(f"  IV BRL 1M ATM = {state['iv']['BRL_CALL_1M_ATM']*100:.1f}%")
        print(f"  Regime probs : {state['regime_probs'].round(2)}")

    print("\nAnalytics BRL :")
    a = sim.analytics_for("BRL/USD")
    print(f"  Vol realisee 252j : {a['vol_realized_252d']*100:.1f}%")
    print(f"  Vol realisee 21j  : {a['vol_realized_21d']*100:.1f}%")
    print(f"  Vol EWMA         : {a['vol_ewma']*100:.1f}%")
    print(f"  GARCH forecast   : {a['garch']['sigma_t1']*np.sqrt(252)*100:.1f}%")
