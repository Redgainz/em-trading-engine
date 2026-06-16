"""
portfolio.py
============
Classes Position et Portfolio pour le simulateur de trading.

Position : un actif (spot, option, ou bond) avec quantite signee et prix d'entree
Portfolio : collection de positions + cash, calcule P&L et risk metrics.
"""
from __future__ import annotations
import dataclasses
import numpy as np
import pandas as pd
from typing import Optional

from .quant_engine import (bs_call, bs_put, bs_greeks,
                            historical_var, historical_cvar)


# ============================================================
# Position dataclass
# ============================================================
@dataclasses.dataclass
class Position:
    instrument: str           # ticker (e.g. "BRL/USD" ou "BRL_CALL_1M_ATM")
    asset_class: str          # "FX", "EQUITY", "COMMO", "BOND", "OPTION_FX"
    quantity: float           # > 0 = long, < 0 = short
    entry_price: float
    entry_date: pd.Timestamp
    # Pour options uniquement :
    strike: Optional[float] = None
    maturity_years_remaining: Optional[float] = None
    underlying_ticker: Optional[str] = None
    option_kind: Optional[str] = None  # "call" or "put"

    @property
    def notional(self) -> float:
        return abs(self.quantity * self.entry_price)


# ============================================================
# Portfolio
# ============================================================
class Portfolio:
    """
    Portefeuille de positions + cash.

    Capacite :
      - acheter / vendre n'importe quel instrument (spot, option, bond)
      - mark-to-market a tout moment avec un dict de prix de marche
      - agreger Greeks au niveau portfolio (delta, gamma, vega, theta)
      - calculer VaR / CVaR a partir du P&L historique
      - tracer P&L cumule
    """

    def __init__(self, initial_cash: float = 1_000_000.0, ccy: str = "USD"):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.ccy = ccy
        self.positions: list[Position] = []
        self.closed_positions: list[Position] = []
        self.transaction_history: list[dict] = []
        self.pnl_history: list[dict] = []  # {date, cash, mtm, total_value, pnl}

    # --------------------------------------------------------
    # Trading actions
    # --------------------------------------------------------
    def buy(self, instrument: str, asset_class: str, quantity: float,
             current_price: float, date: pd.Timestamp,
             strike: Optional[float] = None,
             maturity_years: Optional[float] = None,
             option_kind: Optional[str] = None,
             underlying_ticker: Optional[str] = None,
             cost_bps: float = 5.0):
        """Achete `quantity` units de l'instrument a `current_price` (+ cout bps)."""
        cost = quantity * current_price * (1 + cost_bps / 10000)
        if cost > self.cash:
            raise ValueError(f"Insufficient cash: need {cost:.2f}, have {self.cash:.2f}")
        self.cash -= cost
        pos = Position(
            instrument=instrument, asset_class=asset_class,
            quantity=quantity, entry_price=current_price, entry_date=date,
            strike=strike, maturity_years_remaining=maturity_years,
            option_kind=option_kind, underlying_ticker=underlying_ticker,
        )
        self.positions.append(pos)
        self.transaction_history.append(dict(
            date=date, action="BUY", instrument=instrument,
            quantity=quantity, price=current_price, cost=cost, cash_after=self.cash,
        ))

    def sell(self, instrument: str, asset_class: str, quantity: float,
              current_price: float, date: pd.Timestamp,
              strike: Optional[float] = None,
              maturity_years: Optional[float] = None,
              option_kind: Optional[str] = None,
              underlying_ticker: Optional[str] = None,
              cost_bps: float = 5.0):
        """Short ou close. Signed quantity > 0 = on vend quantity unites."""
        # Recherche position existante a fermer
        proceeds = quantity * current_price * (1 - cost_bps / 10000)
        self.cash += proceeds
        # Ajoute une position courte equivalente
        pos = Position(
            instrument=instrument, asset_class=asset_class,
            quantity=-quantity, entry_price=current_price, entry_date=date,
            strike=strike, maturity_years_remaining=maturity_years,
            option_kind=option_kind, underlying_ticker=underlying_ticker,
        )
        self.positions.append(pos)
        self.transaction_history.append(dict(
            date=date, action="SELL", instrument=instrument,
            quantity=quantity, price=current_price, proceeds=proceeds, cash_after=self.cash,
        ))

    def age_positions(self, dt_years: float, current_prices: dict,
                       date: pd.Timestamp):
        """Decremente la maturite des options. Si expiration, settle a l'intrinseque."""
        survivors = []
        for pos in self.positions:
            if pos.asset_class == "OPTION_FX" and pos.maturity_years_remaining is not None:
                pos.maturity_years_remaining -= dt_years
                if pos.maturity_years_remaining <= 0:
                    # Settle a l'intrinseque
                    S = current_prices.get(pos.underlying_ticker, pos.strike)
                    if pos.option_kind == "call":
                        intrinsic = max(S - pos.strike, 0.0)
                    else:
                        intrinsic = max(pos.strike - S, 0.0)
                    self.cash += pos.quantity * intrinsic
                    self.transaction_history.append(dict(
                        date=date, action="EXPIRE", instrument=pos.instrument,
                        quantity=-pos.quantity, price=intrinsic, cash_after=self.cash,
                    ))
                    self.closed_positions.append(pos)
                    continue
            survivors.append(pos)
        self.positions = survivors

    def close_all(self, current_prices: dict, date: pd.Timestamp, cost_bps: float = 5.0):
        """Close toutes les positions au prix courant."""
        for pos in self.positions:
            if pos.instrument in current_prices:
                price = current_prices[pos.instrument]
                proceeds = pos.quantity * price * (1 - cost_bps / 10000 if pos.quantity > 0 else 1 + cost_bps / 10000)
                self.cash += proceeds
                self.transaction_history.append(dict(
                    date=date, action="CLOSE", instrument=pos.instrument,
                    quantity=-pos.quantity, price=price, cash_after=self.cash,
                ))
        self.closed_positions.extend(self.positions)
        self.positions = []

    # --------------------------------------------------------
    # Mark-to-market
    # --------------------------------------------------------
    def mark_to_market(self, market_state: dict, date: pd.Timestamp) -> dict:
        """
        market_state : dict avec
            'spot' : dict(instrument -> price)
            'iv'   : dict(instrument -> implied vol)  (pour options)
            'r_d'  : taux domestique (USD)
            'r_f'  : dict(currency -> taux foreign)

        Renvoie dict total_value, mtm_positions, pnl_unrealized.
        """
        spot = market_state["spot"]
        iv   = market_state.get("iv", {})
        r_d  = market_state.get("r_d", 0.02)
        r_f  = market_state.get("r_f", {})

        mtm = 0.0
        per_position = []
        for pos in self.positions:
            cur_price = self._compute_position_price(pos, spot, iv, r_d, r_f)
            position_mtm = pos.quantity * cur_price
            pnl = pos.quantity * (cur_price - pos.entry_price)
            mtm += position_mtm
            per_position.append({
                "instrument": pos.instrument,
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "current_price": cur_price,
                "mtm": position_mtm,
                "pnl_unrealized": pnl,
            })
        total = self.cash + mtm
        return {
            "cash": self.cash, "mtm_positions": mtm,
            "total_value": total, "pnl_total": total - self.initial_cash,
            "per_position": per_position, "date": date,
        }

    def _compute_position_price(self, pos: Position, spot: dict, iv: dict,
                                 r_d: float, r_f: dict) -> float:
        """Prix theorique de la position selon son asset_class."""
        if pos.asset_class in ("FX", "EQUITY", "COMMO"):
            return spot.get(pos.instrument, pos.entry_price)
        if pos.asset_class == "BOND":
            return spot.get(pos.instrument, pos.entry_price)
        if pos.asset_class == "OPTION_FX":
            # Reprice avec BS Garman-Kohlhagen
            S = spot.get(pos.underlying_ticker, pos.strike)
            K = pos.strike
            T = max(pos.maturity_years_remaining, 1e-6)
            sigma = iv.get(pos.instrument, 0.15)
            ccy_foreign = pos.underlying_ticker.split("/")[0] if pos.underlying_ticker else "USD"
            r_for = r_f.get(ccy_foreign, 0.0)
            if pos.option_kind == "call":
                return bs_call(S, K, T, r_d, r_for, sigma)
            return bs_put(S, K, T, r_d, r_for, sigma)
        return pos.entry_price

    # --------------------------------------------------------
    # Agreger Greeks
    # --------------------------------------------------------
    def aggregate_greeks(self, market_state: dict) -> dict:
        """Somme les Greeks sur toutes les positions options + delta des spots."""
        spot = market_state["spot"]; iv = market_state.get("iv", {})
        r_d = market_state.get("r_d", 0.02); r_f = market_state.get("r_f", {})

        agg = dict(delta_usd=0.0, gamma_usd=0.0, vega_usd=0.0, theta_usd=0.0)
        for pos in self.positions:
            if pos.asset_class in ("FX", "EQUITY", "COMMO"):
                # delta = 1 par unite, scale par spot
                agg["delta_usd"] += pos.quantity * spot.get(pos.instrument, pos.entry_price)
            elif pos.asset_class == "BOND":
                # Approx : delta ~ quantite * prix; DV01 traite separement
                agg["delta_usd"] += pos.quantity * spot.get(pos.instrument, pos.entry_price)
            elif pos.asset_class == "OPTION_FX":
                S = spot.get(pos.underlying_ticker, pos.strike)
                T = max(pos.maturity_years_remaining, 1e-6)
                sigma = iv.get(pos.instrument, 0.15)
                ccy_for = pos.underlying_ticker.split("/")[0] if pos.underlying_ticker else "USD"
                r_for = r_f.get(ccy_for, 0.0)
                g = bs_greeks(S, pos.strike, T, r_d, r_for, sigma, pos.option_kind)
                agg["delta_usd"] += pos.quantity * g["delta"] * S
                agg["gamma_usd"] += pos.quantity * g["gamma"] * (S ** 2) / 100  # gamma USD pour 1% spot move
                agg["vega_usd"] += pos.quantity * g["vega"]                      # vega pour 1% vol
                agg["theta_usd"] += pos.quantity * g["theta"]                    # theta per day
        return agg

    # --------------------------------------------------------
    # Risk metrics : VaR / CVaR sur l'historique de P&L
    # --------------------------------------------------------
    def compute_var_cvar(self, level: float = 0.95) -> tuple[float, float]:
        if len(self.pnl_history) < 5:
            return 0.0, 0.0
        rets = np.diff([h["total_value"] for h in self.pnl_history])
        return historical_var(rets, level), historical_cvar(rets, level)

    # --------------------------------------------------------
    # Snapshot / state pour Streamlit session_state
    # --------------------------------------------------------
    def snapshot(self, market_state: dict, date: pd.Timestamp) -> dict:
        mtm = self.mark_to_market(market_state, date)
        self.pnl_history.append({
            "date": date,
            "cash": mtm["cash"],
            "mtm_positions": mtm["mtm_positions"],
            "total_value": mtm["total_value"],
            "pnl_total": mtm["pnl_total"],
            "n_positions": len(self.positions),
        })
        return mtm


# Smoke test
if __name__ == "__main__":
    pf = Portfolio(initial_cash=1_000_000)
    pf.buy("BRL/USD", "FX", quantity=200_000, current_price=0.25,
            date=pd.Timestamp("2024-01-01"))
    pf.buy("SPY", "EQUITY", quantity=1000, current_price=480.0,
            date=pd.Timestamp("2024-01-01"))
    pf.buy("BRL_CALL_3M_ATM", "OPTION_FX", quantity=10_000, current_price=0.012,
            date=pd.Timestamp("2024-01-01"),
            strike=0.25, maturity_years=0.25, option_kind="call",
            underlying_ticker="BRL/USD")
    print(f"Cash apres achats : {pf.cash:.2f}")
    print(f"Positions : {len(pf.positions)}")
    mkt = {"spot": {"BRL/USD": 0.27, "SPY": 510.0}, "iv": {"BRL_CALL_3M_ATM": 0.18},
           "r_d": 0.04, "r_f": {"BRL": 0.10}}
    mtm = pf.mark_to_market(mkt, pd.Timestamp("2024-02-01"))
    print(f"Total value : {mtm['total_value']:.2f}  P&L : {mtm['pnl_total']:+.2f}")
    g = pf.aggregate_greeks(mkt)
    print(f"Greeks aggregated : delta={g['delta_usd']:.0f}  gamma={g['gamma_usd']:.0f}")
    print(f"                    vega={g['vega_usd']:.0f}  theta={g['theta_usd']:.0f}")
