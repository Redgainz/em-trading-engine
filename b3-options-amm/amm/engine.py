"""
engine.py
=========
Backtest evenementiel a la seconde d'un book de market-making options B3.

Boucle (t = 0 .. 24 899 s) :
  1. reprice (fit ecran + greeks + GLFT) toutes les `reprice_every_s`
  2. maturation des markouts -> boucle d'adverse selection du quoter
  3. traitement des ordres agresseurs arrives a t (bruit amincis / informes)
     - priorite prix-temps simplifiee : meilleur strict -> rempli ; a egalite -> part de file
  4. couverture delta en actions (bande)
  5. echantillonnage KPI toutes les 10 s, kill-switch sur perte intraday
Fin de seance : liquidation du book au prix concurrent (on traverse le spread)
-> PnL "flat EOD" conservateur.
"""
from __future__ import annotations

import datetime as dt
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .conventions import CONTRACT_SIZE, OPTION_FEE_RATE, STOCK_HALF_SPREAD, TICK
from .hedger import execution_cost, hedge_order
from .market import DayMarket
from .params import Config, FlowParams
from .setup_day import load_ticker_context
from .strategy import OptionQuoter

SAMPLE_EVERY_S = 10
FILL_COLUMNS = ["t", "i", "we_side", "price", "size", "theo", "fair", "mid", "delta", "S", "informed"]


@dataclass(frozen=True)
class DayResult:
    ticker: str
    day: dt.date
    strategy: str
    fills: pd.DataFrame
    hedges: pd.DataFrame
    samples: pd.DataFrame
    sample_q: np.ndarray            # (n_samples, N) inventaire en contrats
    totals: dict


class _Book:
    """Comptabilite du book (etat mutable local a une seance)."""

    def __init__(self, n_opt: int) -> None:
        self.q = np.zeros(n_opt)
        self.stock = 0.0
        self.cash = 0.0
        self.option_fees = 0.0
        self.stock_fees = 0.0
        self.hedge_spread_cost = 0.0
        self.delta_ref = 0.0      # 100 * sum q delta0 (au spot de reference du quoter)
        self.gamma_ref = 0.0
        self.vega = 0.0           # R$ par point de vol
        self.killed = False

    def refresh_greeks(self, quoter: OptionQuoter) -> None:
        self.delta_ref = CONTRACT_SIZE * float(self.q @ quoter.delta0)
        self.gamma_ref = CONTRACT_SIZE * float(self.q @ quoter.gamma0)
        self.vega = float(self.q @ quoter.vega0)

    def delta_shares(self, S: float, S0: float) -> float:
        return self.delta_ref + self.gamma_ref * (S - S0) + self.stock


def _participation(we_price, comp_price, size, clip, u_queue, queue_share, improving) -> int:
    if improving:
        return int(min(size, clip))
    if abs(we_price - comp_price) < 1e-9 and u_queue < queue_share:
        return int(min(size, clip))
    return 0


def _match(m: DayMarket, e: int, t: int, quoter: OptionQuoter, bk: _Book, flow: FlowParams, clip: int):
    """Traite l'ordre agresseur e. Renvoie (i, we_side, prix, taille, informe) ou None."""
    ev = m.events
    i, side, size = int(ev.i[e]), int(ev.side[e]), int(ev.size[e])
    cb, ca, S = m.comp_bid[t, i], m.comp_ask[t, i], m.S[t]
    bid, ask, bid_ok, ask_ok = quoter.quote(i, t, S, cb, ca, bk.q, bk.vega, bk.killed)
    best_bid = max(bid, cb) if bid_ok else cb
    best_ask = min(ask, ca) if ask_ok else ca
    fair = m.fair[t, i]
    informed = side == 0
    if informed:
        future = m.fair[min(t + flow.informed_horizon_s, m.n_seconds - 1), i]
        edge = flow.informed_min_edge_ticks * TICK
        side = 1 if future - best_ask >= edge else (-1 if best_bid - future >= edge else 0)
        if side == 0:
            return None
    else:
        h = m.half[t, i]
        distance = (best_ask - fair) if side > 0 else (fair - best_bid)
        accept = min(1.0, np.exp(-flow.price_sensitivity / h * (distance - h))) / flow.max_flow_mult
        if ev.u_accept[e] >= accept:
            return None
    if side > 0 and ask_ok:
        filled = _participation(ask, ca, size, clip, ev.u_queue[e], flow.queue_share, ask < ca - 1e-9)
        return (i, -1, float(ask), filled, informed) if filled else None
    if side < 0 and bid_ok:
        filled = _participation(bid, cb, size, clip, ev.u_queue[e], flow.queue_share, bid > cb + 1e-9)
        return (i, 1, float(bid), filled, informed) if filled else None
    return None


def _book_fill(bk: _Book, quoter: OptionQuoter, m: DayMarket, t: int, fill) -> list:
    i, we_side, price, size, informed = fill
    notional = price * size * CONTRACT_SIZE
    bk.cash -= we_side * notional
    bk.option_fees += OPTION_FEE_RATE * notional
    bk.q[i] += we_side * size
    S = m.S[t]
    delta_now = quoter.delta0[i] + quoter.gamma0[i] * (S - quoter.S0)
    bk.delta_ref += we_side * size * CONTRACT_SIZE * quoter.delta0[i]
    bk.gamma_ref += we_side * size * CONTRACT_SIZE * quoter.gamma0[i]
    bk.vega += we_side * size * quoter.vega0[i]
    theo = float(quoter.theo(i, t, S))
    quoter.on_fill(t, i, we_side, size)
    return [t, i, we_side, price, size, theo, m.fair[t, i], 0.5 * (m.comp_bid[t, i] + m.comp_ask[t, i]),
            float(delta_now), S, informed]


def _hedge(bk: _Book, cfg: Config, quoter: OptionQuoter, S: float, t: int, hedges: list) -> None:
    shares = hedge_order(cfg.hedge, bk.delta_shares(S, quoter.S0), S, bk.gamma_ref)
    if shares == 0:
        return
    price, fee = execution_cost(shares, S)
    bk.stock += shares
    bk.cash -= shares * price
    bk.stock_fees += fee
    bk.hedge_spread_cost += abs(shares) * STOCK_HALF_SPREAD
    hedges.append((t, shares, price, S))


def _sample(bk: _Book, quoter: OptionQuoter, m: DayMarket, t: int, cfg: Config) -> dict:
    S = m.S[t]
    idx = np.arange(m.book.n)
    cb, ca = m.comp_bid[t], m.comp_ask[t]
    bid, ask, bid_ok, ask_ok = quoter.quote(idx, t, S, cb, ca, bk.q, bk.vega, bk.killed)
    two_sided = bid_ok & ask_ok
    fees = bk.option_fees + bk.stock_fees
    pnl_mid = bk.cash + CONTRACT_SIZE * float(bk.q @ (0.5 * (cb + ca))) + bk.stock * S - fees
    pnl_true = bk.cash + CONTRACT_SIZE * float(bk.q @ m.fair[t]) + bk.stock * S - fees
    if pnl_mid < -cfg.risk.max_daily_loss_brl:
        bk.killed = True
    return {
        "t": t, "S": S, "pnl_mid": pnl_mid, "pnl_true": pnl_true,
        "delta_shares": bk.delta_shares(S, quoter.S0), "vega_volpt": bk.vega,
        "gross_contracts": float(np.abs(bk.q).sum()), "stock": bk.stock,
        "uptime": float(two_sided.mean()),
        "spread_ticks": float(np.mean((ask - bid)[two_sided]) / TICK) if two_sided.any() else np.nan,
        "comp_spread_ticks": float(np.mean(ca - cb) / TICK),
        "at_best": float(np.mean((bid_ok & (bid >= cb)) | (ask_ok & (ask <= ca)))),
        "x_hat": quoter.x_hat, "killed": bk.killed,
    }


def _liquidate(bk: _Book, m: DayMarket) -> float:
    """Debouclage au prix concurrent en fin de seance. Renvoie le cout vs mid."""
    t = m.n_seconds - 1
    cb, ca, S = m.comp_bid[t], m.comp_ask[t], m.S[t]
    px = np.where(bk.q > 0, cb, ca)
    notional = np.abs(bk.q) * px * CONTRACT_SIZE
    cost = float(np.sum(np.abs(bk.q) * (ca - cb) / 2 * CONTRACT_SIZE))
    bk.cash += float(np.sum(bk.q * px * CONTRACT_SIZE))
    bk.option_fees += OPTION_FEE_RATE * float(notional.sum())
    stock_px, stock_fee = execution_cost(int(-bk.stock), S) if bk.stock else (S, 0.0)
    bk.cash += bk.stock * stock_px
    bk.stock_fees += stock_fee
    cost += abs(bk.stock) * STOCK_HALF_SPREAD
    bk.q = np.zeros_like(bk.q)
    bk.stock = 0.0
    return cost


def run_day(m: DayMarket, cfg: Config, flow: FlowParams) -> DayResult:
    sp = cfg.strategy
    surface = load_ticker_context(m.ticker).surface
    quoter = OptionQuoter(m.book, surface, m.level_scale, sp, cfg.risk)
    bk = _Book(m.book.n)
    fills, hedges, samples, sample_q, pending = [], [], [], [], deque()
    ev_t, e, n_ev = m.events.t, 0, len(m.events.t)
    mid = m.comp_mid

    for t in range(m.n_seconds):
        S = m.S[t]
        if t % sp.reprice_every_s == 0:
            quoter.reprice(t, S, m.T[t], m.comp_bid[t], m.comp_ask[t], bk.q)
            bk.refresh_greeks(quoter)
        while pending and pending[0][0] <= t:
            tm, i, we_side, mid_f, delta_f, S_f = pending.popleft()
            hedged_move = (mid[tm, i] - mid_f) - delta_f * (m.S[tm] - S_f)
            quoter.on_markout(i, -we_side * hedged_move * CONTRACT_SIZE)
        while e < n_ev and ev_t[e] == t:
            fill = _match(m, e, t, quoter, bk, flow, sp.clip_contracts)
            e += 1
            if fill is not None:
                row = _book_fill(bk, quoter, m, t, fill)
                fills.append(row)
                pending.append((min(t + sp.as_horizon_s, m.n_seconds - 1), row[1], row[2], row[7], row[8], S))
        _hedge(bk, cfg, quoter, S, t, hedges)
        if t % SAMPLE_EVERY_S == 0 or t == m.n_seconds - 1:
            samples.append(_sample(bk, quoter, m, t, cfg))
            sample_q.append(bk.q.copy())

    pre_liq = samples[-1]["pnl_mid"]
    liquidation_cost = _liquidate(bk, m)
    net = bk.cash - bk.option_fees - bk.stock_fees
    totals = {"net_pnl": net, "pnl_before_liquidation": pre_liq, "liquidation_cost": liquidation_cost,
              "option_fees": bk.option_fees, "stock_fees": bk.stock_fees,
              "hedge_spread_cost": bk.hedge_spread_cost, "killed": bk.killed}
    return DayResult(ticker=m.ticker, day=m.day, strategy=sp.name,
                     fills=pd.DataFrame(fills, columns=FILL_COLUMNS),
                     hedges=pd.DataFrame(hedges, columns=["t", "shares", "price", "S"]),
                     samples=pd.DataFrame(samples), sample_q=np.array(sample_q), totals=totals)
