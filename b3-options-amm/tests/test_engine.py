"""Tests d'integration du simulateur et du moteur, sur le cache de donnees reelles."""
from dataclasses import replace

import numpy as np
import pytest

from amm.engine import run_day
from amm.kpi import day_summary, fill_markouts
from amm.market import Events
from amm.params import Config, FlowParams, RiskParams, VERSIONS
from amm.setup_day import build_market, load_ticker_context

TICKER = "PETR4"
FLOW = FlowParams()


@pytest.fixture(scope="module")
def market():
    ctx = load_ticker_context(TICKER)
    return build_market(TICKER, ctx.days[len(ctx.days) // 2], FLOW)


@pytest.fixture(scope="module")
def full_result(market):
    return run_day(market, VERSIONS["v4_fade"], FLOW)


def test_market_is_deterministic_per_session():
    ctx = load_ticker_context(TICKER)
    a, b = build_market(TICKER, ctx.days[0], FLOW), build_market(TICKER, ctx.days[0], FLOW)
    np.testing.assert_array_equal(a.S, b.S)
    np.testing.assert_array_equal(a.events.t, b.events.t)


def test_market_spot_follows_real_minute_closes(market):
    ctx = load_ticker_context(TICKER)
    bars = ctx.intraday[ctx.intraday.index.date == market.day]
    assert market.S[-1] == pytest.approx(bars["Close"].iloc[-1], abs=0.01)
    assert np.all(market.comp_ask > market.comp_bid)


def test_no_flow_means_no_trades_and_zero_pnl(market):
    empty = Events(*(np.array([], dtype=int) for _ in range(4)), np.array([]), np.array([]))
    res = run_day(replace(market, events=empty), VERSIONS["v4_fade"], FLOW)
    assert res.fills.empty and res.hedges.empty
    assert res.totals["net_pnl"] == pytest.approx(0.0)


def test_engine_is_reproducible(market, full_result):
    again = run_day(market, VERSIONS["v4_fade"], FLOW)
    assert again.totals == full_result.totals


def test_inventory_limits_are_respected(market):
    cfg = Config(strategy=VERSIONS["v0_naive"].strategy, risk=RiskParams(max_contracts_per_option=30))
    res = run_day(market, cfg, FLOW)
    assert np.abs(res.sample_q).max() <= 30


def test_liquidation_accounting_identity(full_result):
    t = full_result.totals
    expected = t["pnl_before_liquidation"] - t["liquidation_cost"]
    # seuls les frais de liquidation separent les deux montants
    assert t["net_pnl"] <= expected + 1e-6
    assert expected - t["net_pnl"] < 0.01 * abs(expected) + 50


def test_kpi_summary_is_consistent(market, full_result):
    row = day_summary(full_result, market)
    assert row["fills"] == len(full_result.fills) > 0
    parts = row["edge"] + row["attr_inventory"] - row["fees"] - row["hedge_cost"] + row["attr_timing"]
    assert parts == pytest.approx(row["pnl_mtm"])
    assert 0.0 <= row["uptime"] <= 1.0


def test_markouts_start_from_edge(market, full_result):
    mk = fill_markouts(full_result, market, horizons=(0,))
    np.testing.assert_allclose(mk["mk_0s"], mk["edge"])


def test_quotes_never_cross_the_screen(market, full_result):
    f = full_result.fills
    bids, asks = f[f["we_side"] > 0], f[f["we_side"] < 0]
    assert np.all(bids["price"].to_numpy() <= market.comp_ask[bids["t"], bids["i"]] - 0.01 + 1e-9)
    assert np.all(asks["price"].to_numpy() >= market.comp_bid[asks["t"], asks["i"]] + 0.01 - 1e-9)
