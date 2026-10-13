import datetime as dt

import numpy as np
import pytest

from amm.conventions import business_days_between, option_type_from_ticker, round_to_tick, year_fraction
from amm.glft import factor_covariance, glft_c1, riccati_gamma, single_asset_gamma
from amm.svi import SVIParams, butterfly_density, fit_svi


# ---------------------------------------------------------------- SVI
def test_fit_svi_recovers_known_smile():
    true = SVIParams(a=0.004, b=0.08, rho=-0.4, m=0.02, sigma=0.12)
    k = np.linspace(-0.3, 0.3, 25)
    fitted = fit_svi(k, true.total_variance(k))
    np.testing.assert_allclose(fitted.total_variance(k), true.total_variance(k), atol=2e-5)


def test_fitted_smile_has_no_butterfly_arbitrage():
    true = SVIParams(a=0.01, b=0.1, rho=-0.3, m=0.0, sigma=0.2)
    k = np.linspace(-0.35, 0.35, 15)
    noisy = true.total_variance(k) * (1 + 0.01 * np.sin(7 * k))
    assert butterfly_density(fit_svi(k, noisy), np.linspace(-0.35, 0.35, 200)).min() >= 0


def test_fit_svi_requires_enough_points():
    with pytest.raises(ValueError):
        fit_svi(np.array([0.0, 0.1]), np.array([0.01, 0.012]))


# ---------------------------------------------------------------- GLFT
def test_riccati_reduces_to_single_asset_glft():
    sigma2, A, k, clip, gamma = 0.003, 0.01, 1.2, 10.0, 5e-3
    G = riccati_gamma(np.array([[sigma2]]), np.array([A]), np.array([k]), np.array([clip]), gamma)
    assert G[0, 0] == pytest.approx(single_asset_gamma(sigma2, A, k, clip, gamma), rel=1e-10)


def test_riccati_solves_matrix_equation():
    rng = np.random.default_rng(0)
    n = 6
    vega = rng.uniform(1, 8, n) * 100
    sigma = factor_covariance(vega, rng.uniform(0.7, 1.5, n), 7e-5, 1e-5,
                              delta_contract=rng.uniform(-90, 90, n), spot_vol_per_s=0.01, delta_weight=0.1)
    A, k, clip, gamma = rng.uniform(1e-3, 1e-2, n), rng.uniform(0.5, 2, n), np.full(n, 10.0), 5e-3
    G = riccati_gamma(sigma, A, k, clip, gamma)
    x = gamma * clip / k
    D = np.diag(A * k * clip * np.exp(-(1 + 1 / x) * np.log1p(x)))
    np.testing.assert_allclose(G @ D @ G, 0.5 * gamma * sigma, rtol=1e-8, atol=1e-14)
    assert np.all(np.linalg.eigvalsh(G) >= -1e-12)


def test_long_vega_inventory_lowers_all_reservation_prices():
    vega = np.array([300.0, 600.0, 400.0])
    sigma = factor_covariance(vega, np.ones(3), 8e-5, 1e-5)
    G = riccati_gamma(sigma, np.full(3, 5e-3), np.ones(3), np.full(3, 10.0), 5e-3)
    skew = G @ np.array([50.0, 0.0, 0.0])          # long uniquement l'option 0
    assert np.all(skew > 0)                         # tous les quotes baissent (on vend la vega)


def test_c1_tends_to_inverse_k():
    assert glft_c1(1e-9, np.array([2.0]), np.array([10.0]))[0] == pytest.approx(0.5, rel=1e-6)


# ---------------------------------------------------------------- Conventions B3
def test_business_days_skip_b3_holiday():
    # 12/10/2026 (Nossa Senhora Aparecida) n'est pas ouvre
    assert business_days_between(dt.date(2026, 10, 9), dt.date(2026, 10, 16)) == 4


def test_year_fraction_decays_intraday():
    open_, close = (year_fraction(dt.date(2026, 10, 9), dt.date(2026, 10, 16), e) for e in (0.0, 1.0))
    assert open_ - close == pytest.approx(1 / 252)


def test_ticker_letter_conventions():
    assert option_type_from_ticker("PETRK516") == "CALL"
    assert option_type_from_ticker("PETRW516") == "PUT"


def test_round_to_tick_never_favours_market_maker():
    np.testing.assert_allclose(round_to_tick(np.array([1.234]), "down"), [1.23])
    np.testing.assert_allclose(round_to_tick(np.array([1.231]), "up"), [1.24])
