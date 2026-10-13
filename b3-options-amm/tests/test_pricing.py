import numpy as np
import pytest

from amm.pricing import bs_greeks, bs_price, implied_vol, norm_cdf

S, K, T, R = 56.0, np.array([45.0, 50.0, 56.0, 62.0, 70.0]), 0.1, np.log1p(0.1365)


def test_norm_cdf_matches_known_values():
    x = np.array([-6.0, -1.96, 0.0, 1.0, 3.0])
    expected = np.array([9.865876450377e-10, 0.024997895148, 0.5, 0.841344746069, 0.998650101968])
    np.testing.assert_allclose(norm_cdf(x), expected, rtol=2e-7)


def test_put_call_parity():
    sigma = np.full_like(K, 0.33)
    call = bs_price(S, K, T, R, sigma, True)
    put = bs_price(S, K, T, R, sigma, False)
    np.testing.assert_allclose(call - put, S - K * np.exp(-R * T), atol=1e-9)


def test_greeks_match_finite_differences():
    # pas assez grands pour que l'erreur de norm_cdf (~1e-7 relatif) soit negligeable
    sigma, h_s, h_v = 0.35, 1e-2, 1e-3
    for is_call in (True, False):
        g = bs_greeks(S, K, T, R, sigma, is_call)
        mid = bs_price(S, K, T, R, sigma, is_call)
        up, dn = bs_price(S + h_s, K, T, R, sigma, is_call), bs_price(S - h_s, K, T, R, sigma, is_call)
        np.testing.assert_allclose(g["delta"], (up - dn) / (2 * h_s), atol=1e-5)
        np.testing.assert_allclose(g["gamma"], (up - 2 * mid + dn) / h_s ** 2, rtol=1e-3, atol=1e-5)
        v_up, v_dn = bs_price(S, K, T, R, sigma + h_v, is_call), bs_price(S, K, T, R, sigma - h_v, is_call)
        np.testing.assert_allclose(g["vega"], (v_up - v_dn) / (2 * h_v), rtol=1e-4, atol=1e-5)
        t_up = bs_price(S, K, T + 1e-4, R, sigma, is_call)
        np.testing.assert_allclose(g["theta"], -(t_up - mid) / 1e-4, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("is_call", [True, False])
def test_implied_vol_round_trip(is_call):
    sigma = np.array([0.15, 0.25, 0.33, 0.6, 1.2])
    price = bs_price(S, K, T, R, sigma, is_call)
    iv = implied_vol(price, S, K, T, R, is_call)
    np.testing.assert_allclose(bs_price(S, K, T, R, iv, is_call), price, atol=1e-8)
    well_conditioned = bs_greeks(S, K, T, R, sigma, is_call)["vega"] > 1e-2
    np.testing.assert_allclose(iv[well_conditioned], sigma[well_conditioned], atol=1e-6)


def test_implied_vol_rejects_arbitrage_prices():
    below_intrinsic = np.array([S - 45.0 - 1.0])
    assert np.isnan(implied_vol(below_intrinsic, S, np.array([45.0]), T, R, True)[0])


def test_expired_option_pays_intrinsic():
    assert bs_price(56.0, 50.0, 0.0, R, 0.3, True) == pytest.approx(6.0)
    assert bs_price(56.0, 50.0, 0.0, R, 0.3, False) == pytest.approx(0.0)
