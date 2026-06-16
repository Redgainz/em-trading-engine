"""
quant_engine.py
===============
Tous les modeles quants utilises dans le simulateur de trading.
Implementation NUMPY PURE — aucune dependance scipy/sklearn.

Contient :
  - Black-Scholes Garman-Kohlhagen (FX) call/put + Greeks complets
  - SABR Hagan formula pour implied vol
  - GARCH(1,1) variance targeting + forecast
  - EWMA volatility (RiskMetrics lambda=0.94)
  - Parkinson range-based volatility
  - Markov regime detection via Baum-Welch (HMM)
  - VAR(p) regression OLS + forecast
  - Historical VaR / CVaR
  - Nelson-Siegel yield curve
"""
from __future__ import annotations
import numpy as np


# ============================================================
# Helpers : normal CDF/PDF en numpy pur
# ============================================================
SQRT_2PI = np.sqrt(2.0 * np.pi)

def _norm_pdf(x: float | np.ndarray) -> float | np.ndarray:
    return np.exp(-0.5 * x ** 2) / SQRT_2PI

def _erf(x: float | np.ndarray) -> float | np.ndarray:
    """Abramowitz-Stegun 7.1.26, max abs error 1.5e-7."""
    a1, a2, a3 = 0.254829592, -0.284496736, 1.421413741
    a4, a5, p = -1.453152027, 1.061405429, 0.3275911
    sign = np.sign(x); x = np.abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y

def _norm_cdf(x: float | np.ndarray) -> float | np.ndarray:
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


# ============================================================
# Black-Scholes Garman-Kohlhagen (FX options)
# ============================================================
# Notation : S = spot, K = strike, T = maturite (en annees),
#            r_d = taux domestique, r_f = taux etranger,
#            sigma = vol implicite annualisee.
# Pour options actions : r_f = q (taux dividende).
# Pour options actions sans div : r_f = 0.

def _bs_d1_d2(S, K, T, r_d, r_f, sigma):
    T_safe = max(T, 1e-8)
    sigma_safe = max(sigma, 1e-8)
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma_safe ** 2) * T_safe) / (sigma_safe * np.sqrt(T_safe))
    d2 = d1 - sigma_safe * np.sqrt(T_safe)
    return d1, d2

def bs_call(S, K, T, r_d, r_f, sigma) -> float:
    """Prix Call europeen Garman-Kohlhagen."""
    if T <= 0: return max(S - K, 0.0)
    d1, d2 = _bs_d1_d2(S, K, T, r_d, r_f, sigma)
    return float(S * np.exp(-r_f * T) * _norm_cdf(d1) - K * np.exp(-r_d * T) * _norm_cdf(d2))

def bs_put(S, K, T, r_d, r_f, sigma) -> float:
    """Prix Put europeen Garman-Kohlhagen (parite call-put)."""
    if T <= 0: return max(K - S, 0.0)
    d1, d2 = _bs_d1_d2(S, K, T, r_d, r_f, sigma)
    return float(K * np.exp(-r_d * T) * _norm_cdf(-d2) - S * np.exp(-r_f * T) * _norm_cdf(-d1))

def bs_greeks(S, K, T, r_d, r_f, sigma, kind: str = "call") -> dict:
    """
    Renvoie dict avec :
      delta, gamma, vega, theta, rho_d, rho_f.
    vega en pourcentage de vol (1.0 = 100% mvt vol)
    theta en jour ouvre (T -> T - 1/252)
    """
    if T <= 0:
        intrinsic_call = max(S - K, 0.0); intrinsic_put = max(K - S, 0.0)
        delta = 1.0 if (kind == "call" and S > K) else (-1.0 if (kind == "put" and S < K) else 0.0)
        return dict(delta=delta, gamma=0.0, vega=0.0, theta=0.0, rho_d=0.0, rho_f=0.0)

    d1, d2 = _bs_d1_d2(S, K, T, r_d, r_f, sigma)
    Nd1 = _norm_cdf(d1); Nd2 = _norm_cdf(d2)
    Nmd1 = _norm_cdf(-d1); Nmd2 = _norm_cdf(-d2)
    nd1 = _norm_pdf(d1)
    sqrtT = np.sqrt(T)
    discount_d = np.exp(-r_d * T); discount_f = np.exp(-r_f * T)

    if kind == "call":
        delta = float(discount_f * Nd1)
        theta = float(- (S * discount_f * nd1 * sigma) / (2 * sqrtT)
                       - r_d * K * discount_d * Nd2
                       + r_f * S * discount_f * Nd1) / 252.0  # per business day
        rho_d = float(K * T * discount_d * Nd2) / 100.0  # per 1% rate move
        rho_f = float(-T * S * discount_f * Nd1) / 100.0
    else:  # put
        delta = float(-discount_f * Nmd1)
        theta = float(- (S * discount_f * nd1 * sigma) / (2 * sqrtT)
                       + r_d * K * discount_d * Nmd2
                       - r_f * S * discount_f * Nmd1) / 252.0
        rho_d = float(-K * T * discount_d * Nmd2) / 100.0
        rho_f = float(T * S * discount_f * Nmd1) / 100.0

    gamma = float((discount_f * nd1) / (S * sigma * sqrtT))
    vega  = float(S * discount_f * nd1 * sqrtT) / 100.0  # per 1% vol move

    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta,
                rho_d=rho_d, rho_f=rho_f)


# ============================================================
# SABR (Hagan 2002)
# ============================================================
def sabr_implied_vol(F: float, K: float, T: float,
                       alpha: float, beta: float, rho: float, nu: float) -> float:
    """
    Renvoie la vol implicite Black SABR via la formule de Hagan 2002.
    F : forward, K : strike, T : maturite (annees).
    alpha, beta, rho, nu : parametres SABR.
    """
    if T <= 0 or alpha <= 0:
        return float("nan")
    if abs(F - K) < 1e-10:
        # ATM formula simplifiee
        term1 = alpha / (F ** (1 - beta))
        term2 = 1 + T * (
            ((1 - beta) ** 2 / 24) * (alpha ** 2 / (F ** (2 * (1 - beta))))
            + (rho * beta * nu * alpha) / (4 * F ** (1 - beta))
            + ((2 - 3 * rho ** 2) / 24) * nu ** 2
        )
        return float(term1 * term2)
    z = (nu / alpha) * (F * K) ** ((1 - beta) / 2) * np.log(F / K)
    sqrt_term = np.sqrt(1 - 2 * rho * z + z ** 2)
    x_z = np.log((sqrt_term + z - rho) / (1 - rho)) if abs(z) > 1e-10 else 1.0
    log_FK = np.log(F / K)
    factor1 = alpha / (
        ((F * K) ** ((1 - beta) / 2)) *
        (1 + (1 - beta) ** 2 / 24 * log_FK ** 2 + (1 - beta) ** 4 / 1920 * log_FK ** 4)
    )
    factor2 = (z / x_z) if abs(z) > 1e-10 else 1.0
    factor3 = 1 + T * (
        ((1 - beta) ** 2 / 24) * (alpha ** 2 / ((F * K) ** (1 - beta)))
        + (rho * beta * nu * alpha) / (4 * (F * K) ** ((1 - beta) / 2))
        + ((2 - 3 * rho ** 2) / 24) * nu ** 2
    )
    return float(factor1 * factor2 * factor3)

def sabr_vol_surface(F: float, T: float, params: dict,
                       moneyness_grid: np.ndarray | None = None) -> np.ndarray:
    """Genere une surface (moneyness x vol) pour une maturite T."""
    if moneyness_grid is None:
        moneyness_grid = np.linspace(0.85, 1.15, 13)
    vols = np.array([sabr_implied_vol(F, F * m, T, **params) for m in moneyness_grid])
    return moneyness_grid, vols


# ============================================================
# GARCH(1,1) variance targeting (no MLE needed)
# ============================================================
def garch_forecast(returns: np.ndarray, omega_target: float | None = None,
                     alpha: float = 0.07, beta: float = 0.91, horizon: int = 21) -> dict:
    """
    GARCH(1,1) avec variance targeting : omega est calcule pour que la vol
    long-run = la vol empirique. alpha et beta sont les parametres standards.

    Returns dict : sigma_t1 (forecast 1 step), sigma_h (horizon multi-step),
                    sigma_long_run, sigma_path (path 1..h).
    """
    r = np.asarray(returns)
    var_lt = r.var(ddof=1)
    if omega_target is None:
        omega = var_lt * (1 - alpha - beta)
    else:
        omega = omega_target

    # Compute conditional variance path
    sigma2 = np.zeros(len(r))
    sigma2[0] = var_lt
    for t in range(1, len(r)):
        sigma2[t] = omega + alpha * r[t-1] ** 2 + beta * sigma2[t-1]

    # 1-step forecast
    sigma2_t1 = omega + alpha * r[-1] ** 2 + beta * sigma2[-1]
    # Multi-step forecast (mean-reverting)
    sigma_path = np.zeros(horizon)
    s2 = sigma2_t1
    for h in range(horizon):
        sigma_path[h] = np.sqrt(s2)
        s2 = omega + (alpha + beta) * s2

    return dict(
        sigma_t1=float(np.sqrt(sigma2_t1)),
        sigma_h=float(np.sqrt(sigma_path.mean() ** 2 + 0)),
        sigma_long_run=float(np.sqrt(var_lt)),
        sigma_path=sigma_path,
        omega=float(omega), alpha=float(alpha), beta=float(beta),
    )


def ewma_vol(returns: np.ndarray, lambda_: float = 0.94) -> float:
    """Vol EWMA (RiskMetrics). lambda=0.94 = standard."""
    r = np.asarray(returns)
    weights = (1 - lambda_) * lambda_ ** np.arange(len(r))[::-1]
    weights /= weights.sum()
    var = np.sum(weights * (r - r.mean()) ** 2)
    return float(np.sqrt(var))


def realized_vol_parkinson(high: np.ndarray, low: np.ndarray) -> float:
    """Vol realisee par estimateur de Parkinson (range-based)."""
    log_hl = np.log(high / low)
    var = (1.0 / (4 * np.log(2))) * np.mean(log_hl ** 2)
    return float(np.sqrt(var))


# ============================================================
# Markov regime detection (Baum-Welch HMM en numpy)
# ============================================================
def markov_regimes(returns: np.ndarray, n_states: int = 3,
                     n_iter: int = 50, tol: float = 1e-4) -> dict:
    """
    HMM gaussien : K etats avec means mu_k et stds sigma_k.
    Renvoie les probas d'etat a chaque date (gamma) + parametres.
    """
    r = np.asarray(returns)
    T = len(r)
    rng = np.random.default_rng(42)

    # Init : K-means like sur quantiles de r^2 (proxy regime)
    abs_r = np.abs(r)
    quantiles = np.quantile(abs_r, np.linspace(0, 1, n_states + 1)[1:-1])
    means = np.zeros(n_states)
    sigmas = np.zeros(n_states)
    for k in range(n_states):
        lo = quantiles[k-1] if k > 0 else -np.inf
        hi = quantiles[k] if k < n_states - 1 else np.inf
        mask = (abs_r >= lo) & (abs_r < hi) if k < n_states - 1 else (abs_r >= lo)
        if mask.sum() > 0:
            means[k] = r[mask].mean()
            sigmas[k] = max(r[mask].std(), 1e-5)
        else:
            means[k] = r.mean(); sigmas[k] = r.std()

    # Init transition : sticky (0.9 diagonal)
    P = np.full((n_states, n_states), 0.1 / (n_states - 1))
    np.fill_diagonal(P, 0.9)
    pi = np.ones(n_states) / n_states

    def emission_prob(x, mu, sd):
        return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * SQRT_2PI)

    log_lik_prev = -np.inf
    for it in range(n_iter):
        # E step : forward-backward
        B = np.array([emission_prob(r, means[k], sigmas[k]) for k in range(n_states)]).T  # T x K
        alpha = np.zeros((T, n_states)); c = np.zeros(T)
        alpha[0] = pi * B[0]; c[0] = alpha[0].sum(); alpha[0] /= max(c[0], 1e-12)
        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ P) * B[t]
            c[t] = alpha[t].sum()
            alpha[t] /= max(c[t], 1e-12)
        log_lik = np.sum(np.log(np.clip(c, 1e-12, None)))

        beta = np.zeros((T, n_states))
        beta[-1] = 1.0 / max(c[-1], 1e-12)
        for t in range(T - 2, -1, -1):
            beta[t] = P @ (B[t+1] * beta[t+1]) / max(c[t], 1e-12)

        gamma = alpha * beta
        gamma = gamma / np.clip(gamma.sum(1, keepdims=True), 1e-12, None)

        # xi (joint probas)
        xi = np.zeros((T-1, n_states, n_states))
        for t in range(T-1):
            denom = (alpha[t][:, None] * P * B[t+1][None, :] * beta[t+1][None, :]).sum()
            if denom > 1e-12:
                xi[t] = alpha[t][:, None] * P * B[t+1][None, :] * beta[t+1][None, :] / denom

        # M step
        pi = gamma[0]
        P_new = xi.sum(0) / np.clip(gamma[:-1].sum(0)[:, None], 1e-12, None)
        for k in range(n_states):
            w = gamma[:, k]; w_sum = w.sum()
            if w_sum > 1e-12:
                means[k] = np.sum(w * r) / w_sum
                sigmas[k] = max(np.sqrt(np.sum(w * (r - means[k]) ** 2) / w_sum), 1e-5)
        P = P_new

        if abs(log_lik - log_lik_prev) < tol:
            break
        log_lik_prev = log_lik

    # Trier les etats par vol croissante
    order = np.argsort(sigmas)
    means = means[order]; sigmas = sigmas[order]
    P = P[order][:, order]; gamma = gamma[:, order]

    return dict(means=means, sigmas=sigmas, transition=P,
                gamma=gamma, log_lik=float(log_lik_prev),
                current_regime_probs=gamma[-1])


# ============================================================
# VAR(p) regression
# ============================================================
def var_p_fit(Y: np.ndarray, p: int = 5):
    """
    Fit VAR(p) par OLS multivariate.
    Y : (T, K) matrix de K series de longueur T.
    Renvoie dict : intercept (K,), A (p*K, K), residuals, sigma_cov (KxK).
    """
    T, K = Y.shape
    # Construire X = [1, Y_{t-1}, ..., Y_{t-p}]
    X = np.ones((T - p, p * K + 1))
    for lag in range(1, p + 1):
        X[:, 1 + (lag - 1) * K : 1 + lag * K] = Y[p - lag : T - lag]
    Y_target = Y[p:]
    beta, *_ = np.linalg.lstsq(X, Y_target, rcond=None)
    residuals = Y_target - X @ beta
    sigma = residuals.T @ residuals / (T - p - p * K - 1)
    return dict(intercept=beta[0], A=beta[1:], residuals=residuals,
                sigma_cov=sigma, p=p, K=K)


def var_p_forecast(fit: dict, Y_recent: np.ndarray, horizon: int = 21) -> np.ndarray:
    """Forecast deterministe VAR(p) sur `horizon` steps (no noise)."""
    p = fit["p"]; K = fit["K"]
    intercept = fit["intercept"]; A = fit["A"]
    history = list(Y_recent[-p:])
    forecasts = []
    for h in range(horizon):
        X_row = np.concatenate([history[-lag] for lag in range(1, p + 1)])
        y_next = intercept + X_row @ A
        forecasts.append(y_next)
        history.append(y_next)
    return np.array(forecasts)


# ============================================================
# Historical VaR / CVaR
# ============================================================
def historical_var(returns: np.ndarray, level: float = 0.95) -> float:
    """VaR a `level` (e.g. 0.95 = 5%-tile). Renvoie un nombre positif."""
    return float(-np.percentile(returns, 100 * (1 - level)))

def historical_cvar(returns: np.ndarray, level: float = 0.95) -> float:
    """CVaR / Expected Shortfall a `level`."""
    var = historical_var(returns, level)
    tail = returns[returns <= -var]
    return float(-tail.mean()) if len(tail) > 0 else var


# ============================================================
# Nelson-Siegel yield curve
# ============================================================
def nelson_siegel_curve(taus: np.ndarray, beta0: float, beta1: float,
                          beta2: float, lambda_: float = 0.5) -> np.ndarray:
    """Diebold-Li parametrization of Nelson-Siegel."""
    taus = np.asarray(taus)
    f1 = (1 - np.exp(-lambda_ * taus)) / (lambda_ * taus + 1e-10)
    f2 = f1 - np.exp(-lambda_ * taus)
    return beta0 + beta1 * f1 + beta2 * f2


# ============================================================
# Self-test
# ============================================================
if __name__ == "__main__":
    print("Test quant_engine\n")
    # BS call ATM
    p = bs_call(100, 100, 1.0, 0.02, 0.005, 0.20)
    g = bs_greeks(100, 100, 1.0, 0.02, 0.005, 0.20, "call")
    print(f"BS Call ATM(100, 100, 1Y, sigma=20%) = {p:.4f}")
    print(f"  Delta={g['delta']:.4f}  Gamma={g['gamma']:.4f}")
    print(f"  Vega={g['vega']:.4f}  Theta={g['theta']:.4f}")
    # SABR
    F, T = 100.0, 0.5
    params = dict(alpha=0.15, beta=0.5, rho=-0.2, nu=0.4)
    iv_atm = sabr_implied_vol(F, F, T, **params)
    print(f"\nSABR ATM vol = {iv_atm:.4f}")
    moneyness, vols = sabr_vol_surface(F, T, params)
    print(f"SABR smile (90% to 110%) : {vols[2:-2].round(3)}")
    # GARCH
    rng = np.random.default_rng(0)
    r = rng.standard_normal(500) * 0.01
    fc = garch_forecast(r)
    print(f"\nGARCH forecast: sigma_t+1 = {fc['sigma_t1']*np.sqrt(252)*100:.2f}% annualized")
    # Markov
    fast = markov_regimes(r, n_states=3)
    print(f"\nMarkov regimes: sigmas={fast['sigmas']*np.sqrt(252)*100}")
    # VaR
    print(f"\nVaR-95 = {historical_var(r, 0.95)*100:.2f}%")
    print(f"CVaR-95= {historical_cvar(r, 0.95)*100:.2f}%")
