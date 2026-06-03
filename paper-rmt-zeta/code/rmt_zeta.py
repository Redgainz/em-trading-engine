"""
rmt_zeta.py
===========
Module central : Random Matrix Theory (RMT) + test d'universalite
Montgomery-Odlyzko (statistique GUE = pair correlation des zeros de
la fonction zeta de Riemann).

Toute la logique purement numpy : aucun scipy ni sklearn requis.

REFERENCES :
  - Marchenko & Pastur (1967) : distribution des valeurs propres
  - Tracy & Widom (1994)      : loi de la plus grande valeur propre
  - Montgomery (1973)         : pair correlation des zeros de zeta
  - Odlyzko (1987)            : confirmation empirique sur 10^9 zeros
  - Laloux et al. (1999)      : application aux matrices de covariance
  - Plerou et al. (2002)      : structure des marches d'actions
  - Bouchaud & Potters (2009) : nettoyage spectral en finance
"""

from __future__ import annotations
import numpy as np


# ============================================================
# 1.  STATISTIQUES SPECTRALES DE BASE
# ============================================================

def correlation_matrix(R: np.ndarray) -> np.ndarray:
    """Matrice de correlation empirique a partir de rendements (T x N)."""
    # standardise puis correlation = C = X.T @ X / T
    X = (R - R.mean(0)) / R.std(0, ddof=1)
    return X.T @ X / X.shape[0]


def eigen_spectrum(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Renvoie (eigenvalues triees decroissantes, eigenvectors associes)."""
    w, V = np.linalg.eigh(C)         # eigh garantit reel + tri ascendant
    idx = np.argsort(w)[::-1]
    return w[idx], V[:, idx]


# ============================================================
# 2.  MARCHENKO-PASTUR (borne "bruit")
# ============================================================

def mp_bounds(T: int, N: int, sigma2: float = 1.0) -> tuple[float, float]:
    """Bornes lambda_minus et lambda_plus de la distribution MP."""
    Q = T / N
    sqrtQ = 1.0 / np.sqrt(Q)
    lam_p = sigma2 * (1 + sqrtQ) ** 2
    lam_m = sigma2 * (1 - sqrtQ) ** 2
    return lam_m, lam_p


def mp_pdf(lmbda: np.ndarray, T: int, N: int, sigma2: float = 1.0) -> np.ndarray:
    """Densite Marchenko-Pastur evaluee sur la grille `lmbda`."""
    Q = T / N
    lam_m, lam_p = mp_bounds(T, N, sigma2)
    out = np.zeros_like(lmbda, dtype=float)
    inside = (lmbda > lam_m) & (lmbda < lam_p)
    x = lmbda[inside]
    out[inside] = Q / (2 * np.pi * sigma2 * x) * np.sqrt((lam_p - x) * (x - lam_m))
    return out


def mp_noise_threshold(T: int, N: int, sigma2: float = 1.0) -> float:
    """Seuil au-dessus duquel un eigenvalue est juge non-bruit."""
    return mp_bounds(T, N, sigma2)[1]


def mp_sigma2_fit(eigvals: np.ndarray, T: int, N: int,
                   max_iter: int = 50, tol: float = 1e-4) -> float:
    """
    Estime sigma^2 effectif du bruit en retirant iterativement les eigenvalues
    qui depassent lambda_+ (Laloux 1999).

    Approche : K = # eigenvalues > lambda_+ (sigma2), puis
               sigma2 <- sum(bulk) / (N - K).  Bulk-clamp pour stabilite.
    """
    eig = np.sort(eigvals)[::-1]              # descending
    sigma2 = 1.0
    K_prev = -1
    for _ in range(max_iter):
        lam_p = mp_bounds(T, N, sigma2)[1]
        K = int(np.sum(eig > lam_p))
        # Stabilite : on garde au moins N // 2 eigenvalues dans le bulk
        K = min(K, max(0, N - max(5, N // 2)))
        bulk_sum = eig[K:].sum()
        new_sigma2 = bulk_sum / (N - K) if N > K else sigma2
        if abs(new_sigma2 - sigma2) < tol and K == K_prev:
            sigma2 = new_sigma2
            break
        sigma2 = new_sigma2
        K_prev = K
    return float(max(sigma2, 1e-3))


# ============================================================
# 3.  TRACY-WIDOM (loi du plus grand eigenvalue)
# ============================================================

def tracy_widom_normalize(lmax: float, T: int, N: int, sigma2: float = 1.0) -> float:
    """
    Normalisation pour que (lmax - mu) / sigma -> TW_1 (orthogonal).
    Formule de Johnstone (2001).
    """
    # mu = (sqrt(T-1) + sqrt(N))^2 / T
    mu = (np.sqrt(T - 1) + np.sqrt(N)) ** 2 / T
    sigma = (np.sqrt(T - 1) + np.sqrt(N)) / T * (1 / np.sqrt(T - 1) + 1 / np.sqrt(N)) ** (1/3)
    return (lmax / sigma2 - mu) / sigma


# Table d'approximation TW_1 (Bornemann 2010) : quantiles
TW1_QUANTILES = {
    0.01: -3.90, 0.05: -3.18, 0.10: -2.78, 0.25: -2.04, 0.50: -1.27,
    0.75: -0.59, 0.90:  0.45, 0.95:  0.98, 0.99:  2.02
}

def tracy_widom_pvalue_upper(z: float) -> float:
    """p-value approximative pour Tracy-Widom_1 (test : largest eig anormal)."""
    quants = sorted(TW1_QUANTILES.items())
    # interpolation lineaire inverse
    if z <= quants[0][1]:
        return 1.0
    if z >= quants[-1][1]:
        return 0.005
    for i in range(len(quants) - 1):
        q1, z1 = quants[i]
        q2, z2 = quants[i + 1]
        if z1 <= z <= z2:
            p = q1 + (q2 - q1) * (z - z1) / (z2 - z1)
            return 1.0 - p
    return 0.5


# ============================================================
# 4.  STATISTIQUE GUE / RIEMANN ZETA (Montgomery-Odlyzko)
# ============================================================
#
# Montgomery (1973) a montre que les espacements normalises des zeros
# non-triviaux de la fonction zeta de Riemann suivent (conjecturalement)
# la pair correlation function du Gaussian Unitary Ensemble (GUE) :
#
#   R_2(s) = 1 - (sin(pi s) / (pi s))^2
#
# La distribution des espacements des plus proches voisins (NN) est
# tres bien approximee par la "Wigner surmise" :
#
#   p_GUE(s) = (32 / pi^2) s^2 exp( -(4/pi) s^2 )
#
# Pour appliquer ceci a des eigenvalues empiriques d'une matrice de
# correlation, on doit UNFOLDER le spectre (= reescaler pour avoir
# densite moyenne = 1) avant de calculer les spacings.

def unfold_spectrum(eigvals: np.ndarray) -> np.ndarray:
    """
    Unfold (resp. "spectral unfolding") : transforme les eigenvalues en
    variables uniformement distribuees (densite locale moyenne = 1).

    Methode : fit polynomial de la fonction de comptage cumulee N(lambda).
    """
    eig = np.sort(eigvals)
    # On enleve les outliers (premier(s) eigenvalues "non-bruit") pour le fit
    n = len(eig)
    # Fonction de comptage empirique : N(lambda_i) = i
    counts = np.arange(1, n + 1)
    # Fit polynomial degre 5 (robuste pour spectres bruites de MP)
    coefs = np.polyfit(eig, counts, deg=min(5, n - 1))
    N_smooth = np.polyval(coefs, eig)
    # Les valeurs N_smooth sont les eigenvalues unfolded
    return N_smooth


def nn_spacings(unfolded: np.ndarray) -> np.ndarray:
    """Espacements aux plus proches voisins (nearest-neighbor spacings)."""
    return np.diff(np.sort(unfolded))


def wigner_gue_pdf(s: np.ndarray) -> np.ndarray:
    """Surmise de Wigner pour le GUE."""
    return (32 / np.pi ** 2) * s ** 2 * np.exp(-(4 / np.pi) * s ** 2)


def wigner_gue_cdf(s: np.ndarray) -> np.ndarray:
    """
    CDF de la Wigner-GUE.
    Integrale analytique :
      F(s) = 1 - exp(-4 s^2 / pi) ( 1 + (4 s^2)/pi )^{?}
    Formule plus simple : integration numerique par trapeze sur grille fine.
    """
    s = np.asarray(s, dtype=float)
    grid = np.linspace(0, max(s.max(), 5), 5000)
    pdf = wigner_gue_pdf(grid)
    cdf_grid = np.cumsum(pdf) * (grid[1] - grid[0])
    cdf_grid = np.clip(cdf_grid, 0, 1)
    return np.interp(s, grid, cdf_grid)


def wigner_poisson_pdf(s: np.ndarray) -> np.ndarray:
    """Reference : niveaux non-correles (Poisson) -> exp(-s)."""
    return np.exp(-s)


def ks_test(sample: np.ndarray, cdf_fn) -> tuple[float, float]:
    """
    Test KS one-sample contre une CDF theorique.
    Implementation numpy : statistique D = max |F_n(x) - F(x)|.
    Renvoie (D, p_value_approx).
    """
    x = np.sort(sample)
    n = len(x)
    F_emp = np.arange(1, n + 1) / n
    F_th = cdf_fn(x)
    D = np.max(np.abs(F_emp - F_th))
    # p-value approx : Kolmogorov distribution Q(lambda) = 2 sum (-1)^{k-1} exp(-2 k^2 lambda^2)
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * D
    p = 2 * sum(((-1) ** (k - 1)) * np.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return float(D), float(np.clip(p, 0, 1))


def zeta_universality_distance(eigvals: np.ndarray) -> dict:
    """
    Mesure la deviation du spectre empirique a la statistique GUE
    (= statistique des zeros de zeta).

    Output :
      D_KS_GUE       : distance Kolmogorov-Smirnov a la Wigner-GUE
      p_GUE          : p-value (H0 = GUE-distributed)
      D_KS_Poisson   : distance KS a Poisson (reference de levels NON correles)
      zeta_score     : log-ratio des deux distances (>0 -> plus proche GUE)
    """
    unf = unfold_spectrum(eigvals)
    s = nn_spacings(unf)
    # Normalisation : <s> = 1
    s = s / s.mean()
    D_gue, p_gue = ks_test(s, wigner_gue_cdf)
    # Pour Poisson : CDF = 1 - exp(-s)
    D_poi, p_poi = ks_test(s, lambda x: 1 - np.exp(-x))
    zeta_score = float(np.log((D_poi + 1e-6) / (D_gue + 1e-6)))
    return dict(D_KS_GUE=float(D_gue), p_GUE=float(p_gue),
                D_KS_Poisson=float(D_poi), p_Poisson=float(p_poi),
                zeta_score=zeta_score)


# ============================================================
# 5.  EXTRACTION DE FEATURES POUR LA PREDICTION
# ============================================================

def rmt_features(R_window: np.ndarray) -> dict:
    """
    Extrait l'ensemble des features RMT + Zeta d'une fenetre de rendements.

    R_window : (T_w, N)
    Output   : dict de scalaires (~12 features)
    """
    T_w, N = R_window.shape
    C = correlation_matrix(R_window)
    eigvals, V = eigen_spectrum(C)

    lam_m, lam_p = mp_bounds(T_w, N)
    sigma2 = mp_sigma2_fit(eigvals, T_w, N)
    lam_m2, lam_p2 = mp_bounds(T_w, N, sigma2)

    # Top eigenvalue (mode marche)
    lmax = float(eigvals[0])
    lam2 = float(eigvals[1])
    lam3 = float(eigvals[2])

    # Tracy-Widom z-score
    tw_z = tracy_widom_normalize(lmax, T_w, N, sigma2)
    p_tw = tracy_widom_pvalue_upper(tw_z)

    # Variance expliquee par mode marche
    var_market = lmax / N

    # Nombre de facteurs informatifs
    n_factors = int(np.sum(eigvals > lam_p2))

    # Participation ratio du premier eigenvector (= inverse Herfindahl)
    v1 = V[:, 0]
    IPR = np.sum(v1 ** 4)             # plus grand = plus concentre
    PR  = 1.0 / (N * IPR)             # entre 1/N et 1 (normalise)

    # Zeta-GUE statistic
    z = zeta_universality_distance(eigvals)

    return dict(
        lambda_max=lmax,
        lambda_2=lam2,
        lambda_3=lam3,
        var_market=var_market,
        sigma2_noise=sigma2,
        n_factors=n_factors,
        TW_z=float(tw_z),
        TW_p=float(p_tw),
        participation_ratio=float(PR),
        IPR_v1=float(IPR),
        D_KS_GUE=z["D_KS_GUE"],
        D_KS_Poisson=z["D_KS_Poisson"],
        zeta_score=z["zeta_score"],
    )


def realized_vol(R_future: np.ndarray, annualize: bool = True) -> float:
    """
    Volatilite realisee de l'indice equally-weighted sur la fenetre future.
    R_future : (T_f, N)
    """
    idx_returns = R_future.mean(axis=1)   # equally-weighted "marche"
    sigma_daily = idx_returns.std(ddof=1)
    if annualize:
        return float(sigma_daily * np.sqrt(252))
    return float(sigma_daily)


# ============================================================
# 6.  AUTOTEST
# ============================================================
if __name__ == "__main__":
    print("Test du module rmt_zeta.py\n")
    # Matrice de Wishart pure (toute en bruit MP)
    rng = np.random.default_rng(42)
    T_test, N_test = 800, 60
    X = rng.standard_normal((T_test, N_test))
    R = X
    C = correlation_matrix(R)
    eig, _ = eigen_spectrum(C)
    print(f"Wishart pur : top 5 eig = {np.round(eig[:5], 3)}")
    print(f"MP bounds   = {mp_bounds(T_test, N_test)}")
    feats = rmt_features(R)
    for k, v in feats.items():
        print(f"  {k:25s}: {v:.4f}")

    print("\nMatrice avec un facteur cache (signal + bruit) :")
    f = rng.standard_normal((T_test, 1))
    loadings = rng.uniform(0.5, 1.0, N_test).reshape(1, -1)
    R2 = f @ loadings + 0.5 * X
    feats2 = rmt_features(R2)
    for k, v in feats2.items():
        print(f"  {k:25s}: {v:.4f}")
