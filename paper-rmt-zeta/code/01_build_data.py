"""
01_build_data.py
================
Generation d'un dataset multi-asset (N=60 series, T~4000 jours) calibre sur
des stats realistes 2010-2026 :
  - 30 actions large-cap US (sectors : tech, finance, healthcare, energy, industrials, consumer)
  - 12 paires FX (G10 + EM)
  - 10 commodities (energy, metals, agricultural)
  - 8 taux souverains (yields)

Structure factorielle :
  - 1 facteur "marche" (loading ~ 1 partout, eigenvalue dominant ~ 0.4 * N)
  - 4 facteurs "secteurs"
  - Innovations Student-t (df=6) pour generer des queues epaisses
  - 5 regimes de volatilite-correlation (chaine de Markov sticky)
    R0=calme, R1=normal, R2=stress, R3=crise (COVID-style), R4=recovery

Output: data/returns.csv (T x N), data/regimes.csv (T x 1), data/meta.csv (N x metadata)
"""

import os
import numpy as np
import pandas as pd

OUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(OUT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

RNG = np.random.default_rng(seed=20260527)

# ----------------------------------------------------------------------
# 1. UNIVERS (60 series)
# ----------------------------------------------------------------------
ASSETS = []
# Equities US (30) -- 6 sectors x 5 names
SECTORS_EQ = ["TECH", "FIN", "HEALTH", "ENRG", "INDU", "CONS"]
for sec in SECTORS_EQ:
    for k in range(5):
        ASSETS.append({"ticker": f"EQ_{sec}_{k+1}", "asset_class": "EQUITY",
                       "sector": sec, "base_vol_ann": 0.22, "df_t": 6})

# FX (12) -- 6 G10 + 6 EM
FX_G10 = ["EURUSD", "JPYUSD", "GBPUSD", "AUDUSD", "CADUSD", "CHFUSD"]
FX_EM  = ["BRLUSD", "ZARUSD", "TRYUSD", "MXNUSD", "CNYUSD", "KRWUSD"]
for t in FX_G10:
    ASSETS.append({"ticker": f"FX_{t}", "asset_class": "FX", "sector": "FX_G10",
                   "base_vol_ann": 0.09, "df_t": 7})
for t in FX_EM:
    ASSETS.append({"ticker": f"FX_{t}", "asset_class": "FX", "sector": "FX_EM",
                   "base_vol_ann": 0.16, "df_t": 5})

# Commodities (10)
COMMOS = [("WTI", "ENERGY", 0.35, 5), ("BRENT", "ENERGY", 0.34, 5),
          ("NATGAS", "ENERGY", 0.55, 4),
          ("GOLD", "METAL", 0.16, 6), ("SILVER", "METAL", 0.27, 5),
          ("COPPER", "METAL", 0.24, 6),
          ("WHEAT", "AGRI", 0.30, 5), ("CORN", "AGRI", 0.28, 5),
          ("SUGAR", "AGRI", 0.33, 5), ("COCOA", "AGRI", 0.36, 5)]
for sym, sec, v, d in COMMOS:
    ASSETS.append({"ticker": f"COM_{sym}", "asset_class": "COMMO", "sector": sec,
                   "base_vol_ann": v, "df_t": d})

# Rates (8) -- yields changes
RATES = [("UST2Y", 0.012), ("UST10Y", 0.011), ("UST30Y", 0.011),
         ("BUND10Y", 0.009), ("JGB10Y", 0.005),
         ("GILT10Y", 0.011), ("BTP10Y", 0.014), ("EMBI", 0.015)]
for sym, v in RATES:
    ASSETS.append({"ticker": f"RT_{sym}", "asset_class": "RATE", "sector": "RATE",
                   "base_vol_ann": v, "df_t": 7})

N = len(ASSETS)
print(f"Univers : {N} actifs")
for sec in set(a["sector"] for a in ASSETS):
    n = sum(1 for a in ASSETS if a["sector"] == sec)
    print(f"  {sec:10s}: {n}")

# ----------------------------------------------------------------------
# 2. STRUCTURE FACTORIELLE
# ----------------------------------------------------------------------
# Factor 0 : MARKET (charge tous les actifs, positivement sauf rates qui
#            chargent negativement = flight-to-quality)
# Factor 1 : EQ_TECH_SECTOR (charge actions tech + un peu marche)
# Factor 2 : COMMO (charge commos + USD)
# Factor 3 : EM (charge FX_EM + EMBI + un peu commos)
# Factor 4 : RATE_CURVE (charge rates de facon parallele)
K = 5

def build_loadings():
    B = np.zeros((N, K))
    for i, a in enumerate(ASSETS):
        cls, sec = a["asset_class"], a["sector"]
        # MARKET
        if cls == "EQUITY":   B[i, 0] = 0.85
        elif cls == "FX" and sec == "FX_EM": B[i, 0] = 0.55
        elif cls == "FX":     B[i, 0] = 0.25
        elif cls == "COMMO":  B[i, 0] = 0.40
        elif cls == "RATE":   B[i, 0] = -0.35   # flight to quality
        # TECH SECTOR
        if sec == "TECH": B[i, 1] = 0.55
        elif cls == "EQUITY": B[i, 1] = 0.12
        # COMMO
        if cls == "COMMO": B[i, 2] = 0.65
        elif sec == "ENRG" and cls == "EQUITY": B[i, 2] = 0.35
        elif sec == "FX_EM": B[i, 2] = 0.20
        # EM
        if sec == "FX_EM": B[i, 3] = 0.55
        elif a["ticker"] == "RT_EMBI": B[i, 3] = 0.50
        elif cls == "COMMO" and sec == "AGRI": B[i, 3] = 0.20
        # RATE_CURVE
        if cls == "RATE" and a["ticker"] != "RT_EMBI": B[i, 4] = 0.80
    return B

B = build_loadings()

# ----------------------------------------------------------------------
# 3. REGIMES DE VOLATILITE & CORRELATION
# ----------------------------------------------------------------------
# Pour chaque regime r : multiplicateur du facteur market (rho)
#                       multiplicateur de la vol idio
#                       presence de "contagion" (heavy tail trigger)
REGIMES = {
    0: dict(name="CALME",    f0_scale=0.5, sig_idio_scale=0.7, jump_p=0.000),
    1: dict(name="NORMAL",   f0_scale=1.0, sig_idio_scale=1.0, jump_p=0.001),
    2: dict(name="STRESS",   f0_scale=1.6, sig_idio_scale=1.3, jump_p=0.003),
    3: dict(name="CRISE",    f0_scale=2.5, sig_idio_scale=1.8, jump_p=0.010),
    4: dict(name="RECOVERY", f0_scale=1.2, sig_idio_scale=1.1, jump_p=0.002),
}

# Matrice de transition Markov (sticky)
P = np.array([
    [0.985, 0.013, 0.001, 0.000, 0.001],  # CALME ->
    [0.010, 0.970, 0.015, 0.001, 0.004],  # NORMAL ->
    [0.000, 0.020, 0.940, 0.030, 0.010],  # STRESS ->
    [0.000, 0.005, 0.040, 0.940, 0.015],  # CRISE ->
    [0.005, 0.025, 0.020, 0.000, 0.950],  # RECOVERY ->
])

# ----------------------------------------------------------------------
# 4. SIMULATION
# ----------------------------------------------------------------------
T = 4000   # ~16 ans business days
START = pd.Timestamp("2010-01-04")

# Pour mimer le 16-ans : forcer regime de crise vers ~indices 2500-2600 (~COVID 2020)
# et stress vers ~indices 1500 (~Taper Tantrum 2013) et 1800 (~Devaluation Yuan 2015)
regimes = np.zeros(T, dtype=int)
state = 1  # NORMAL au depart

# Forcer quelques "evenements" deterministes pour realisme
EVENTS = {
    1500: 2,   # Taper tantrum 2013
    1800: 2,   # 2015 China
    2570: 3,   # COVID 2020 March
    2570+120: 4,  # Recovery 2020 mid
    3200: 2,   # Energy crisis 2022
    3600: 1,   # Disinflation 2023
}

for t in range(T):
    if t in EVENTS:
        state = EVENTS[t]
    else:
        state = RNG.choice(5, p=P[state])
    regimes[t] = state

# Frequence des regimes
print("\nFrequence des regimes :")
for r in range(5):
    pct = (regimes == r).mean() * 100
    print(f"  R{r} ({REGIMES[r]['name']:8s}): {pct:5.1f}%")

# Pour les loadings : ajouter une dependance regime
def loadings_regime(B_base, r):
    # En regime de stress/crise, la 1ere colonne (marche) s'amplifie
    sc = REGIMES[r]["f0_scale"]
    B_r = B_base.copy()
    B_r[:, 0] *= np.sqrt(sc)  # eigenvalue augmente comme sc
    return B_r

# Vol idio annualisee (ramenee daily)
sig_idio_daily = np.array([a["base_vol_ann"] for a in ASSETS]) / np.sqrt(252) * 0.35
df_idio = np.array([a["df_t"] for a in ASSETS])

# Vol facteurs (annualisee)
sig_factor_daily = np.array([0.014, 0.010, 0.013, 0.012, 0.005])
# Student-t df pour facteurs
df_factor = np.array([6, 7, 6, 5, 8])

# Initialisation
R = np.zeros((T, N))

for t in range(T):
    r = regimes[t]
    R_reg = REGIMES[r]
    # Facteurs (Student-t standardise)
    F = RNG.standard_t(df_factor, size=K) * sig_factor_daily * R_reg["f0_scale"]
    # Cas particulier : factor 0 = market
    B_r = loadings_regime(B, r)
    common = B_r @ F  # (N,)
    # Idiosyncratiques
    eps = RNG.standard_t(df_idio, size=N) * sig_idio_daily * R_reg["sig_idio_scale"]
    # Jumps en queue (rare events)
    if RNG.random() < R_reg["jump_p"]:
        # 1 actif au hasard avec un saut large
        i = RNG.integers(N)
        sign = RNG.choice([-1, 1])
        eps[i] += sign * sig_idio_daily[i] * 8  # 8-sigma jump
    R[t] = common + eps

# ----------------------------------------------------------------------
# 5. DATAFRAMES + SAUVEGARDE
# ----------------------------------------------------------------------
dates = pd.bdate_range(start=START, periods=T)
df_R = pd.DataFrame(R, index=dates, columns=[a["ticker"] for a in ASSETS])
df_R.index.name = "Date"

df_meta = pd.DataFrame(ASSETS)
df_reg = pd.DataFrame({"regime": regimes,
                        "regime_name": [REGIMES[r]["name"] for r in regimes]},
                       index=dates)
df_reg.index.name = "Date"

df_R.to_csv(os.path.join(DATA_DIR, "returns.csv"), float_format="%.6e")
df_reg.to_csv(os.path.join(DATA_DIR, "regimes.csv"))
df_meta.to_csv(os.path.join(DATA_DIR, "meta.csv"), index=False)

# ----------------------------------------------------------------------
# 6. SANITY CHECKS
# ----------------------------------------------------------------------
print("\n=== Sanity checks ===")
print(f"Shape returns : {df_R.shape}")
print(f"Periode       : {df_R.index[0].date()} -> {df_R.index[-1].date()}")
print(f"Vol ann moy   : {df_R.std().mean()*np.sqrt(252)*100:.1f}%")
print(f"Vol ann med   : {df_R.std().median()*np.sqrt(252)*100:.1f}%")
print(f"Kurt moyenne  : {df_R.kurtosis().mean():.2f}")
print(f"Skew moyenne  : {df_R.skew().mean():.2f}")
print(f"Max |return|  : {df_R.abs().values.max()*100:.2f}%")

# Eigenvalues sur toute la periode
C = df_R.corr().values
ev = np.sort(np.linalg.eigvalsh(C))[::-1]
print(f"\nTop 5 eigenvalues (full sample) : {np.round(ev[:5], 2)}")
Q = T / N
lambda_plus  = (1 + 1/np.sqrt(Q))**2
lambda_minus = (1 - 1/np.sqrt(Q))**2
print(f"MP bounds [lambda-, lambda+] = [{lambda_minus:.3f}, {lambda_plus:.3f}]")
print(f"# eigenvalues > lambda+ : {np.sum(ev > lambda_plus)}")
print(f"Variance expliquee par lambda_max : {ev[0]/N*100:.1f}%")
