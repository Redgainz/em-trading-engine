"""
conventions.py
==============
Conventions de marche B3 (Bolsa Brasil Balcao) pour les options sur actions.

  - Base 252 jours ouvres (convention bresilienne "dias uteis")
  - Taux Selic annuel effectif, capitalisation discrete : DF = (1 + r)^(-T)
  - Seance reguliere 10:00 - 16:55 (heure de Sao Paulo)
  - Tick option R$ 0.01, lot standard 100 actions par contrat
  - Strikes ajustes des proventos (dividendes + JCP) par la B3 :
    les options sont donc protegees du dividende -> pricing sans q.
  - Lettre du ticker : A-L = calls Jan-Dec, M-X = puts Jan-Dec
"""
from __future__ import annotations

import datetime as dt

import numpy as np

BUSINESS_DAYS_YEAR = 252
TICK = 0.01
CONTRACT_SIZE = 100
STOCK_LOT = 100
TIMEZONE = "America/Sao_Paulo"

SESSION_OPEN = dt.time(10, 0)
SESSION_CLOSE = dt.time(16, 55)
SESSION_SECONDS = (
    (SESSION_CLOSE.hour * 3600 + SESSION_CLOSE.minute * 60)
    - (SESSION_OPEN.hour * 3600 + SESSION_OPEN.minute * 60)
)

# Couts (ordre de grandeur B3 day-trade, hors exoneration Formador de Mercado)
OPTION_FEE_RATE = 0.00030     # emolumentos + registro, en fraction du premium
STOCK_FEE_RATE = 0.00005      # 0.5 bp sur le notionnel actions
STOCK_HALF_SPREAD = 0.005     # demi-spread typique PETR4 / VALE3 (1 tick = 0.01)

CALL_LETTERS = "ABCDEFGHIJKL"
PUT_LETTERS = "MNOPQRSTUVWX"

# Feriados nacionais B3 (pas de seance)
B3_HOLIDAYS = np.array([
    # 2026
    "2026-01-01", "2026-02-16", "2026-02-17", "2026-04-03", "2026-04-21",
    "2026-05-01", "2026-06-04", "2026-09-07", "2026-10-12", "2026-11-02",
    "2026-11-20", "2026-12-24", "2026-12-25", "2026-12-31",
    # 2027
    "2027-01-01", "2027-02-08", "2027-02-09", "2027-03-26", "2027-04-21",
    "2027-05-27", "2027-09-07", "2027-10-12", "2027-11-02", "2027-11-15",
    "2027-12-24", "2027-12-31",
    # 2028
    "2028-01-01", "2028-02-28", "2028-02-29", "2028-04-14", "2028-04-21",
    "2028-05-01", "2028-06-15", "2028-09-07", "2028-10-12", "2028-11-02",
    "2028-11-15", "2028-11-20", "2028-12-25",
], dtype="datetime64[D]")


def business_days_between(start: dt.date, end: dt.date) -> int:
    """Nombre de jours ouvres B3 dans ]start, end] (convention 'du')."""
    s = np.datetime64(start, "D") + 1
    e = np.datetime64(end, "D") + 1
    return int(np.busday_count(s, e, holidays=B3_HOLIDAYS))


def year_fraction(valuation: dt.date, expiry: dt.date,
                  intraday_elapsed: float | np.ndarray = 0.0) -> float | np.ndarray:
    """
    Maturite en annees base 252 a un instant de la seance.

    intraday_elapsed : fraction de la seance ecoulee (0 = ouverture, 1 = cloture).
    Une option expire a la cloture de son jour d'echeance : le jour courant
    compte pour (1 - intraday_elapsed) jour ouvre.
    """
    du_after_today = business_days_between(valuation, expiry)
    remaining_today = 1.0 - np.clip(intraday_elapsed, 0.0, 1.0)
    return (du_after_today + remaining_today) / BUSINESS_DAYS_YEAR


def continuous_rate(selic_annual: float) -> float:
    """Selic effective annuelle (ex: 0.1365) -> taux continu equivalent."""
    if selic_annual <= -1.0:
        raise ValueError(f"Selic invalide : {selic_annual}")
    return float(np.log1p(selic_annual))


def option_type_from_ticker(ticker: str) -> str:
    """'PETRK516' -> 'CALL' ; 'PETRW516' -> 'PUT' (5e caractere = lettre serie)."""
    letter = ticker[4].upper()
    if letter in CALL_LETTERS:
        return "CALL"
    if letter in PUT_LETTERS:
        return "PUT"
    raise ValueError(f"Lettre de serie inconnue dans {ticker!r}")


def round_to_tick(price: np.ndarray, side: str) -> np.ndarray:
    """Arrondi au tick : 'down' pour un bid, 'up' pour un ask (jamais en faveur du MM)."""
    scaled = np.asarray(price) / TICK
    if side == "down":
        return np.floor(scaled + 1e-9) * TICK
    if side == "up":
        return np.ceil(scaled - 1e-9) * TICK
    raise ValueError(f"side doit etre 'down' ou 'up', recu {side!r}")
