"""
events.py
=========
Generation et execution des evenements mensuels du desk :
  - RFQ Spot     : client veut acheter/vendre FX EM
  - RFQ Option   : client veut acheter une option ATM
  - News flash   : info de marche (biaise le mois suivant)

Chaque RFQ peut etre acceptee ou refusee par l'utilisateur.
Si acceptee, le market-maker prend la position INVERSE au prix du client
et encaisse le spread.
"""
from __future__ import annotations
import dataclasses
import numpy as np
import pandas as pd
from typing import Optional


CLIENTS_INST = [
    "Banco Itaú",          "Banco Bradesco",      "BBVA Trading",
    "Goldman EM Desk",     "Citi EM FX",          "BNP Paribas EM",
    "HSBC Emerging",       "Magnetar Hedge Fund", "Brevan Howard EM",
    "Citadel Macro",       "Millennium FX",       "Petrobras Treasury",
    "Sasol Treasury",      "Türkiye Iş Bankası",  "Standard Bank ZAR",
    "Aberdeen EM Bond",    "Pimco EM",            "BlackRock EM Equity",
]

NEWS_TEMPLATES = [
    ("BCB surprise rate hike +75 bps",      "BRL", +0.012),   # drift positif BRL
    ("BCB surprise rate cut −50 bps",       "BRL", -0.010),
    ("SARB hawkish forward guidance",       "ZAR", +0.008),
    ("Tensions politiques Turquie",         "TRY", -0.020),
    ("CBRT raise rates +500 bps",           "TRY", +0.025),
    ("Petrobras dividend payout boost",     "BRL", +0.005),
    ("S&P downgrades South Africa",         "ZAR", -0.015),
    ("Fed dovish surprise",                  "BRL", +0.006),  # tous EM rallient
    ("Fed dovish surprise",                  "ZAR", +0.006),
    ("Fed dovish surprise",                  "TRY", +0.008),
    ("Reuters: massive carry-trade unwind", "TRY", -0.030),
    ("Brazil Real GDP beats expectations",  "BRL", +0.010),
]


@dataclasses.dataclass
class MarketEvent:
    """Un evenement de marche que l'utilisateur doit traiter."""
    event_id: str
    kind: str                    # "RFQ_SPOT", "RFQ_OPTION", "NEWS"
    client_name: str
    description: str             # Pour affichage

    # Champs RFQ (sinon None)
    instrument: Optional[str] = None
    asset_class: Optional[str] = None
    client_side: Optional[str] = None   # BUY ou SELL (cote client)
    quantity: float = 0.0
    notional_usd: float = 0.0           # Notional USD du deal
    market_mid: float = 0.0
    client_price: float = 0.0
    spread_bps: float = 0.0
    expected_pnl_spread: float = 0.0    # gain immediat si accepte

    # Champs option supplementaires
    strike: Optional[float] = None
    maturity_years: Optional[float] = None
    option_kind: Optional[str] = None
    underlying: Optional[str] = None
    implied_vol: Optional[float] = None

    # Champs news
    news_currency: Optional[str] = None
    news_drift_impact: float = 0.0      # appliquer au mois suivant

    @property
    def is_actionable(self) -> bool:
        return self.kind != "NEWS"

    @property
    def mm_side(self) -> str:
        """Cote du market-maker (oppose au client)."""
        if self.client_side == "BUY":
            return "SELL"
        elif self.client_side == "SELL":
            return "BUY"
        return ""


# ============================================================
# Generateurs d'evenements
# ============================================================
def generate_spot_rfq(rng: np.random.Generator, market_state: dict,
                       regime_idx: int, event_idx: int) -> MarketEvent:
    """Genere une RFQ spot."""
    # Tirer un actif FX EM (BRL/ZAR/TRY plus probable)
    instruments = ["BRL/USD", "ZAR/USD", "TRY/USD",
                    "MXN/USD", "CNY/USD", "KRW/USD"] \
                    if False else ["BRL/USD", "ZAR/USD", "TRY/USD"]  # focus EM
    instrument = rng.choice(instruments)
    mid = market_state["spot"][instrument]

    # Notional client : 1M a 10M USD
    notional_usd = float(rng.uniform(1_000_000, 10_000_000))
    # Quantite dans devise foreign (BRL): notional / spot
    quantity = notional_usd / mid

    # Side aleatoire
    client_side = rng.choice(["BUY", "SELL"])

    # Spread depend du regime (CALME 5-15 bps, NORMAL 10-25, CRISE 20-50)
    if regime_idx == 0:
        spread = float(rng.uniform(5, 15))
    elif regime_idx == 1:
        spread = float(rng.uniform(8, 25))
    else:
        spread = float(rng.uniform(15, 50))

    # Prix client = mid + (signe) * spread
    if client_side == "BUY":
        client_price = mid * (1 + spread / 10_000)  # client paie + cher
    else:
        client_price = mid * (1 - spread / 10_000)  # client recoit - cher

    expected_pnl = abs(client_price - mid) * quantity

    client = rng.choice(CLIENTS_INST)
    desc = (f"Demande un quote pour {client_side} "
            f"{notional_usd/1e6:.1f}M USD de {instrument}.")

    return MarketEvent(
        event_id=f"evt_{event_idx}",
        kind="RFQ_SPOT",
        client_name=client,
        description=desc,
        instrument=instrument,
        asset_class="FX",
        client_side=client_side,
        quantity=quantity,
        notional_usd=notional_usd,
        market_mid=mid,
        client_price=client_price,
        spread_bps=spread,
        expected_pnl_spread=expected_pnl,
    )


def generate_option_rfq(rng: np.random.Generator, market_state: dict,
                          regime_idx: int, event_idx: int) -> MarketEvent:
    """Genere une RFQ d'option."""
    from .market_state import OPTION_TO_UNDERLYING

    opt_ticker = rng.choice(list(OPTION_TO_UNDERLYING.keys()))
    underlying, kind, T = OPTION_TO_UNDERLYING[opt_ticker]
    mid_opt = market_state["spot"][opt_ticker]
    iv = market_state["iv"][opt_ticker]
    S = market_state["spot"][underlying]
    K = S * np.exp((market_state["r_d"] - market_state["r_f"][underlying.split("/")[0]]) * T)

    # Notional client : 5M a 15M USD
    notional_usd = float(rng.uniform(5_000_000, 15_000_000))
    quantity = notional_usd / S  # quantite en devise foreign

    # Spread vol depend regime (markup en bps de premium)
    if regime_idx == 0:
        markup = float(rng.uniform(3, 10))
    elif regime_idx == 1:
        markup = float(rng.uniform(8, 20))
    else:
        markup = float(rng.uniform(15, 40))

    # Client veut ACHETER l'option (cas le plus frequent en EM)
    # Le market-maker VEND donc l'option (cote SELL pour nous)
    client_side = "BUY"
    client_price = mid_opt * (1 + markup / 10_000)
    expected_pnl = abs(client_price - mid_opt) * quantity

    client = rng.choice(CLIENTS_INST)
    name_short = opt_ticker.replace("_ATM", " ATM")
    desc = (f"Veut ACHETER {notional_usd/1e6:.1f}M USD de {name_short}  "
            f"(IV {iv*100:.1f}%).")

    return MarketEvent(
        event_id=f"evt_{event_idx}",
        kind="RFQ_OPTION",
        client_name=client,
        description=desc,
        instrument=opt_ticker,
        asset_class="OPTION_FX",
        client_side=client_side,
        quantity=quantity,
        notional_usd=notional_usd,
        market_mid=mid_opt,
        client_price=client_price,
        spread_bps=markup,
        expected_pnl_spread=expected_pnl,
        strike=K,
        maturity_years=T,
        option_kind=kind,
        underlying=underlying,
        implied_vol=iv,
    )


def generate_news(rng: np.random.Generator, event_idx: int) -> MarketEvent:
    """Genere un evenement news (informationnel)."""
    title, ccy, drift = NEWS_TEMPLATES[int(rng.integers(len(NEWS_TEMPLATES)))]
    return MarketEvent(
        event_id=f"evt_{event_idx}",
        kind="NEWS",
        client_name="Marché",
        description=f"📰  {title}",
        news_currency=ccy,
        news_drift_impact=drift,
    )


def generate_monthly_events(rng: np.random.Generator, market_state: dict,
                              regime_idx: int, n_events: int | None = None) -> list[MarketEvent]:
    """
    Genere une liste d'evenements pour le mois en cours.

    Volume par regime :
      - CALME  : 1 event toujours
      - NORMAL : 1 event (proba 70%) ou 2 events (proba 30%)
      - CRISE  : 3 events toujours

    Type d'event :
      - RFQ_SPOT   : 55%
      - RFQ_OPTION : 30%
      - NEWS       : 15%
    """
    if n_events is None:
        if regime_idx == 0:
            n_events = 1
        elif regime_idx == 1:
            n_events = 1 if rng.random() < 0.70 else 2
        else:
            n_events = 3

    events = []
    for i in range(n_events):
        u = rng.random()
        if u < 0.55:
            ev = generate_spot_rfq(rng, market_state, regime_idx, i)
        elif u < 0.85:
            ev = generate_option_rfq(rng, market_state, regime_idx, i)
        else:
            ev = generate_news(rng, i)
        events.append(ev)
    return events


# ============================================================
# Execution d'une RFQ acceptee
# ============================================================
def execute_rfq_acceptance(portfolio, event: MarketEvent, date: pd.Timestamp,
                             cost_bps: float = 0.0):
    """
    Quand le market-maker accepte une RFQ :
      - Le client BUY  -> le MM SELL au prix client (cash + qty*price)
      - Le client SELL -> le MM BUY au prix client (cash - qty*price)

    Le MM encaisse le spread (price ecarte du mid en sa faveur).
    Aucun cout de transaction additionnel applique (event execute directement).
    """
    if event.kind not in ("RFQ_SPOT", "RFQ_OPTION"):
        raise ValueError(f"Cannot execute event of kind {event.kind}")

    if event.client_side == "BUY":
        # MM vend -> short l'instrument
        portfolio.sell(
            event.instrument, event.asset_class, event.quantity,
            event.client_price, date,
            strike=event.strike,
            maturity_years=event.maturity_years,
            option_kind=event.option_kind,
            underlying_ticker=event.underlying,
            cost_bps=0.0,
        )
    else:
        # MM achete -> long l'instrument
        portfolio.buy(
            event.instrument, event.asset_class, event.quantity,
            event.client_price, date,
            strike=event.strike,
            maturity_years=event.maturity_years,
            option_kind=event.option_kind,
            underlying_ticker=event.underlying,
            cost_bps=0.0,
        )

    # Marquer l'event comme execute dans l'historique
    portfolio.transaction_history.append(dict(
        date=date,
        action=f"RFQ_{event.client_side}",
        instrument=event.instrument,
        quantity=event.quantity if event.client_side == "SELL" else -event.quantity,
        price=event.client_price,
        client=event.client_name,
        spread_bps=event.spread_bps,
        pnl_spread=event.expected_pnl_spread,
        cash_after=portfolio.cash,
    ))
