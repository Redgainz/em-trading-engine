"""
data.py
=======
Acquisition et cache des donnees reelles B3.

Sources (toutes publiques, sans cle API) :
  - Yahoo Finance (yfinance) : barres 1 minute du sous-jacent (30 derniers jours max,
    8 jours par requete) + historique daily pour la vol realisee.
  - opcoes.net.br : chaine d'options B3 (strikes, dernier prix, nb de trades,
    volume financier) par echeance.
  - Banco Central do Brasil, SGS serie 1178 : Selic annualisee base 252.

Les reponses externes sont validees (colonnes, types) avant d'etre ecrites en CSV.
Le reste du pipeline ne lit que le cache local -> reproductible hors ligne / en CI.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from .conventions import TIMEZONE

MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(MODULE_ROOT, "data")

OPCOES_URL = ("https://opcoes.net.br/listaopcoes/completa?idAcao={ticker}"
              "&listarVencimentos={list_exp}&cotacoes=true{exp_param}")
BCB_SELIC_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1178/dados/ultimos/1?formato=json"
HTTP_TIMEOUT_S = 30
POLITE_DELAY_S = 2.0
BACKOFF_BASE_S = 15.0
USER_AGENT = "Mozilla/5.0 (research; em-trading-engine)"

CHAIN_COLUMNS = ["ticker", "fm", "type", "style", "moneyness", "strike", "dist_pct",
                 "premium_pct", "last", "n_trades", "volume_brl", "last_date"]


def _http_json(url: str, max_retries: int = 4) -> dict | list:
    """GET JSON poli : pause entre requetes + backoff exponentiel sur HTTP 429."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(max_retries + 1):
        time.sleep(POLITE_DELAY_S)
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as resp:  # noqa: S310
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == max_retries:
                raise RuntimeError(f"HTTP {exc.code} sur {url}") from exc
            wait = BACKOFF_BASE_S * 2 ** attempt
            print(f"    HTTP 429 (rate limit), nouvelle tentative dans {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"Echec apres {max_retries} tentatives : {url}")


# ----------------------------------------------------------------------
# Selic
# ----------------------------------------------------------------------
def fetch_selic() -> float:
    """Derniere Selic annualisee (fraction, ex 0.1365)."""
    payload = _http_json(BCB_SELIC_URL)
    if not isinstance(payload, list) or not payload or "valor" not in payload[-1]:
        raise RuntimeError(f"Reponse BCB inattendue : {payload!r}")
    value = float(payload[-1]["valor"]) / 100.0
    if not 0.0 < value < 1.0:
        raise RuntimeError(f"Selic hors plage plausible : {value}")
    return value


# ----------------------------------------------------------------------
# Chaine d'options B3
# ----------------------------------------------------------------------
def fetch_expiries(ticker: str) -> list[dict]:
    """Echeances listees : [{'expiry': '2026-11-19', 'du': 27, 'monthly': True}, ...]"""
    url = OPCOES_URL.format(ticker=ticker, list_exp="true", exp_param="")
    payload = _http_json(url)
    if not payload.get("success"):
        raise RuntimeError(f"opcoes.net.br a refuse la requete pour {ticker}")
    out = []
    for item in payload["data"]["vencimentos"]:
        attrs = item.get("dataAttributes", {})
        out.append({"expiry": item["value"], "du": int(attrs.get("du", 0)),
                    "monthly": attrs.get("m") == "1"})
    return out


def _parse_chain_rows(rows: list, expiry: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=CHAIN_COLUMNS + ["expiry"])
    width = len(CHAIN_COLUMNS)
    clean = [r[:width] for r in rows if isinstance(r, list) and len(r) >= width]
    df = pd.DataFrame(clean, columns=CHAIN_COLUMNS)
    df["ticker"] = df["ticker"].astype(str).str.split("_").str[0]
    for col in ("strike", "last", "n_trades", "volume_brl"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["n_trades"] = df["n_trades"].fillna(0).astype(int)
    df["volume_brl"] = df["volume_brl"].fillna(0.0)
    df["type"] = df["type"].str.upper()
    df["expiry"] = expiry
    df = df[df["type"].isin(["CALL", "PUT"]) & (df["strike"] > 0)]
    return df.drop(columns=["fm", "dist_pct", "premium_pct", "moneyness"])


def fetch_option_chain(ticker: str, expiries: list[str]) -> pd.DataFrame:
    """Chaine complete pour les echeances demandees (une requete par echeance)."""
    frames = []
    for expiry in expiries:
        url = OPCOES_URL.format(ticker=ticker, list_exp="false",
                                exp_param=f"&vencimentos={expiry}")
        payload = _http_json(url)
        if not payload.get("success"):
            raise RuntimeError(f"Chaine indisponible {ticker} {expiry}")
        frames.append(_parse_chain_rows(payload["data"]["cotacoesOpcoes"], expiry))
    chain = pd.concat(frames, ignore_index=True)
    return chain.drop_duplicates(subset="ticker").reset_index(drop=True)


# ----------------------------------------------------------------------
# Sous-jacent (Yahoo Finance)
# ----------------------------------------------------------------------
def fetch_intraday(ticker: str, n_days: int = 29) -> pd.DataFrame:
    """Barres 1 minute sur les n_days derniers jours calendaires (limite Yahoo : 30)."""
    import yfinance as yf  # import local : dependance optionnelle

    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=n_days)
    frames, cursor = [], start
    while cursor < end:
        chunk_end = min(cursor + dt.timedelta(days=7), end)
        df = yf.Ticker(f"{ticker}.SA").history(start=cursor, end=chunk_end, interval="1m")
        if not df.empty:
            frames.append(df[["Open", "High", "Low", "Close", "Volume"]])
        cursor = chunk_end
    if not frames:
        raise RuntimeError(f"Aucune barre 1m Yahoo pour {ticker}.SA")
    out = pd.concat(frames)
    out = out[~out.index.duplicated()].sort_index()
    out.index = out.index.tz_convert(TIMEZONE)
    return validate_bars(out)


def fetch_daily(ticker: str, period: str = "2y") -> pd.DataFrame:
    import yfinance as yf

    df = yf.Ticker(f"{ticker}.SA").history(period=period, interval="1d")
    if df.empty:
        raise RuntimeError(f"Aucune barre daily Yahoo pour {ticker}.SA")
    df.index = df.index.tz_convert(TIMEZONE)
    return validate_bars(df[["Open", "High", "Low", "Close", "Volume"]])


def validate_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Elimine les barres incoherentes (prix <= 0, High < Low, NaN)."""
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    ok = (df[["Open", "High", "Low", "Close"]] > 0).all(axis=1) & (df["High"] >= df["Low"])
    return df[ok]


# ----------------------------------------------------------------------
# Cache local
# ----------------------------------------------------------------------
def _path(name: str) -> str:
    return os.path.join(DATA_DIR, name)


def save_snapshot(ticker: str, intraday: pd.DataFrame, daily: pd.DataFrame,
                  chain: pd.DataFrame, selic: float, as_of: str) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    intraday.round(4).to_csv(_path(f"{ticker}_1m.csv"))
    daily.round(4).to_csv(_path(f"{ticker}_daily.csv"))
    chain.to_csv(_path(f"{ticker}_chain.csv"), index=False)
    meta_path = _path("market_meta.json")
    previous = load_market_meta() if os.path.exists(meta_path) else {}
    tickers = sorted(set(previous.get("tickers", [])) | {ticker})
    meta = {**previous, "selic": selic, "as_of": as_of, "tickers": tickers}
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)


def load_market_meta() -> dict:
    with open(_path("market_meta.json")) as fh:
        return json.load(fh)


def _load_bars(name: str) -> pd.DataFrame:
    df = pd.read_csv(_path(name), index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(TIMEZONE)
    return validate_bars(df)


def load_intraday(ticker: str) -> pd.DataFrame:
    return _load_bars(f"{ticker}_1m.csv")


def load_daily(ticker: str) -> pd.DataFrame:
    return _load_bars(f"{ticker}_daily.csv")


def load_chain(ticker: str) -> pd.DataFrame:
    chain = pd.read_csv(_path(f"{ticker}_chain.csv"))
    chain["last_date"] = pd.to_datetime(chain["last_date"], errors="coerce")
    chain["expiry"] = pd.to_datetime(chain["expiry"]).dt.date
    chain["is_call"] = chain["type"].eq("CALL")
    return chain


def trading_days(intraday: pd.DataFrame, min_bars: int = 300) -> list[dt.date]:
    """Seances completes presentes dans le cache 1 minute."""
    counts = intraday.groupby(intraday.index.date).size()
    return [d for d, n in counts.items() if n >= min_bars]


def realized_vol_before(daily: pd.DataFrame, day: dt.date, window: int = 20) -> float:
    """Vol realisee close-to-close annualisee (252) sur les `window` seances avant `day`."""
    closes = daily["Close"][daily.index.date < day].tail(window + 1)
    if len(closes) < window // 2:
        raise ValueError(f"Historique daily insuffisant avant {day}")
    rets = np.diff(np.log(closes.to_numpy()))
    return float(np.std(rets, ddof=1) * np.sqrt(252))
