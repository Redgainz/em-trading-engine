"""
Téléchargement des données historiques réelles 2010-2026
pour la thèse de Reda Mikou (EDHEC MSc DAI).

Usage :
    1. pip install yfinance pandas
    2. python fetch_data.py
    3. Re-upload le fichier `fx_real_2010_2026.csv` dans Cowork.

Le script crée également des fichiers OHLC séparés (utiles pour Parkinson RV).
"""
import os
import sys
import time
import pandas as pd
import yfinance as yf

START = "2010-01-01"
END   = "2026-01-01"

# Univers complet utilisé dans les chapitres 1 à 6
TICKERS = {
    # Chapitre 1
    "CNY": "CNY=X",   # Yuan
    "KRW": "KRW=X",   # Won
    "MXN": "MXN=X",   # Peso mexicain
    # Chapitres 1, 3, 4, 5
    "BRL": "BRL=X",   # Real bresilien
    "ZAR": "ZAR=X",   # Rand
    "TRY": "TRY=X",   # Livre turque
    # Proxy taux / spread souverain
    "TNX": "^TNX",    # US 10Y (proxy)
    "EMB": "EMB",     # iShares JPM EM Bond
}

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_CLOSE = os.path.join(OUT_DIR, "fx_real_2010_2026.csv")
CSV_OHLC  = os.path.join(OUT_DIR, "fx_real_2010_2026_ohlc.csv")


def main():
    print(f"Telechargement des series {START} -> {END}\n")

    close_frames = {}
    ohlc_frames  = []

    for name, ticker in TICKERS.items():
        print(f"  > {name:5s} ({ticker})", end=" ... ", flush=True)
        try:
            df = yf.download(ticker,
                             start=START, end=END,
                             auto_adjust=False,
                             progress=False,
                             interval="1d")
            if df.empty:
                print("VIDE")
                continue

            # Pour le close
            close_frames[name] = df["Close"]

            # Pour OHLC : conserver toutes les colonnes prefixees par le nom
            df_ohlc = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df_ohlc.columns = [f"{name}_{c}" for c in df_ohlc.columns]
            ohlc_frames.append(df_ohlc)

            print(f"OK ({len(df)} obs)")
        except Exception as e:
            print(f"ECHEC : {e}")
        time.sleep(0.2)  # politesse

    # Consolidation Close
    if not close_frames:
        print("\nAucune donnee recuperee. Verifie ta connexion / yfinance.")
        sys.exit(1)

    df_close = pd.DataFrame(close_frames)
    df_close = df_close.asfreq("B", method="ffill").dropna(how="all")
    df_close.index.name = "Date"
    df_close.to_csv(CSV_CLOSE, float_format="%.6f")
    print(f"\nClose -> {CSV_CLOSE}  ({df_close.shape[0]} lignes, {df_close.shape[1]} series)")

    # Consolidation OHLC
    if ohlc_frames:
        df_ohlc_all = pd.concat(ohlc_frames, axis=1)
        df_ohlc_all = df_ohlc_all.asfreq("B", method="ffill").dropna(how="all")
        df_ohlc_all.index.name = "Date"
        df_ohlc_all.to_csv(CSV_OHLC, float_format="%.6f")
        print(f"OHLC  -> {CSV_OHLC}  ({df_ohlc_all.shape[0]} lignes, {df_ohlc_all.shape[1]} colonnes)")

    print("\nUploade maintenant les deux CSV dans Cowork.")
    print("Stats rapides :")
    print(df_close.describe().round(3).to_string())


if __name__ == "__main__":
    main()
