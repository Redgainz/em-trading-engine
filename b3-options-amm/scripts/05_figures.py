"""
05_figures.py
=============
Genere les figures du module dans b3-options-amm/figures/ (PNG 160 dpi).
Prerequis : scripts 01 -> 04 executes (lit data/*.csv).

  fig_01_smiles.png        smiles B3 reels (points) vs SVI calibre, par echeance
  fig_02_session.png       une seance : spot + fills, PnL vs edge cumule, inventaire vega
  fig_03_iterations.png    PnL quotidien cumule du desk, v0 -> v5
  fig_04_attribution.png   decomposition du PnL de v5 (edge -> net)
  fig_05_markouts.png      courbes de markout par contrat, brut vs delta-couvert
  fig_06_retention.png     PnL retention et Sharpe par version
  fig_07_walk_forward.png  score train vs test des configurations candidates
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amm import calibration, data  # noqa: E402
from amm.kpi import MARKOUT_HORIZONS_S  # noqa: E402
from amm.runner import run_detailed  # noqa: E402
from amm.setup_day import load_ticker_context  # noqa: E402

FIG_DIR = os.path.join(data.MODULE_ROOT, "figures")
# Palette de reference (dataviz) : categoriel valide CVD, rampe sequentielle bleue
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SEQ = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
GOOD, CRITICAL = "#0ca30c", "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 2,
})


def save(fig, name: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, name), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def dodge_labels(ends: dict[str, float], min_gap: float) -> dict[str, float]:
    """Positions verticales d'etiquettes de fin de courbe, ecartees d'au moins min_gap."""
    placed: dict[str, float] = {}
    last = -np.inf
    for name, value in sorted(ends.items(), key=lambda kv: kv[1]):
        last = max(value, last + min_gap)
        placed[name] = last
    return placed


def fig_smiles(tickers: list[str]) -> None:
    fig, axes = plt.subplots(1, len(tickers), figsize=(5.2 * len(tickers), 3.8))
    for ax, tk in zip(np.atleast_1d(axes), tickers):
        ctx = load_ticker_context(tk)
        for color, sm in zip(CAT, ctx.surface.smiles):
            pts = ctx.iv_points[ctx.iv_points["expiry"] == sm.expiry]
            k = np.linspace(pts["k"].min(), pts["k"].max(), 200)
            ax.scatter(pts["k"], pts["iv"] * 100, s=12, color=color, alpha=0.55, linewidths=0)
            ax.plot(k, sm.svi.implied_vol(k, sm.T_snapshot) * 100, color=color,
                    label=f"{sm.expiry:%d/%m}  ({sm.n_points} pts, RMSE {sm.rmse_vol * 100:.1f} pt)")
        ax.legend(frameon=False, fontsize=8, title="echeance", title_fontsize=8, loc="upper center")
        ax.set_title(f"{tk} - smiles B3 du {ctx.surface.valuation_date:%d/%m/%Y}")
        ax.set_xlabel("log-moneyness ln(K/F)")
        ax.set_ylabel("vol implicite (%)")
    fig.text(0.01, -0.03, "Points : derniers prix reels (opcoes.net.br), options OTM traitees le jour. "
             "Courbes : SVI raw calibre (quasi-explicite).", fontsize=8, color=INK_2)
    save(fig, "fig_01_smiles.png")


def fig_session(cfg, ticker: str) -> None:
    ctx = load_ticker_context(ticker)
    day = ctx.days[-1]
    res, m = run_detailed(ticker, day, cfg)
    s, f = res.samples, res.fills
    minutes = s["t"] / 60
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True, gridspec_kw={"height_ratios": [2, 1.4, 1]})
    ax = axes[0]
    ax.plot(np.arange(m.n_seconds) / 60, m.S, color=INK_2, linewidth=1)
    ax.set_title(f"{ticker} {day:%d/%m/%Y} - {cfg.strategy.name} : {len(f)} fills sur {m.book.n} options, "
                 f"{len(res.hedges)} hedges")
    ax.set_ylabel("spot reel (R$)")
    edge_cum = (f["we_side"] * (f["mid"] - f["price"]) * f["size"] * 100).groupby(f["t"] // 10 * 10).sum()
    edge_cum = edge_cum.reindex(s["t"], fill_value=0).cumsum()
    axes[1].plot(minutes, edge_cum.to_numpy(), color=CAT[2], label="edge capture cumule")
    axes[1].plot(minutes, s["pnl_mid"], color=CAT[0], label="PnL net (MTM mid)")
    axes[1].set_ylabel("R$")
    axes[1].legend(loc="upper left", frameon=False)
    axes[2].plot(minutes, s["vega_volpt"], color=CAT[0])
    axes[2].axhline(0, color=INK_2, linewidth=0.8)
    axes[2].set_ylabel("vega (R$/pt)")
    axes[2].set_xlabel("minutes depuis l'ouverture (10:00)")
    save(fig, "fig_02_session.png")


def fig_iterations(rows: pd.DataFrame) -> None:
    daily = rows.groupby(["config", "day"], sort=False)["pnl_mtm"].sum().unstack("config")
    order = [c for c in calibration.all_versions() if c in daily.columns]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(daily))
    ends = {name: daily[name].sum() / 1000 for name in order}
    label_y = dodge_labels(ends, min_gap=0.045 * max(ends.values()))
    for color, name in zip(SEQ, order):
        cum = daily[name].cumsum().to_numpy() / 1000
        ax.plot(x, cum, color=color, linewidth=2.4 if name == order[-1] else 1.8)
        ax.annotate(f"{name}  {cum[-1]:.0f}k", (x[-1], cum[-1]), xytext=(x[-1] + 0.6, label_y[name]),
                    textcoords="data", va="center", fontsize=8.5, color=INK)
    ax.set_xticks(x[::3], [f"{d:%d/%m}" for d in daily.index[::3]])
    ax.set_ylabel("PnL cumule du desk (milliers R$)")
    ax.set_title("Iteration du modele : PnL cumule PETR4 + VALE3, meme marche pour chaque version")
    ax.set_xlim(0, len(x) + 3)
    save(fig, "fig_03_iterations.png")


def fig_attribution(summary: pd.DataFrame, version: str) -> None:
    r = summary.loc[version]
    steps = [("edge capture", r["edge"]), ("frais B3", -r["fees"]), ("spread hedge", -r["hedge_cost"]),
             ("delta", r["attr_delta"]), ("gamma", r["attr_gamma"]), ("vega", r["attr_vega"]),
             ("theta", r["attr_theta"]), ("intra-echantillon (<10 s)", r["attr_residual"] + r["attr_timing"])]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    level = 0.0
    for j, (label, value) in enumerate(steps):
        color = CAT[2] if j == 0 else (GOOD if value >= 0 else CRITICAL)
        ax.bar(j, value / 1000, bottom=level / 1000, color=color, width=0.62)
        ax.annotate(f"{value / 1000:+.1f}k", (j, (level + max(value, 0)) / 1000), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8, color=INK)
        level += value
    ax.bar(len(steps), r["pnl_mtm"] / 1000, color=CAT[0], width=0.62)
    ax.annotate(f"{r['pnl_mtm'] / 1000:.1f}k", (len(steps), r["pnl_mtm"] / 1000), xytext=(0, 3),
                textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(range(len(steps) + 1), [s[0] for s in steps] + ["PnL net"], rotation=20, ha="right")
    ax.set_ylim(0, r["edge"] / 1000 * 1.12)
    ax.set_ylabel("milliers R$")
    ax.set_title(f"Attribution du PnL - {version} (desk, toutes seances) : "
                 f"retention {r['retention']:.0%}")
    save(fig, "fig_04_attribution.png")


def fig_markouts(rows: pd.DataFrame, versions: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.array([0, *MARKOUT_HORIZONS_S])
    for color, name in zip(CAT, versions):
        g = rows[rows["config"] == name]
        contracts = g["contracts"].sum()
        raw = [g["edge"].sum() / contracts] + [g[f"mk_{h}s"].sum() / contracts for h in MARKOUT_HORIZONS_S]
        hedged = [g["edge"].sum() / contracts] + [g[f"mkh_{h}s"].sum() / contracts for h in MARKOUT_HORIZONS_S]
        ax.plot(x + 1, raw, color=color, marker="o", markersize=4)
        ax.plot(x + 1, hedged, color=color, linestyle="--", linewidth=1.4)
        ax.annotate(f"{name} brut", (x[-1] + 1, raw[-1]), xytext=(5, 0), textcoords="offset points",
                    fontsize=8, va="center")
        ax.annotate(f"{name} couvert", (x[-1] + 1, hedged[-1]), xytext=(5, 0), textcoords="offset points",
                    fontsize=8, va="center", color=INK_2)
    ax.set_xscale("log")
    ax.set_xticks(x + 1, ["fill"] + [f"{h}s" for h in MARKOUT_HORIZONS_S])
    ax.set_ylabel("R$ par contrat")
    ax.set_title("Markouts : l'edge brut decroit (toxicite directionnelle), le delta-couvert tient")
    ax.set_xlim(0.8, 1500)
    save(fig, "fig_05_markouts.png")


def fig_retention(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    names = list(summary.index)
    y = np.arange(len(names))
    for ax, col, fmt, title in ((axes[0], "retention", "{:.0%}", "PnL retention (PnL net / edge)"),
                                (axes[1], "sharpe_ann", "{:.1f}", "Sharpe annualise (PnL quotidien)")):
        vals = summary[col].to_numpy()
        ax.barh(y, vals, color=SEQ[: len(names)], height=0.6)
        for yi, v in zip(y, vals):
            ax.annotate(fmt.format(v), (v, yi), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    save(fig, "fig_06_retention.png")


def fig_walk_forward() -> None:
    train_p, test_p = (os.path.join(data.DATA_DIR, f) for f in ("calibration_train.csv", "calibration_test.csv"))
    if not (os.path.exists(train_p) and os.path.exists(test_p)):
        return
    train = calibration.score(pd.read_csv(train_p, parse_dates=["day"]))
    test = calibration.score(pd.read_csv(test_p, parse_dates=["day"]))
    with open(calibration.CALIBRATED_PATH) as fh:
        selected = json.load(fh)["meta"]["selected"]
    both = train[["ce"]].join(test[["ce"]], lsuffix="_train", rsuffix="_test", how="inner")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.scatter(both["ce_train"] / 1000, both["ce_test"] / 1000, s=40, color=CAT[0])
    ax.axline((0, 0), slope=1, color=INK_2, linewidth=0.8, linestyle=":")
    for name, row in both.iterrows():
        tag = "retenu" if name == selected else ("v4 (base)" if name == calibration.BASE_VERSION else "")
        if tag:
            ax.annotate(tag, (row["ce_train"] / 1000, row["ce_test"] / 1000), xytext=(6, 4),
                        textcoords="offset points", fontsize=8.5)
    ax.set_xlabel("equivalent certain train (k R$/jour)")
    ax.set_ylabel("equivalent certain test (k R$/jour)")
    ax.set_title("Walk-forward : selection sur train, verification hors echantillon")
    fig.text(0.01, -0.04, f"{len(train)} configurations evaluees sur le train ; les {len(both) - 1} meilleures + v4 "
             "rejouees sur le test. Pointille : test = train.", fontsize=8, color=INK_2)
    save(fig, "fig_07_walk_forward.png")


def main() -> None:
    tickers = data.load_market_meta()["tickers"]
    rows = pd.read_csv(os.path.join(data.DATA_DIR, "results_days.csv"), parse_dates=["day"])
    summary = pd.read_csv(os.path.join(data.DATA_DIR, "results_summary.csv"), index_col=0)
    versions = calibration.all_versions()
    final = list(versions)[-1]
    print("Figures :")
    fig_smiles(tickers)
    fig_session(versions[final], tickers[0])
    fig_iterations(rows)
    fig_attribution(summary, final)
    fig_markouts(rows, ["v0_naive", final])
    fig_retention(summary)
    fig_walk_forward()


if __name__ == "__main__":
    main()
