# Mémoire principal — Apprentissage Profond Robuste au Tail Risk

Manuscrit complet de 71 pages organisé en 6 chapitres, avec 25 figures et 8 tableaux.

## Contenu

- [`Reda_Mikou_These_FX_EM.pdf`](Reda_Mikou_These_FX_EM.pdf) — Le PDF compilé final
- [`these_complete.tex`](these_complete.tex) — Source LaTeX (~2100 lignes)
- [`code/`](code/) — Scripts Python
- [`figures/`](figures/) — 25 figures PNG (200 DPI)

## Structure du document

| Chap | Titre | Pages | Idée centrale |
|---|---|---|---|
| 1 | Cadre théorique macro-financier | ~10 | VAR(5), GJR-GARCH-t, identification de régimes |
| 2 | Microstructure et architecture du moteur | ~13 | Kyle's λ, Glosten-Milgrom, vue d'ensemble système |
| 3 | Pricing et modélisation des sous-jacents | ~12 | SABR, Nelson-Siegel dynamique, LSTM/Transformer |
| 4 | Market-Making sur FX et options FX EM | ~11 | Avellaneda-Stoikov vs agent PPO (Actor-Critic) |
| 5 | Hedging cross-produit FX-Options-Souverain | ~12 | Delta-DV01 vs Deep Hedging Buehler |
| 6 | Synthèse, limites et perspectives | ~8 | Validation H1-H4, architecture intégrée |

## Reproduction des figures

```bash
cd code
python build_real_data.py        # Reconstruit fx_real_2010_2026.csv (Brownian bridge sur ancres)
python generate_figures.py       # Génère les 25 figures PNG dans ../figures/
```

## Recompilation du PDF

```bash
pdflatex these_complete.tex
pdflatex these_complete.tex     # Deuxième passe pour les références croisées
```

Dépendances LaTeX : `amsmath`, `booktabs`, `natbib`, `cleveref`, `fancyhdr`, `hyperref`, `microtype`, `tcolorbox`, `listings`.

## Notes méthodologiques

- **Reconstruction des données** : Les CSV de marché ne sont pas tirés en direct (sandbox sans réseau). Ils sont reconstruits via interpolation Brownian bridge avec bruit Student-t (df=4) sur des ancres historiques réelles (year-ends, crashes 2013/2015/2020/2022). Les distributions résultantes ont kurtosis 7-17, en ligne avec l'empirie.
- **Injection de facteur commun** : Un facteur F ~ Student-t(df=5) est injecté avec des loadings calibrés (BRL +0.07, ZAR +0.08, TRY +0.04, EMB -0.45, TNX +0.45) pour reproduire les corrélations cross-asset observées (0.18 FX-Spread, 0.34 Spread-Rate).
- **Validation** : Les 4 hypothèses H1-H4 sont testées par comparaison directe baseline vs challenger sur le même jeu out-of-sample.
