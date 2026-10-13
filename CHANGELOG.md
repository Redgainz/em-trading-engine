# Changelog

## [1.1.0] - 2026-10-13

### Added
- `b3-options-amm/` : strategie de market-making haute frequence sur options actions B3 (PETR4, VALE3)
  - donnees reelles : Yahoo Finance 1 min, chaine B3 opcoes.net.br, Selic BCB, calendrier B3
  - nappe SVI calibree, simulateur de marche a la seconde, quoting Factor-GLFT, hedge delta
  - KPI (markouts, attribution greeks, PnL retention), iteration v0 -> v5, calibration walk-forward
  - 27 tests pytest, 7 figures
- Page Streamlit `B3 Options AMM`

## [1.0.0] - 2026-05-27

### Added
- Code complet du memoire de fin d'etudes (thesis/code/)
- Paper RMT-Zeta avec PDF, LaTeX, code numpy pur (paper-rmt-zeta/)
- Dashboard interactif HTML pour GitHub Pages (docs/)
- App Streamlit multi-pages (app/)
- CI workflow (lint + smoke test sur Python 3.10, 3.11, 3.12)
- Citation au format CFF
- 35 figures generees (25 these + 10 paper)

### Composants techniques
- 13 features RMT/Zeta implementees from scratch en numpy
- 6 modeles ML : Ridge, Random Forest, Gradient Boosting, MLP-Adam, MC k-NN
- Backtest rolling-window de 772 fenetres sur 16 ans
- Analyse conditionnelle par regime de marche
