# Paper RMT-Zeta — Universalité GUE et prédiction de volatilité

Side paper de 13 pages explorant la connexion mathématique entre Random Matrix Theory et la fonction zêta de Riemann (via la statistique de Montgomery-Odlyzko) comme signal de prédiction de volatilité multi-asset.

## Contenu

- [`Reda_Mikou_Paper_RMT_Zeta.pdf`](Reda_Mikou_Paper_RMT_Zeta.pdf) — PDF compilé (13 pages)
- [`paper.tex`](paper.tex) — Source LaTeX
- [`code/`](code/) — 7 scripts numpy purs (~1500 lignes)
- [`data/`](data/) — CSV générés
- [`figures/`](figures/) — 10 figures PNG

## Pipeline de reproduction (15 secondes total)

```bash
cd code
python 01_build_data.py          # Génère 60 séries x 4000 jours
python 02_simple_example.py      # Exemple pédagogique : fenêtre pivot 2019-01-04
python 03_backtest_features.py   # Extraction features RMT/Zeta sur 772 fenêtres
python 04_ml_compare.py          # Entraîne 6 modèles ML (Ridge, RF, GBR, MLP, MC, ...)
python 05_figures_all.py         # Génère les 10 figures du paper
```

## Modules réutilisables

### `rmt_zeta.py` — Framework spectral
```python
from rmt_zeta import rmt_features, mp_bounds, tracy_widom_normalize, \
                     zeta_universality_distance, wigner_gue_pdf

C = correlation_matrix(returns_T_x_N)
eig, V = eigen_spectrum(C)
lam_minus, lam_plus = mp_bounds(T, N, sigma2=1.0)
z = zeta_universality_distance(eig)       # dict: D_KS_GUE, D_KS_Poisson, zeta_score
features = rmt_features(returns_T_x_N)    # 13 scalars
```

### `ml_models.py` — Modèles ML from scratch
```python
from ml_models import Ridge, RandomForestRegressor, GradientBoostingRegressor, \
                      MLPRegressor, MonteCarloVolPredictor, rmse, r2

model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.03,
                                   max_depth=3, subsample=0.7, seed=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(f"RMSE = {rmse(y_test, predictions):.4f}")
```

## Résultats principaux

| Modèle | RMSE | R² | Gain vs HAR |
|---|---|---|---|
| HAR-Ridge (baseline) | 0.1432 | +0.078 | — |
| **Ridge-ALL (gagnant)** | **0.1394** | **+0.127** | **+2.65%** |
| GradBoost | 0.1457 | +0.046 | -1.8% |
| MLP-Adam | 0.1463 | +0.038 | -2.2% |
| Ridge-RMT seul | 0.1642 | -0.212 | -14.7% |

**Conclusion** : Le signal RMT/Zeta est **complémentaire** (pas substitutif) à la vol historique. Le gain marginal est modeste en agrégat (+2.65%) mais devient significatif en régime CRISE (-4.4% pour MLP-Adam).

## Pourquoi numpy pur ?

L'implémentation évite délibérément scipy / sklearn / xgboost pour deux raisons :
1. **Portabilité maximale** : tourne dans n'importe quel sandbox Python sans dépendances
2. **Valeur pédagogique** : tous les algorithmes (CART, RF, Gradient Boosting Friedman, Adam) sont écrits explicitement et lisibles en ~600 lignes

Ce n'est pas optimisé pour la production — la version XGBoost natif serait 50-100x plus rapide. Mais l'objectif est démonstratif.
