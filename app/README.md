# Streamlit App — RMT-Zeta Volatility Engine

App interactive multi-pages pour explorer le framework RMT + Zeta du paper.

## Lancer en local

```bash
pip install streamlit numpy pandas matplotlib
cd app
streamlit run streamlit_app.py
```

Puis ouvre http://localhost:8501

## Deployer gratuitement sur Streamlit Cloud

1. Pousse ce repo sur GitHub
2. Va sur https://share.streamlit.io/
3. Connecte ton compte GitHub
4. Selectionne ce repo, branche `main`, fichier `app/streamlit_app.py`
5. **Deploy** — ton app est en ligne en ~2 minutes a l'URL `<repo>.streamlit.app`

## 6 pages disponibles

1. **Vue d'ensemble** — Metriques cles + tableau des resultats
2. **Explorateur de spectres** — Slider de date, calcul live du spectre eigenvalue + MP fit
3. **Universalite GUE / Zeta** — Distribution des spacings vs Wigner & Poisson
4. **Comparaison de modeles** — Bar chart RMSE + tableau ordonne
5. **Predictions hors echantillon** — Multiselect de modeles vs vol realisee
6. **Analyse par regime** — Bar chart par regime + heatmap des gains vs HAR

## Architecture

L'app importe les modules `rmt_zeta.py` et `ml_models.py` depuis `paper-rmt-zeta/code/`.
Les donnees sont chargees depuis `paper-rmt-zeta/data/` et cachees via `@st.cache_data`
pour une UI fluide.

Aucune base de donnees, aucun backend : tout est computation-on-demand a partir des
CSV pre-calcules.
