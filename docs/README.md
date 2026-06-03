# Interactive Dashboard

This folder hosts the interactive dashboard accompanying the
**RMT-Zeta paper**, served via GitHub Pages.

## Live URL

Once GitHub Pages is enabled on this repo, the dashboard will be available at:

```
https://<your-username>.github.io/em-trading-engine/
```

## To enable GitHub Pages

1. Push the repo to GitHub.
2. Go to **Settings** -> **Pages**.
3. Source: **Deploy from a branch**.
4. Branch: **main** / **/docs**.
5. Save. The site will be live in ~1 minute.

## Local preview

The HTML is fully self-contained (all data embedded as JSON, Plotly loaded from CDN).
Just open `index.html` in any modern browser:

```bash
cd docs
python3 -m http.server 8000
# then open http://localhost:8000
```

## What's in it

- **Vue d'ensemble** — Key metrics + results table (sortable)
- **Series temporelles** — Interactive time series of features RMT/Zeta with regime overlay
- **Comparaison modeles** — RMSE bar chart + multi-metric radar
- **Predictions vs realise** — Side-by-side comparison of up to 3 models
- **Analyse par regime** — Regime-conditional RMSE + heatmap of gains vs HAR

Built with vanilla HTML + CSS + Plotly.js. No build step.
