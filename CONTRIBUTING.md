# Contributing

Ce repo est principalement un travail academique personnel, mais les contributions
sont bienvenues — notamment :

- Bugs reports (issues)
- Suggestions methodologiques (discussions)
- Pull requests sur le code (paper-rmt-zeta uniquement)

## Conventions

- Code : Python pur numpy/pandas/matplotlib, evite scipy/sklearn pour garder la portabilite
- Style : PEP 8, fonctions documentees par docstrings
- Commits : messages clairs en anglais ou francais
- LaTeX : compilation sans warning

## Tests

Le CI GitHub Actions execute un smoke test :
```bash
cd paper-rmt-zeta/code
python 01_build_data.py
python rmt_zeta.py
python ml_models.py
```

## Contact

Reda Mikou — redamedmikou@gmail.com
