# SATRIA DATA 2026 — National Data Science & Statistics Competition

![ci](https://github.com/henryferdianto/satria-data-2026/actions/workflows/ci.yml/badge.svg)

> **Best Regional Performance (Lampung Region) — Kemendikbudristek**

## Overview

End-to-end analytical and predictive solution for Indonesia's premier national statistics and data science competition, combining statistical inference, feature engineering, and machine learning.

## Approach

1. **Statistical profiling** of the provided datasets (descriptive statistics, distribution tests).
2. **Feature engineering** driven by domain understanding of the socioeconomic indicators.
3. **Modeling**: benchmark linear, tree-based, and ensemble models with stratified cross-validation.
4. **Inference & interpretation**: hypothesis testing and feature-importance analysis to support recommendations.
5. **Reporting**: consolidated metrics and business recommendations.

## Repository Structure

```
satria-data-2026/
├── src/
│   ├── data_prep.py      # loading & cleaning
│   ├── features.py       # feature engineering
│   ├── model.py          # training & CV
│   └── evaluate.py       # metrics & reporting
├── data/                 # place competition data here (git-ignored)
├── reports/              # generated figures & metrics
├── requirements.txt
└── README.md
```

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `SciPy` · `statsmodels`

## Getting Started

```bash
git clone https://github.com/henryferdianto/satria-data-2026.git
cd satria-data-2026
pip install -r requirements.txt
```

```bash
python src/data_prep.py --input data/raw/train.csv --output data/processed/train.csv
python src/model.py --train data/processed/train.csv
python src/evaluate.py --pred reports/predictions.csv
```


<!-- completed -->

## Quickstart with Sample Data

The repository ships a synthetic data generator so the full pipeline can be run
without any private dataset:

```bash
make sample && python -m src.model --train data/raw/train.csv
python -m src.run_demo
```

## Tests

```bash
pip install -r requirements.txt
pytest -q
```

## Project Status

This repository contains the project documentation and a runnable reference
implementation. A synthetic data generator is included so every pipeline stage
can be executed end-to-end; replace the sample data with your own datasets in
`data/raw/` to reproduce real results.

## License

MIT License — see [LICENSE](LICENSE).


---

**Author:** Rifky Henry Ferdianto — Data Science undergraduate at ITERA  
GitHub: [@henryferdianto](https://github.com/henryferdianto) · Portfolio: [rifky-henry-ferdianto](https://github.com/henryferdianto/rifky-henry-ferdianto)
