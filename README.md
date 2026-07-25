# Urban Traffic Volume Prediction

Predicting hourly interstate traffic volume from time and weather features, and comparing five
regression models on accuracy, stability and computational cost.

Graduation thesis project. Full analysis lives in
[`Traffic_Volume_Prediction_Thesis.ipynb`](Traffic_Volume_Prediction_Thesis.ipynb).

---

## Dataset

**Metro Interstate Traffic Volume** — hourly westbound traffic on I-94 between Minneapolis and
St. Paul, 2012–2018.

| | |
|---|---|
| Rows | 48,204 |
| Raw columns | 9 |
| Target | `traffic_volume` (vehicles/hour) |
| Target range | 0 – 7,280 (mean 3,260, std 1,987) |

Raw features: `holiday`, `temp`, `rain_1h`, `snow_1h`, `clouds_all`, `weather_main`,
`weather_description`, `date_time`.

Note on missingness: `holiday` is null 99.87% of the time — it is not a broken column, it simply
marks the handful of hours that fall on a public holiday. It is therefore converted to a binary
flag rather than imputed or dropped.

## Feature Engineering

Everything happens in a single `preprocessing()` function so the transform is reproducible:

- **Calendar features** from `date_time`: `hour`, `day_of_week`, `month`, `year`, `is_weekend`.
- **`is_rush_hour`** — weekdays at 07–09 and 16–19.
- **Cyclical encoding** — `hour_sin/cos` and `month_sin/cos`, so hour 23 and hour 0 are neighbours
  instead of opposite extremes.
- **`holiday` → binary**, **`rain_1h` → binary** (raining / not raining).
- **Temperature** — Kelvin converted to Celsius; sentinel outliers removed by keeping
  `100 K < temp < 320 K` (drops 10 rows).
- **`weather_main`** one-hot encoded (`drop_first=True`).
- **Dropped**: `date_time` (already decomposed), `snow_1h` (near-constant),
  `weather_description` (redundant with `weather_main`), raw `temp`.

Result: **48,194 × 25** → 24 features + target. Split 80/20 (`random_state=42`):
38,555 train / 9,639 test.

Strongest correlations with the target:

| Feature | Corr. |
|---|---|
| `hour_cos` | −0.764 |
| `is_rush_hour` | +0.492 |
| `hour` | +0.352 |
| `hour_sin` | −0.244 |
| `is_weekend` | −0.218 |
| `temp_celsius` | +0.132 |

Time of day dominates. Weather is a secondary effect.

## Models & Evaluation

Five models, each scored on three dimensions rather than accuracy alone:

1. **Accuracy** — RMSE, MAE, R² on the held-out test set
2. **Stability** — 5-fold cross-validated RMSE (mean and std)
3. **Efficiency** — wall-clock training time

| Model | RMSE | MAE | R² | CV RMSE (mean) | CV RMSE (std) | Train (s) |
|---|---|---|---|---|---|---|
| **XGBoost** | **374.58** | **211.14** | **0.96** | **390.26** | 9.34 | 2.00 |
| LightGBM | 396.26 | 223.10 | 0.96 | 405.63 | 8.84 | 0.90 |
| Random Forest | 417.24 | 232.08 | 0.96 | 433.52 | 8.77 | 4.11 |
| SVR | 924.39 | 601.83 | 0.78 | 947.49 | 13.98 | 2.63 |
| Linear Regression | 943.47 | 700.55 | 0.78 | 941.39 | 4.81 | 0.09 |

SVR is trained and cross-validated on an 8,000-row subsample — the RBF kernel does not scale to
38k rows in reasonable time. That caveat is part of the efficiency finding, not a workaround
around it.

### Findings

- **XGBoost wins on accuracy**; its test RMSE of ~375 vehicles/hour is roughly 19% of the target's
  standard deviation.
- **LightGBM is the practical pick**: 6% worse RMSE for 2.2× faster training, and the fastest of
  the three tree ensembles by a wide margin.
- **Non-linearity is the whole story.** Linear Regression and SVR both land at R² ≈ 0.78 while
  every tree ensemble reaches 0.96 — traffic responds to hour-of-day in a sharply non-linear,
  bimodal way (morning and evening peaks) that a linear model cannot represent.
- **CV std is small and consistent** across all models (≈5–14), so the ranking is stable rather
  than an artifact of one lucky split.
- Feature importance across all three tree models agrees with the correlation table: the hour
  features and `is_rush_hour` carry the signal.

## Repository Contents

| File | Description |
|---|---|
| `Traffic_Volume_Prediction_Thesis.ipynb` | Full analysis: EDA, preprocessing, training, comparison |
| `Metro_Interstate_Traffic_Volume.csv` | Dataset (48,204 rows) |
| `df_report.html` | Automated profiling report of the processed dataframe |
| `tez sunum.pdf` | Thesis presentation slides |

## Running It

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm jupyter
```

```bash
jupyter notebook Traffic_Volume_Prediction_Thesis.ipynb
```

Run all cells top to bottom. The CSV is read from the repository root, so no path changes are
needed. Developed on Python 3.13.

## Notebook Structure

1. Library imports
2. Data loading & initial exploration
3. Exploratory data analysis — distributions, hourly/daily/monthly patterns, weather effects
4. Preprocessing & feature engineering
5. Train/test split
6. Model training & evaluation (Linear Regression, Random Forest, XGBoost, LightGBM, SVR)
7. Comparative analysis dashboard
8. Predicted vs actual, residual diagnostics
9. Feature importance across tree models
10. Results summary & conclusions

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · Matplotlib · seaborn
