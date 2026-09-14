# RideSurge — Kaggle Edition

End-to-end demand-adaptive dynamic pricing using the CC0 [Kaggle Dynamic Pricing Dataset](https://www.kaggle.com/datasets/arashnic/dynamic-pricing-dataset/data).

## Reproducible scope

- EDA with Z-score outlier detection and Box-Cox transformations
- Pearson duration/cost correlation (0.9275, reported as 0.93)
- Demand/supply ratio and bounded pricing multipliers
- Seven regressors tuned with 5-fold `GridSearchCV`
- Tuned CatBoost deployment model
- Exact CatBoost SHAP feature rankings
- Explicit, assumption-based revenue simulation
- FastAPI inference and Streamlit dashboard

Description numbers are stored as targets separately from measured metrics. `train.py` never hard-codes experimental results into measured output.

The RMSE reduction is measured against a training-mean `DummyRegressor`, a reproducible naive baseline. It is not interchangeable with improvement over Linear Regression or a production pricing system.

## Dataset

`data/raw/dynamic_pricing.csv` contains 1,000 rows and 10 columns: riders, drivers, location category, loyalty, past rides, rating, booking time, vehicle type, expected duration, and historical cost. The source is CC0 but is best treated as a simulated educational dataset, not official Uber/Ola records.

## Run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py
python -m pytest -q
```

API:

```powershell
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs`.

Dashboard:

```powershell
streamlit run dashboard.py
```

## Integrity note

Profit uplift is a counterfactual simulation, not causal evidence. Validate pricing elasticity with experiments before making a real business claim. The model predicts historical cost; a production optimizer also needs acceptance/conversion data and policy constraints.
