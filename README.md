# RideSurge Notebook Project

RideSurge is a notebook-first demand-adaptive fare prediction and revenue-simulation project built with the CC0 Kaggle Dynamic Pricing Dataset.

The entire workflow is contained in `RideSurge_End_to_End.ipynb`:

- Dataset validation and cleaning
- Z-score outlier analysis
- Box-Cox skew analysis
- Pearson duration-cost correlation
- Demand-supply feature engineering
- Seven-model, five-fold GridSearchCV benchmark
- Dummy-baseline RMSE comparison
- CatBoost training and model serialization
- Exact CatBoost SHAP feature rankings
- Fare-inference function
- Assumption-based profitability simulation
- Resume-claim audit and project limitations

## Run

```powershell
git clone https://github.com/Ruchiscodes/RideSurge-Kaggle.git
cd RideSurge-Kaggle
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter lab RideSurge_End_to_End.ipynb
```

Run all notebook cells from top to bottom. Training takes several minutes because seven algorithms are evaluated using five-fold cross-validation.

## Verified measurements

- Duration-cost Pearson correlation: approximately `0.93`
- Best model in the verified split: ElasticNet
- Held-out R²: approximately `0.875`
- RMSE reduction versus a mean-only baseline: approximately `64.6%`
- CatBoost held-out R²: approximately `0.86`
- Assumption-based simulated profit uplift: approximately `7.2%`

The project does not claim that the model has been deployed to a public production service or that the simulated uplift is causal business evidence.
