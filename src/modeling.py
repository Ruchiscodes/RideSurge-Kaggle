import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.config import ARTIFACTS, CATEGORICAL, FEATURES, NUMERIC, RANDOM_STATE, TARGET


def split_data(df):
    return train_test_split(df[FEATURES], df[TARGET], test_size=.20, random_state=RANDOM_STATE)


def preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
    ])


def candidates():
    return {
        "Linear Regression": (LinearRegression(), [{}]),
        "Ridge": (Ridge(), {"model__alpha": [.1, 1, 10]}),
        "ElasticNet": (ElasticNet(max_iter=10000), {"model__alpha": [.001, .01, .1], "model__l1_ratio": [.2, .5, .8]}),
        "KNN": (KNeighborsRegressor(), {"model__n_neighbors": [5, 9, 15], "model__weights": ["uniform", "distance"]}),
        "Random Forest": (RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1), {"model__n_estimators": [200], "model__max_depth": [6, 10, None], "model__min_samples_leaf": [1, 3]}),
        "Extra Trees": (ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=1), {"model__n_estimators": [200], "model__max_depth": [6, 10, None], "model__min_samples_leaf": [1, 3]}),
        "Gradient Boosting": (GradientBoostingRegressor(random_state=RANDOM_STATE), {"model__n_estimators": [100, 200], "model__max_depth": [2, 3], "model__learning_rate": [.03, .05]}),
    }


def benchmark(df, output_dir: Path = ARTIFACTS):
    X_train, X_test, y_train, y_test = split_data(df)
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    baseline_rmse = mean_squared_error(y_test, baseline.predict(X_test)) ** .5
    results = []
    for name, (model, params) in candidates().items():
        search = GridSearchCV(Pipeline([("prep", preprocessor()), ("model", model)]), params,
                              cv=5, scoring="neg_root_mean_squared_error", n_jobs=1, return_train_score=False)
        search.fit(X_train, y_train)
        pred = search.predict(X_test)
        results.append({"model": name, "cv_rmse": -search.best_score_,
                        "test_rmse": mean_squared_error(y_test, pred) ** .5,
                        "test_mae": mean_absolute_error(y_test, pred), "test_r2": r2_score(y_test, pred),
                        "best_params": json.dumps(search.best_params_)})
    table = pd.DataFrame(results).sort_values("test_rmse").reset_index(drop=True)
    table.to_csv(output_dir / "seven_model_comparison.csv", index=False)
    best_rmse = float(table.iloc[0].test_rmse)
    baseline_report = {"method": "training-mean DummyRegressor", "rmse": float(baseline_rmse),
                       "best_model_rmse": best_rmse,
                       "rmse_reduction_pct": float((baseline_rmse - best_rmse) / baseline_rmse * 100)}
    (output_dir / "baseline_comparison.json").write_text(json.dumps(baseline_report, indent=2), encoding="utf-8")
    return table, baseline_report


def train_catboost(df, output_dir: Path = ARTIFACTS):
    from catboost import CatBoostRegressor
    X_train, X_test, y_train, y_test = split_data(df)
    cat_indices = [X_train.columns.get_loc(c) for c in CATEGORICAL]
    search = GridSearchCV(
        CatBoostRegressor(verbose=False, random_seed=RANDOM_STATE),
        # Four combinations x five folds: rigorous but practical on a laptop.
        {"iterations": [400, 700], "depth": [4, 6], "learning_rate": [.05], "l2_leaf_reg": [3]},
        cv=5, scoring="neg_root_mean_squared_error", n_jobs=1,
    )
    search.fit(X_train, y_train, cat_features=cat_indices)
    model = search.best_estimator_
    pred = model.predict(X_test)
    metrics = {"rmse": float(mean_squared_error(y_test, pred) ** .5),
               "mae": float(mean_absolute_error(y_test, pred)), "r2": float(r2_score(y_test, pred)),
               "best_params": search.best_params_}
    model.save_model(str(output_dir / "catboost_model.cbm"))
    joblib.dump({"features": FEATURES, "categorical": CATEGORICAL}, output_dir / "model_contract.joblib")
    (output_dir / "catboost_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return model, X_test, y_test, pred, metrics
