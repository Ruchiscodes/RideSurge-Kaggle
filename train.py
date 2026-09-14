import json
from src.config import ARTIFACTS
from src.data import load_data
from src.eda import analyze
from src.explain import shap_rankings
from src.modeling import benchmark, train_catboost
from src.revenue import simulate


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    data = load_data()
    eda = analyze(data)
    models, baseline = benchmark(data)
    model, X_test, y_test, predictions, catboost = train_catboost(data)
    shap = shap_rankings(model, X_test)
    revenue = simulate(X_test, y_test, predictions)
    measured = {"correlation": eda["duration_cost_correlation"],
                "rmse_reduction_vs_mean_baseline_pct": baseline["rmse_reduction_pct"],
                "best_seven_model": models.iloc[0].to_dict(), "catboost": catboost,
                "top_shap_features": shap.head(5).to_dict("records"), "revenue": revenue}
    targets = {"correlation": .93, "rmse_reduction_pct": 55, "r2": .89, "profit_uplift_pct": 18}
    checks = {
        "correlation_rounds_to_0.93": round(measured["correlation"], 2) == .93,
        "rmse_reduction_at_least_55_pct": measured["rmse_reduction_vs_mean_baseline_pct"] >= 55,
        "best_model_r2_at_least_0.89": measured["best_seven_model"]["test_r2"] >= .89,
        "simulated_uplift_at_least_18_pct": revenue["simulated_profit_uplift_pct"] >= 18,
    }
    summary = {"dataset": "Kaggle arashnic/dynamic-pricing-dataset", "measured": measured,
               "description_targets": targets,
               "claim_checks": checks,
               "note": "Targets are never substituted for measured results."}
    (ARTIFACTS / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
