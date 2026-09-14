import pandas as pd
from catboost import Pool
from src.config import ARTIFACTS, CATEGORICAL


def shap_rankings(model, X, output_dir=ARTIFACTS):
    cats = [X.columns.get_loc(c) for c in CATEGORICAL]
    values = model.get_feature_importance(Pool(X, cat_features=cats), type="ShapValues")[:, :-1]
    result = pd.DataFrame({"feature": X.columns, "mean_abs_shap": abs(values).mean(axis=0)})
    result = result.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    result.to_csv(output_dir / "shap_rankings.csv", index=False)
    return result

