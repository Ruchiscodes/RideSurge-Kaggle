import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import boxcox, skew, zscore
from src.config import ARTIFACTS, TARGET


def analyze(df: pd.DataFrame, output_dir: Path = ARTIFACTS) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric = df.select_dtypes(include=np.number)
    z_scores = numeric.apply(zscore)
    transforms = {}
    transformed = pd.DataFrame(index=df.index)
    for col in ["number_of_riders", "number_of_drivers", "expected_ride_duration", TARGET]:
        values = df[col].astype(float).to_numpy()
        values = values - values.min() + .001 if values.min() <= 0 else values
        converted, lam = boxcox(values)
        transformed[f"{col}_boxcox"] = converted
        transforms[col] = {"lambda": float(lam), "skew_before": float(skew(values)),
                           "skew_after": float(skew(converted))}
    transformed.to_csv(output_dir / "boxcox_transformed.csv", index=False)
    report = {
        "source": "Kaggle: arashnic/dynamic-pricing-dataset",
        "rows": int(len(df)), "columns_after_engineering": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()), "duplicates": int(df.duplicated().sum()),
        "duration_cost_correlation": float(df["expected_ride_duration"].corr(df[TARGET])),
        "zscore_outliers_abs_gt_3": {k: int(v) for k, v in (z_scores.abs() > 3).sum().items()},
        "boxcox": transforms,
    }
    (output_dir / "eda_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

