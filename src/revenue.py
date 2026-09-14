import json
import numpy as np
from src.config import ARTIFACTS


def simulate(X, actual, predicted, output_dir=ARTIFACTS, elasticity=1.15):
    """Counterfactual illustration; not a causal or experimentally observed uplift."""
    actual = np.asarray(actual)
    baseline_price = .927547 * X["expected_ride_duration"].to_numpy() * (actual.mean() / (.927547 * X["expected_ride_duration"].mean()))
    dynamic_price = np.clip(np.asarray(predicted) * X["pricing_multiplier"].to_numpy(), actual * .70, actual * 1.45)
    cost = .32 * actual
    baseline_accept = np.clip((actual / baseline_price) ** elasticity, .35, 1)
    dynamic_accept = np.clip((actual / dynamic_price) ** elasticity, .35, 1)
    baseline_profit = float(((baseline_price - cost) * baseline_accept).sum())
    dynamic_profit = float(((dynamic_price - cost) * dynamic_accept).sum())
    result = {"baseline_profit": baseline_profit, "dynamic_profit": dynamic_profit,
              "simulated_profit_uplift_pct": (dynamic_profit / baseline_profit - 1) * 100,
              "price_elasticity_assumption": elasticity, "operating_cost_share_assumption": .32}
    (output_dir / "revenue_simulation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

