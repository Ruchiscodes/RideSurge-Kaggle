from functools import lru_cache
import pandas as pd
from catboost import CatBoostRegressor
from src.config import ARTIFACTS, FEATURES
from src.data import engineer_market_features


@lru_cache
def load_model():
    model = CatBoostRegressor()
    model.load_model(str(ARTIFACTS / "catboost_model.cbm"))
    return model


def predict(payload):
    row = engineer_market_features(pd.DataFrame([payload]))
    fare = max(0, float(load_model().predict(row[FEATURES])[0]))
    return {"recommended_fare": round(fare, 2),
            "demand_supply_ratio": round(float(row.demand_supply_ratio.iloc[0]), 3),
            "pricing_multiplier": round(float(row.pricing_multiplier.iloc[0]), 3)}

