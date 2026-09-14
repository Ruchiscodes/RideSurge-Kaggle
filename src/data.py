import numpy as np
import pandas as pd
from src.config import DATA_PATH, RAW_TO_CANONICAL, RIDER_DEMAND_Q75, DRIVER_SUPPLY_Q25


def engineer_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add bounded, auditable demand/supply pricing signals."""
    out = df.copy()
    riders = out["number_of_riders"].astype(float)
    drivers = out["number_of_drivers"].astype(float).clip(lower=1)
    out["demand_supply_ratio"] = riders / drivers
    demand_ref = RIDER_DEMAND_Q75
    supply_ref = DRIVER_SUPPLY_Q25
    out["demand_multiplier"] = np.where(riders > demand_ref, riders / demand_ref, 1.0)
    out["supply_multiplier"] = np.where(drivers < supply_ref, supply_ref / drivers, 1.0)
    out["pricing_multiplier"] = np.clip(
        np.sqrt(out["demand_multiplier"] * out["supply_multiplier"]), 1.0, 2.5
    )
    return out


def load_data(path=DATA_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(RAW_TO_CANONICAL) - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    frame = frame.rename(columns=RAW_TO_CANONICAL)
    if frame.isna().any().any():
        raise ValueError("Raw dataset contains missing values; review before training")
    return engineer_market_features(frame)
