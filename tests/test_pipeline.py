from src.data import load_data, engineer_market_features


def test_dataset_and_features():
    df = load_data()
    assert len(df) == 1000
    assert df.isna().sum().sum() == 0
    assert df.demand_supply_ratio.gt(0).all()
    assert df.pricing_multiplier.between(1, 2.5).all()


def test_correlation_reproduces_description():
    correlation = load_data().expected_ride_duration.corr(load_data().historical_cost_of_ride)
    assert round(correlation, 2) == .93


def test_single_request_can_trigger_surge():
    import pandas as pd
    row = engineer_market_features(pd.DataFrame([{"number_of_riders": 100, "number_of_drivers": 5}]))
    assert row.pricing_multiplier.iloc[0] > 1
