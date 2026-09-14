from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "dynamic_pricing.csv"
ARTIFACTS = ROOT / "artifacts"
RANDOM_STATE = 42
TARGET = "historical_cost_of_ride"
# Fitted on the bundled training dataset. In production these belong in a
# versioned feature-store/model artifact and must never be recomputed per request.
RIDER_DEMAND_Q75 = 81.0
DRIVER_SUPPLY_Q25 = 11.0

RAW_TO_CANONICAL = {
    "Number_of_Riders": "number_of_riders",
    "Number_of_Drivers": "number_of_drivers",
    "Location_Category": "location_category",
    "Customer_Loyalty_Status": "customer_loyalty_status",
    "Number_of_Past_Rides": "number_of_past_rides",
    "Average_Ratings": "average_ratings",
    "Time_of_Booking": "time_of_booking",
    "Vehicle_Type": "vehicle_type",
    "Expected_Ride_Duration": "expected_ride_duration",
    "Historical_Cost_of_Ride": TARGET,
}
NUMERIC = ["number_of_riders", "number_of_drivers", "number_of_past_rides",
           "average_ratings", "expected_ride_duration", "demand_supply_ratio",
           "demand_multiplier", "supply_multiplier", "pricing_multiplier"]
CATEGORICAL = ["location_category", "customer_loyalty_status", "time_of_booking", "vehicle_type"]
FEATURES = NUMERIC + CATEGORICAL
