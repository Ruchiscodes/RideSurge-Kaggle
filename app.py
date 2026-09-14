from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.inference import predict

app = FastAPI(title="RideSurge Kaggle Pricing API", version="1.0.0")


class RideRequest(BaseModel):
    number_of_riders: int = Field(ge=1, le=10000)
    number_of_drivers: int = Field(ge=1, le=10000)
    number_of_past_rides: int = Field(ge=0)
    average_ratings: float = Field(ge=1, le=5)
    expected_ride_duration: float = Field(gt=0, le=500)
    location_category: Literal["Urban", "Suburban", "Rural"]
    customer_loyalty_status: Literal["Regular", "Silver", "Gold"]
    time_of_booking: Literal["Morning", "Afternoon", "Evening", "Night"]
    vehicle_type: Literal["Economy", "Premium"]


@app.get("/health")
def health(): return {"status": "ok"}


@app.post("/predict")
def fare(request: RideRequest):
    try:
        return predict(request.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(503, "Train the model with: python train.py") from exc

