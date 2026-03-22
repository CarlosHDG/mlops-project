from pydantic import BaseModel,Field
from typing import List

class HousePredictionRequest(BaseModel):
    sqft:float = Field(description="",gt=0)
    bathrooms:int = Field(description="",ge=0)
    year_built:int = Field(description="",ge=1800)
    bedrooms:int =Field(description="",ge=1)
    location:str = Field(description="Location (urban, suburban, rural)")
    condition:str = Field(description="Condition (e.g. Good, Excellent, Fair)")

class PredictionResponse(BaseModel):
    predicted_price: float
    confidence_interval:List[float]
    features_importance: dict
    prediction_time:str