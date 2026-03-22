import pandas as pd
from schemas import HousePredictionRequest,PredictionResponse
from features.features import create_features
import joblib
from datetime import datetime


MODEL_PATH="models/trained/house_price_model.pkl"
PREPROCESSOR_PATH="models/trained/preprocessor.pkl"

try:
    preprocessor=joblib.load(PREPROCESSOR_PATH)
    model=joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model or preprocessor: {str(e)}")

def predict_price(request:HousePredictionRequest) -> PredictionResponse:
    input_data=pd.DataFrame([request.model_dump()])
    input_data_featured=create_features(input_data)
    processed_features=preprocessor.transform(input_data_featured)
    predicted_price=model.predict(processed_features)[0]
    predicted_price=round(float(predicted_price),2)
    confidence_interval=[predicted_price*0.9,predicted_price*1.1]
    confidence_interval=[round(float(value),2) for value in confidence_interval]
    return PredictionResponse(
        confidence_interval=confidence_interval,
        features_importance={},
        predicted_price=predicted_price,
        prediction_time=datetime.now().isoformat()
    )
#COMPLETE BATCH PREDICTION
def batch_predict(request:list[HousePredictionRequest]) -> list[PredictionResponse]:
    input_data=pd.DataFrame([req.model_dump() for req in request])
    input_data_featured=create_features(input_data)
    processed_features=preprocessor.transform(input_data_featured)
    predictions=model.predict(processed_features)
    predictions= [round(float(pred),2) for pred in predictions]
    confidence_intervals=[[pred*0.9,pred*1.1] for pred in predictions]
    return [PredictionResponse(
        confidence_interval=conf_inter,
        features_importance={},
        predicted_price=pred,
        prediction_time=datetime.now().isoformat()
        )for pred, conf_inter in zip(predictions,confidence_intervals)]



