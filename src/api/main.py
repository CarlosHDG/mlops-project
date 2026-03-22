from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas import PredictionResponse,HousePredictionRequest
from src.api.inference import predict_price,batch_predict

app=FastAPI(
    title="MLOps project - House price Predicion API",
    description=("An API for predicting house prices based on various features."
                 "This is end-to-end MLOPS project that I used to understand all the process to bring a ML model to production, the I reviewed to apply good practices and adapt to a real world"),
    version="1.0.1",
    contact={
        "name":"Carlos Andres Hidalgo",
        "url":"https://www.linkedin.com/in/carlos-andres-vasquez-hidalgo-780518215/",
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

#Health check endpoint
@app.get("/health",response_model=dict)
async def health_check():
    return {"status":"healthy","model_loaded":True}

#Prediction endpoint
@app.post("/predict",response_model=PredictionResponse)
async def predict(request:HousePredictionRequest):
    return predict_price(request)

@app.post("/predict-batch",response_model=list[PredictionResponse])
async def batch_prediction(request:list[HousePredictionRequest]):
    return batch_predict(request)