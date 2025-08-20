from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from utils.request import CustomerData
from utils.config import APP_NAME, VERSION, API_SECRET_KEY, preprocessor, forest_model, xgboost_model
from utils.inference import predict_new

# Initalize an app
app = FastAPI(title=APP_NAME, version=VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

api_key_header = APIKeyHeader(name="X-API-Key")
async def verify_api_key(api_key: str=Depends(api_key_header)):
    if api_key != API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="You are not authorized to use this API.")
    return api_key
    
    
    
@app.get("/", tags=['Healthy'], description="Healthy Check of API")
async def home(api_key: str=Depends(verify_api_key)) -> dict:
    return {
        "app_name": APP_NAME,
        "version": VERSION,
        "status": "Up & Running"
    }
    

@app.post("/predict/forest", tags=['Models'], description="Prediction of Churn using RandomForest")
async def predict_forest(data: CustomerData, api_key: str=Depends(verify_api_key)) -> dict:
    try:
        # call the function
        response = predict_new(data=data, pipeline=preprocessor, model=forest_model)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"There is a problem in prediction using RandomForest, {str(e)}")
    

@app.post("/predict/xgboost", tags=['Models'], description="Prediction of Churn using XGBoost")
async def predict_xgboost(data: CustomerData, api_key: str=Depends(verify_api_key)) -> dict:
    try:
        # call the function
        response = predict_new(data=data, pipeline=preprocessor, model=xgboost_model)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"There is a problem in prediction using XGBoost, {str(e)}")
    

