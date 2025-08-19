import os
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path("titanic/titanic_model.pkl")

# Load model at startup
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH.resolve()}")
titanic_model = joblib.load(MODEL_PATH)

from pydantic import BaseModel
from typing import List

class ModelInfo(BaseModel):
    model_type: str
    features: List[str]
    model_file: str
    file_size_kb: float
    load_time: str
    api_version: str


# Get model metadata


app = FastAPI(
    title="Titanic Survival Prediction API",
    description="API to predict Titanic passenger survival using ML model",
    version="1.0.0",
)

# -------------------------
# Data models
# -------------------------

class Person(BaseModel):
    pclass: int = Field(..., ge=1, le=3, description="Passenger class (1, 2, or 3)")
    sex: str = Field(..., description="Gender (male or female)")
    age: float = Field(..., ge=0, le=100, description="Age in years")
    sibsp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
    parch: int = Field(..., ge=0, description="Number of parents/children aboard")
    fare: float = Field(..., ge=0, description="Passenger fare")
    embarked: str = Field(..., description="Port of embarkation (C, Q, or S)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "pclass": 3,
                "sex": "male",
                "age": 22.0,
                "sibsp": 1,
                "parch": 0,
                "fare": 7.25,
                "embarked": "S"
            }
        }


class PredictionResponse(BaseModel):
    survived: bool
    survival_probability: float
    person_data: Person
    prediction_time: str


# -------------------------
# Endpoints
# -------------------------

@app.post("/predict", response_model=PredictionResponse)
async def predict_survival(person: Person):
    try:
        if person.sex.lower() not in ['male', 'female']:
            raise HTTPException(status_code=400, detail="Sex must be 'male' or 'female'")
        
        if person.embarked.upper() not in ['C', 'Q', 'S']:
            raise HTTPException(status_code=400, detail="Embarked must be 'C', 'Q', or 'S'")
        
        input_df = pd.DataFrame([person.dict()])

        prediction = titanic_model.predict(input_df)[0]
        prediction_proba = titanic_model.predict_proba(input_df)[0]

        survived = bool(prediction)
        survival_probability = float(prediction_proba[1])

        return PredictionResponse(
            survived=survived,
            survival_probability=survival_probability,
            person_data=person,
            prediction_time=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")



MODEL_INFO = {
    "model_type": type(titanic_model.named_steps['classifier']).__name__,
    "features": titanic_model.named_steps['preprocessor'].transformers_[0][2] \
              + titanic_model.named_steps['preprocessor'].transformers_[1][2],
    "model_file": str(MODEL_PATH.resolve()),
    "file_size_kb": round(os.path.getsize(MODEL_PATH) / 1024, 2),
    "load_time": datetime.now().isoformat(),
    "api_version": "1.0.0"
}

@app.get("/info", response_model=ModelInfo)
async def model_info():
    """Return information about the loaded model"""
    return MODEL_INFO



@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "message": "ML Model API",
        "version": "1.0.0",
        "docs": "/docs"
    }
