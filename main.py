# FastAPI ML Model Template
# Run with: uvicorn main:app --reload

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
import joblib
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store loaded models
models_cache = {}

# Model loading functions
def load_model(model_path: str, model_type: str = "sklearn"):
    """Load a model from file based on its type"""
    try:
        if model_type == "sklearn":
            return joblib.load(model_path)
        elif model_type == "pickle":
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        elif model_type == "tensorflow":
            import tensorflow as tf
            return tf.keras.models.load_model(model_path)
        elif model_type == "pytorch":
            import torch
            return torch.load(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        raise

async def load_models():
    """Load all models on startup"""
    # Define your models here
    model_configs = {
        "classifier_model": {
            "path": "models/classifier.pkl",
            "type": "sklearn",
            "description": "Binary classification model"
        },
        "regression_model": {
            "path": "models/regressor.pkl", 
            "type": "sklearn",
            "description": "Regression model for price prediction"
        }
        # Add more models as needed
    }
    
    for model_name, config in model_configs.items():
        model_path = Path(config["path"])
        if model_path.exists():
            try:
                models_cache[model_name] = {
                    "model": load_model(str(model_path), config["type"]),
                    "type": config["type"],
                    "description": config["description"],
                    "loaded_at": datetime.now().isoformat()
                }
                logger.info(f"Successfully loaded {model_name}")
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {str(e)}")
        else:
            logger.warning(f"Model file not found: {model_path}")

# Lifespan manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up FastAPI ML service...")
    await load_models()
    yield
    # Shutdown
    logger.info("Shutting down FastAPI ML service...")

# Initialize FastAPI app
app = FastAPI(
    title="ML Model API",
    description="FastAPI service for machine learning model predictions",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Generic prediction request"""
    features: Union[List[float], Dict[str, Any], List[Dict[str, Any]]]
    model_name: str = Field(..., description="Name of the model to use")
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [1.2, 3.4, 5.6, 7.8],
                "model_name": "classifier_model"
            }
        }

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    features_batch: List[Union[List[float], Dict[str, Any]]]
    model_name: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "features_batch": [
                    [1.2, 3.4, 5.6, 7.8],
                    [2.1, 4.3, 6.5, 8.7]
                ],
                "model_name": "classifier_model"
            }
        }

class PredictionResponse(BaseModel):
    """Prediction response"""
    prediction: Union[float, int, str, List[Union[float, int, str]]]
    model_name: str
    prediction_time: str
    confidence: Optional[float] = None

class ModelInfo(BaseModel):
    """Model information"""
    name: str
    type: str
    description: str
    loaded_at: str
    status: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: int
    timestamp: str

# Utility functions
def preprocess_features(features: Union[List, Dict, List[Dict]], model_name: str):
    """Preprocess features based on model requirements"""
    # Add your preprocessing logic here
    if isinstance(features, dict):
        # Convert dict to array if needed
        return np.array(list(features.values())).reshape(1, -1)
    elif isinstance(features, list) and len(features) > 0:
        if isinstance(features[0], dict):
            # Convert list of dicts to DataFrame
            df = pd.DataFrame(features)
            return df.values
        else:
            # Convert list to numpy array
            return np.array(features).reshape(1, -1)
    else:
        return np.array(features).reshape(1, -1)

def get_prediction_confidence(model, features):
    """Get prediction confidence if model supports it"""
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)
            return float(np.max(proba))
        elif hasattr(model, 'decision_function'):
            decision = model.decision_function(features)
            return float(np.abs(decision[0]) if len(decision) == 1 else np.max(np.abs(decision)))
    except:
        pass
    return None

# API Routes

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "ML Model API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded=len(models_cache),
        timestamp=datetime.now().isoformat()
    )

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models"""
    models_info = []
    for name, info in models_cache.items():
        models_info.append(ModelInfo(
            name=name,
            type=info["type"],
            description=info["description"],
            loaded_at=info["loaded_at"],
            status="loaded"
        ))
    return models_info

@app.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """Get information about a specific model"""
    if model_name not in models_cache:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    info = models_cache[model_name]
    return ModelInfo(
        name=model_name,
        type=info["type"],
        description=info["description"],
        loaded_at=info["loaded_at"],
        status="loaded"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a single prediction"""
    # Check if model exists
    if request.model_name not in models_cache:
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{request.model_name}' not found"
        )
    
    try:
        model_info = models_cache[request.model_name]
        model = model_info["model"]
        
        # Preprocess features
        processed_features = preprocess_features(request.features, request.model_name)
        
        # Make prediction
        prediction = model.predict(processed_features)
        
        # Get confidence if available
        confidence = get_prediction_confidence(model, processed_features)
        
        # Format prediction