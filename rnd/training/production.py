"""
FastAPI production server for hosting trained forecasting models.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from pathlib import Path
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import model modules for forecasting
from training.models import catboost
from training.models import xgboost
from training.models import random_forest
from training.models import quantile_regression

app = FastAPI(
    title="Forecasting Models API",
    description="API for serving trained time series forecasting models",
    version="1.0.0"
)

# Model configuration mapping
MODEL_CONFIG = {
    'catboost': {
        'module': catboost,
        'dirs': ['catboost_cok', 'catboost_sbd']
    },
    'xgboost': {
        'module': xgboost,
        'dirs': ['xg_boost_sbd1']
    },
    'random_forest': {
        'module': random_forest,
        'dirs': ['random_forest_blr']
    },
    'quantile_regression': {
        'module': quantile_regression,
        'dirs': ['quantile_regression_maa']
    }
}

# Branch mapping from directory names
BRANCH_MAPPING = {
    'catboost_cok': 'COK',
    'catboost_sbd': 'SBD',
    'xg_boost_sbd1': 'SBD1',
    'random_forest_blr': 'BLR',
    'quantile_regression_maa': 'MAA'
}

# Global model cache
MODELS_CACHE: Dict[str, dict] = {}


class ForecastRequest(BaseModel):
    """Request model for forecast endpoint."""
    model_type: str = Field(..., description="Type of model (catboost, xgboost, random_forest, quantile_regression)")
    branch: str = Field(..., description="Branch name (BLR, COK, MAA, SBD, SBD1)")
    n_periods: int = Field(12, ge=1, le=120, description="Number of periods to forecast (default: 12)")
    aggregation: str = Field("monthly", description="Aggregation type (monthly or weekly)")


class ForecastResponse(BaseModel):
    """Response model for forecast endpoint."""
    model_type: str
    branch: str
    n_periods: int
    aggregation: str
    forecast: List[Dict[str, Any]]
    status: str


def load_all_models():
    """Load all models from final_models directory."""
    models_dir = Path(__file__).parent / 'final_models'
    
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    loaded_models = {}
    
    for model_type, config in MODEL_CONFIG.items():
        module = config['module']
        
        for model_dir_name in config['dirs']:
            model_dir = models_dir / model_dir_name
            
            if not model_dir.exists():
                print(f"Warning: Model directory not found: {model_dir}")
                continue
            
            # Find the pickle file
            pkl_files = list(model_dir.glob('*.pkl'))
            
            if not pkl_files:
                print(f"Warning: No .pkl file found in {model_dir}")
                continue
            
            pkl_file = pkl_files[0]  # Take the first .pkl file
            
            try:
                # Load model using the module's load_model function
                model = module.load_model(str(pkl_file))
                
                # Get branch name from mapping
                branch = BRANCH_MAPPING.get(model_dir_name, model_dir_name)
                
                # Determine aggregation from filename
                filename = pkl_file.stem
                if 'monthly' in filename.lower():
                    aggregation = 'monthly'
                elif 'weekly' in filename.lower():
                    aggregation = 'weekly'
                else:
                    aggregation = 'monthly'  # Default
                
                # Create cache key
                cache_key = f"{model_type}_{branch}_{aggregation}"
                loaded_models[cache_key] = {
                    'model': model,
                    'model_type': model_type,
                    'branch': branch,
                    'aggregation': aggregation,
                    'module': module
                }
                
                print(f"✓ Loaded {model_type} model for {branch} ({aggregation}) from {pkl_file}")
                
            except Exception as e:
                print(f"Error loading model from {pkl_file}: {e}")
                continue
    
    return loaded_models


@app.on_event("startup")
async def startup_event():
    """Load all models on startup."""
    global MODELS_CACHE
    print("Loading models...")
    MODELS_CACHE = load_all_models()
    print(f"Loaded {len(MODELS_CACHE)} models")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Forecasting Models API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/models": "List available models",
            "/forecast": "Generate forecast (POST)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": len(MODELS_CACHE),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/models")
async def list_models():
    """List all available models."""
    models_list = []
    
    for cache_key, model_info in MODELS_CACHE.items():
        models_list.append({
            "model_type": model_info['model_type'],
            "branch": model_info['branch'],
            "aggregation": model_info['aggregation'],
            "cache_key": cache_key
        })
    
    return {
        "total_models": len(models_list),
        "models": models_list
    }


@app.post("/forecast", response_model=ForecastResponse)
async def generate_forecast(request: ForecastRequest):
    """
    Generate forecast using a trained model.
    
    Args:
        request: Forecast request with model_type, branch, n_periods, and aggregation
        
    Returns:
        Forecast response with predictions
    """
    # Create cache key
    cache_key = f"{request.model_type}_{request.branch}_{request.aggregation}"
    
    # Check if model exists
    if cache_key not in MODELS_CACHE:
        available_models = list(MODELS_CACHE.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Model not found. Available models: {available_models}"
        )
    
    model_info = MODELS_CACHE[cache_key]
    model = model_info['model']
    module = model_info['module']
    
    # Determine frequency based on aggregation
    if request.aggregation == 'weekly':
        freq = 'W-MON'
    else:
        freq = 'MS'  # Monthly start
    
    try:
        # Generate forecast using the module's forecast function
        forecast_series = module.forecast(
            model=model,
            n_periods=request.n_periods,
            freq=freq,
            branch=request.branch
        )
        
        if len(forecast_series) == 0:
            raise HTTPException(
                status_code=500,
                detail="Forecast generation failed - empty result"
            )
        
        # Convert to response format
        forecast_data = [
            {
                "date": date.isoformat(),
                "value": float(value)
            }
            for date, value in zip(forecast_series.index, forecast_series.values)
        ]
        
        return ForecastResponse(
            model_type=request.model_type,
            branch=request.branch,
            n_periods=request.n_periods,
            aggregation=request.aggregation,
            forecast=forecast_data,
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating forecast: {str(e)}"
        )


@app.get("/forecast/{model_type}/{branch}")
async def generate_forecast_get(
    model_type: str,
    branch: str,
    n_periods: int = 12,
    aggregation: str = "monthly"
):
    """
    Generate forecast using GET request (convenience endpoint).
    
    Args:
        model_type: Type of model (catboost, xgboost, random_forest, quantile_regression)
        branch: Branch name (BLR, COK, MAA, SBD, SBD1)
        n_periods: Number of periods to forecast (default: 12)
        aggregation: Aggregation type (monthly or weekly, default: monthly)
        
    Returns:
        Forecast response with predictions
    """
    request = ForecastRequest(
        model_type=model_type,
        branch=branch,
        n_periods=n_periods,
        aggregation=aggregation
    )
    
    return await generate_forecast(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

