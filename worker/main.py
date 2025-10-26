"""
Python Worker Service for Model Training and Predictions

This service handles ML model training and predictions with bidirectional
communication with the backend service.
"""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import redis
from pydantic import BaseModel, Field
import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('worker_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('worker_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
ACTIVE_TRAINING_JOBS = Gauge('worker_active_training_jobs', 'Number of active training jobs')
ACTIVE_PREDICTION_JOBS = Gauge('worker_active_prediction_jobs', 'Number of active prediction jobs')
MODEL_CACHE_SIZE = Gauge('worker_model_cache_size', 'Number of models in cache')

# Global variables
redis_client: Optional[redis.Redis] = None
backend_client: Optional[httpx.AsyncClient] = None
model_cache: Dict[str, Any] = {}
active_jobs: Dict[str, Dict[str, Any]] = {
    'training': {},
    'prediction': {}
}

class Config:
    """Configuration class"""
    def __init__(self):
        self.WORKER_ID = os.getenv('WORKER_ID', 'worker-1')
        self.BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:3000')
        self.REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        self.MAX_CONCURRENT_JOBS = int(os.getenv('MAX_CONCURRENT_JOBS', '5'))
        self.MODEL_CACHE_TTL = int(os.getenv('MODEL_CACHE_TTL', '3600'))  # 1 hour
        self.HEALTH_CHECK_INTERVAL = int(os.getenv('HEALTH_CHECK_INTERVAL', '30'))  # 30 seconds

config = Config()

# Pydantic models
class TrainingRequest(BaseModel):
    """Model training request"""
    job_id: str
    organization_id: str
    model_type: str = Field(..., description="Type of model to train")
    training_data: Dict[str, Any] = Field(..., description="Training data configuration")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    callback_url: Optional[str] = None

class PredictionRequest(BaseModel):
    """Model prediction request"""
    job_id: str
    organization_id: str
    model_id: str
    input_data: Dict[str, Any] = Field(..., description="Input data for prediction")
    callback_url: Optional[str] = None

class JobStatus(BaseModel):
    """Job status response"""
    job_id: str
    status: str  # pending, running, completed, failed
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    worker_id: str
    version: str = "1.0.0"
    uptime: float
    memory_usage: float
    cpu_usage: float
    active_jobs: Dict[str, int]
    redis_connected: bool
    backend_connected: bool

class InternalMessage(BaseModel):
    """Internal communication message"""
    message_type: str
    payload: Dict[str, Any]
    timestamp: str
    worker_id: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global redis_client, backend_client
    
    logger.info("Starting Python Worker Service", worker_id=config.WORKER_ID)
    
    # Initialize Redis connection
    try:
        redis_client = redis.from_url(config.REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("Connected to Redis", redis_url=config.REDIS_URL)
    except Exception as e:
        logger.error("Failed to connect to Redis", error=str(e))
        redis_client = None
    
    # Initialize HTTP client for backend communication
    backend_client = httpx.AsyncClient(
        base_url=config.BACKEND_URL,
        timeout=30.0,
        headers={"User-Agent": f"PythonWorker/{config.WORKER_ID}"}
    )
    
    # Register worker with backend
    await register_worker()
    
    # Start background tasks
    asyncio.create_task(health_check_loop())
    asyncio.create_task(cleanup_expired_jobs())
    
    yield
    
    # Cleanup
    if backend_client:
        await backend_client.aclose()
    if redis_client:
        redis_client.close()
    
    logger.info("Python Worker Service stopped")

app = FastAPI(
    title="Python Worker Service",
    description="ML Model Training and Prediction Service",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get Redis client
def get_redis() -> redis.Redis:
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")
    return redis_client

# Dependency to get backend client
def get_backend_client() -> httpx.AsyncClient:
    if not backend_client:
        raise HTTPException(status_code=503, detail="Backend client not available")
    return backend_client

async def register_worker():
    """Register this worker with the backend"""
    try:
        response = await backend_client.post("/api/internal/workers/register", json={
            "worker_id": config.WORKER_ID,
            "worker_type": "python_ml",
            "capabilities": ["training", "prediction"],
            "max_concurrent_jobs": config.MAX_CONCURRENT_JOBS,
            "status": "active"
        })
        if response.status_code == 200:
            logger.info("Successfully registered worker with backend")
        else:
            logger.warning("Failed to register worker", status_code=response.status_code)
    except Exception as e:
        logger.error("Error registering worker", error=str(e))

async def notify_backend(job_id: str, status: str, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
    """Notify backend about job status changes"""
    try:
        payload = {
            "job_id": job_id,
            "status": status,
            "result": result,
            "error": error,
            "worker_id": config.WORKER_ID,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        response = await backend_client.post("/api/internal/workers/job-update", json=payload)
        if response.status_code != 200:
            logger.warning("Failed to notify backend", job_id=job_id, status_code=response.status_code)
    except Exception as e:
        logger.error("Error notifying backend", job_id=job_id, error=str(e))

async def health_check_loop():
    """Periodic health check with backend"""
    while True:
        try:
            await asyncio.sleep(config.HEALTH_CHECK_INTERVAL)
            
            health_data = {
                "worker_id": config.WORKER_ID,
                "status": "healthy",
                "active_jobs": {
                    "training": len(active_jobs['training']),
                    "prediction": len(active_jobs['prediction'])
                },
                "memory_usage": psutil.virtual_memory().percent,
                "cpu_usage": psutil.cpu_percent(),
                "timestamp": asyncio.get_event_loop().time()
            }
            
            response = await backend_client.post("/api/internal/workers/health", json=health_data)
            if response.status_code != 200:
                logger.warning("Health check failed", status_code=response.status_code)
                
        except Exception as e:
            logger.error("Health check error", error=str(e))

async def cleanup_expired_jobs():
    """Cleanup expired jobs from memory"""
    while True:
        try:
            await asyncio.sleep(300)  # Run every 5 minutes
            
            current_time = asyncio.get_event_loop().time()
            expired_jobs = []
            
            for job_type in ['training', 'prediction']:
                for job_id, job_data in active_jobs[job_type].items():
                    if current_time - job_data.get('started_at', 0) > 3600:  # 1 hour timeout
                        expired_jobs.append((job_type, job_id))
            
            for job_type, job_id in expired_jobs:
                logger.warning("Cleaning up expired job", job_type=job_type, job_id=job_id)
                del active_jobs[job_type][job_id]
                await notify_backend(job_id, "expired", error="Job expired after 1 hour")
                
        except Exception as e:
            logger.error("Error in cleanup task", error=str(e))

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check Redis connection
        redis_connected = False
        if redis_client:
            try:
                redis_client.ping()
                redis_connected = True
            except:
                pass
        
        # Check backend connection
        backend_connected = False
        if backend_client:
            try:
                response = await backend_client.get("/api/v1/health", timeout=5.0)
                backend_connected = response.status_code == 200
            except:
                pass
        
        return HealthResponse(
            status="healthy" if redis_connected and backend_connected else "degraded",
            worker_id=config.WORKER_ID,
            uptime=psutil.boot_time(),
            memory_usage=psutil.virtual_memory().percent,
            cpu_usage=psutil.cpu_percent(),
            active_jobs={
                "training": len(active_jobs['training']),
                "prediction": len(active_jobs['prediction'])
            },
            redis_connected=redis_connected,
            backend_connected=backend_connected
        )
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=500, detail="Health check failed")

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Model training endpoint
@app.post("/api/training/start", response_model=JobStatus)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    redis: redis.Redis = Depends(get_redis)
):
    """Start model training job"""
    try:
        # Check if we can accept more jobs
        if len(active_jobs['training']) >= config.MAX_CONCURRENT_JOBS:
            raise HTTPException(status_code=429, detail="Maximum concurrent jobs reached")
        
        # Initialize job
        job_data = {
            "job_id": request.job_id,
            "organization_id": request.organization_id,
            "model_type": request.model_type,
            "training_data": request.training_data,
            "hyperparameters": request.hyperparameters,
            "status": "pending",
            "progress": 0.0,
            "started_at": asyncio.get_event_loop().time(),
            "callback_url": request.callback_url
        }
        
        active_jobs['training'][request.job_id] = job_data
        ACTIVE_TRAINING_JOBS.inc()
        
        # Start training in background
        background_tasks.add_task(train_model, request.job_id, job_data)
        
        logger.info("Training job started", job_id=request.job_id, model_type=request.model_type)
        
        return JobStatus(
            job_id=request.job_id,
            status="pending",
            progress=0.0,
            started_at=job_data["started_at"]
        )
        
    except Exception as e:
        logger.error("Failed to start training", job_id=request.job_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Model prediction endpoint
@app.post("/api/prediction/start", response_model=JobStatus)
async def start_prediction(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    redis: redis.Redis = Depends(get_redis)
):
    """Start model prediction job"""
    try:
        # Check if we can accept more jobs
        if len(active_jobs['prediction']) >= config.MAX_CONCURRENT_JOBS:
            raise HTTPException(status_code=429, detail="Maximum concurrent jobs reached")
        
        # Initialize job
        job_data = {
            "job_id": request.job_id,
            "organization_id": request.organization_id,
            "model_id": request.model_id,
            "input_data": request.input_data,
            "status": "pending",
            "progress": 0.0,
            "started_at": asyncio.get_event_loop().time(),
            "callback_url": request.callback_url
        }
        
        active_jobs['prediction'][request.job_id] = job_data
        ACTIVE_PREDICTION_JOBS.inc()
        
        # Start prediction in background
        background_tasks.add_task(predict_model, request.job_id, job_data)
        
        logger.info("Prediction job started", job_id=request.job_id, model_id=request.model_id)
        
        return JobStatus(
            job_id=request.job_id,
            status="pending",
            progress=0.0,
            started_at=job_data["started_at"]
        )
        
    except Exception as e:
        logger.error("Failed to start prediction", job_id=request.job_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Job status endpoint
@app.get("/api/jobs/{job_id}/status", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get job status"""
    # Check training jobs
    if job_id in active_jobs['training']:
        job_data = active_jobs['training'][job_id]
        return JobStatus(
            job_id=job_id,
            status=job_data["status"],
            progress=job_data["progress"],
            result=job_data.get("result"),
            error=job_data.get("error"),
            started_at=job_data.get("started_at"),
            completed_at=job_data.get("completed_at")
        )
    
    # Check prediction jobs
    if job_id in active_jobs['prediction']:
        job_data = active_jobs['prediction'][job_id]
        return JobStatus(
            job_id=job_id,
            status=job_data["status"],
            progress=job_data["progress"],
            result=job_data.get("result"),
            error=job_data.get("error"),
            started_at=job_data.get("started_at"),
            completed_at=job_data.get("completed_at")
        )
    
    raise HTTPException(status_code=404, detail="Job not found")

# List active jobs
@app.get("/api/jobs")
async def list_jobs():
    """List all active jobs"""
    return {
        "training": list(active_jobs['training'].keys()),
        "prediction": list(active_jobs['prediction'].keys()),
        "total": len(active_jobs['training']) + len(active_jobs['prediction'])
    }

# Cancel job endpoint
@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running job"""
    # Check training jobs
    if job_id in active_jobs['training']:
        job_data = active_jobs['training'][job_id]
        if job_data["status"] in ["pending", "running"]:
            job_data["status"] = "cancelled"
            job_data["completed_at"] = asyncio.get_event_loop().time()
            await notify_backend(job_id, "cancelled")
            logger.info("Training job cancelled", job_id=job_id)
            return {"message": "Job cancelled successfully"}
    
    # Check prediction jobs
    if job_id in active_jobs['prediction']:
        job_data = active_jobs['prediction'][job_id]
        if job_data["status"] in ["pending", "running"]:
            job_data["status"] = "cancelled"
            job_data["completed_at"] = asyncio.get_event_loop().time()
            await notify_backend(job_id, "cancelled")
            logger.info("Prediction job cancelled", job_id=job_id)
            return {"message": "Job cancelled successfully"}
    
    raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")

# Internal communication endpoint
@app.post("/api/internal/message")
async def receive_internal_message(message: InternalMessage):
    """Receive internal messages from backend"""
    try:
        logger.info("Received internal message", 
                   message_type=message.message_type, 
                   worker_id=message.worker_id)
        
        # Handle different message types
        if message.message_type == "job_cancel":
            job_id = message.payload.get("job_id")
            if job_id:
                await cancel_job(job_id)
        
        elif message.message_type == "worker_shutdown":
            logger.info("Received shutdown command")
            # Graceful shutdown logic here
            return {"message": "Shutdown command received"}
        
        elif message.message_type == "model_update":
            model_id = message.payload.get("model_id")
            model_data = message.payload.get("model_data")
            if model_id and model_data:
                model_cache[model_id] = model_data
                MODEL_CACHE_SIZE.set(len(model_cache))
        
        return {"message": "Message processed successfully"}
        
    except Exception as e:
        logger.error("Error processing internal message", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Background task functions
async def train_model(job_id: str, job_data: Dict[str, Any]):
    """Background task for model training"""
    try:
        # Update job status
        job_data["status"] = "running"
        job_data["progress"] = 0.1
        await notify_backend(job_id, "running")
        
        # Simulate training process
        model_type = job_data["model_type"]
        training_data = job_data["training_data"]
        hyperparameters = job_data["hyperparameters"]
        
        logger.info("Starting model training", 
                   job_id=job_id, 
                   model_type=model_type,
                   organization_id=job_data["organization_id"])
        
        # Simulate training steps
        for step in range(1, 11):
            await asyncio.sleep(2)  # Simulate training time
            progress = step * 0.1
            job_data["progress"] = progress
            await notify_backend(job_id, "running", {"progress": progress})
        
        # Simulate model training result
        model_result = {
            "model_id": f"model_{job_id}",
            "model_type": model_type,
            "accuracy": 0.95,
            "training_time": 20.0,
            "hyperparameters": hyperparameters,
            "metrics": {
                "precision": 0.94,
                "recall": 0.96,
                "f1_score": 0.95
            }
        }
        
        # Update job status
        job_data["status"] = "completed"
        job_data["progress"] = 1.0
        job_data["result"] = model_result
        job_data["completed_at"] = asyncio.get_event_loop().time()
        
        # Store model in cache
        model_cache[model_result["model_id"]] = model_result
        MODEL_CACHE_SIZE.set(len(model_cache))
        
        await notify_backend(job_id, "completed", model_result)
        
        logger.info("Model training completed", 
                   job_id=job_id, 
                   model_id=model_result["model_id"],
                   accuracy=model_result["accuracy"])
        
    except Exception as e:
        logger.error("Model training failed", job_id=job_id, error=str(e))
        job_data["status"] = "failed"
        job_data["error"] = str(e)
        job_data["completed_at"] = asyncio.get_event_loop().time()
        await notify_backend(job_id, "failed", error=str(e))
    
    finally:
        # Clean up job
        if job_id in active_jobs['training']:
            del active_jobs['training'][job_id]
            ACTIVE_TRAINING_JOBS.dec()

async def predict_model(job_id: str, job_data: Dict[str, Any]):
    """Background task for model prediction"""
    try:
        # Update job status
        job_data["status"] = "running"
        job_data["progress"] = 0.1
        await notify_backend(job_id, "running")
        
        model_id = job_data["model_id"]
        input_data = job_data["input_data"]
        
        logger.info("Starting model prediction", 
                   job_id=job_id, 
                   model_id=model_id,
                   organization_id=job_data["organization_id"])
        
        # Check if model is in cache
        if model_id not in model_cache:
            # Simulate loading model
            await asyncio.sleep(1)
            model_cache[model_id] = {"model_id": model_id, "loaded": True}
            MODEL_CACHE_SIZE.set(len(model_cache))
        
        # Simulate prediction process
        await asyncio.sleep(1)
        job_data["progress"] = 0.5
        await notify_backend(job_id, "running", {"progress": 0.5})
        
        await asyncio.sleep(1)
        job_data["progress"] = 1.0
        
        # Simulate prediction result
        prediction_result = {
            "model_id": model_id,
            "predictions": [0.8, 0.2, 0.9, 0.1, 0.7],
            "confidence": 0.85,
            "prediction_time": 2.0,
            "input_features": len(input_data.get("features", []))
        }
        
        # Update job status
        job_data["status"] = "completed"
        job_data["result"] = prediction_result
        job_data["completed_at"] = asyncio.get_event_loop().time()
        
        await notify_backend(job_id, "completed", prediction_result)
        
        logger.info("Model prediction completed", 
                   job_id=job_id, 
                   model_id=model_id,
                   confidence=prediction_result["confidence"])
        
    except Exception as e:
        logger.error("Model prediction failed", job_id=job_id, error=str(e))
        job_data["status"] = "failed"
        job_data["error"] = str(e)
        job_data["completed_at"] = asyncio.get_event_loop().time()
        await notify_backend(job_id, "failed", error=str(e))
    
    finally:
        # Clean up job
        if job_id in active_jobs['prediction']:
            del active_jobs['prediction'][job_id]
            ACTIVE_PREDICTION_JOBS.dec()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level=config.LOG_LEVEL.lower(),
        access_log=True
    )
