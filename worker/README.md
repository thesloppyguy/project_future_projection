# Python Worker Service

A FastAPI-based microservice for handling machine learning model training and predictions with bidirectional communication with the backend service.

## 🚀 Features

- **Model Training**: Support for classification, regression, clustering, and neural network models
- **Model Predictions**: Real-time prediction processing with model caching
- **Bidirectional Communication**: HTTP-based communication with backend service
- **Health Monitoring**: Comprehensive health checks and metrics
- **Job Management**: Async job processing with status tracking
- **Worker Registry**: Automatic registration and discovery
- **Prometheus Metrics**: Built-in metrics for monitoring
- **Docker Support**: Containerized deployment

## 🏗️ Architecture

```
┌─────────────────┐    HTTP     ┌─────────────────┐
│   Backend API   │◄──────────►│  Python Worker  │
│   (Fastify)     │             │   (FastAPI)     │
└─────────────────┘             └─────────────────┘
         │                               │
         │                               │
         ▼                               ▼
┌─────────────────┐             ┌─────────────────┐
│   PostgreSQL    │             │     Redis       │
│   (Database)    │             │   (Cache/Queue) │
└─────────────────┘             └─────────────────┘
```

## 📋 API Endpoints

### Health & Monitoring
- `GET /health` - Health check endpoint
- `GET /metrics` - Prometheus metrics

### Model Training
- `POST /api/training/start` - Start model training job
- `GET /api/jobs/{jobId}/status` - Get training job status

### Model Predictions
- `POST /api/prediction/start` - Start prediction job
- `GET /api/jobs/{jobId}/status` - Get prediction job status

### Job Management
- `GET /api/jobs` - List all active jobs
- `POST /api/jobs/{jobId}/cancel` - Cancel running job

### Internal Communication
- `POST /api/internal/message` - Receive internal messages from backend

## 🔧 Configuration

Environment variables:

```bash
# Worker identification
WORKER_ID=worker-1
WORKER_TYPE=python_ml

# Service URLs
BACKEND_URL=http://localhost:3000
REDIS_URL=redis://localhost:6379

# Logging
LOG_LEVEL=INFO

# Job configuration
MAX_CONCURRENT_JOBS=5
MODEL_CACHE_TTL=3600
HEALTH_CHECK_INTERVAL=30

# Worker capabilities
CAPABILITIES=training,prediction

# Security
API_KEY=your-api-key
INTERNAL_API_KEY=your-internal-key
```

## 🚀 Quick Start

### Development

1. **Install dependencies**:
   ```bash
   cd worker
   pip install -r requirements.txt
   ```

2. **Set environment variables**:
   ```bash
   export WORKER_ID=worker-1
   export BACKEND_URL=http://localhost:3000
   export REDIS_URL=redis://localhost:6379
   ```

3. **Run the worker**:
   ```bash
   python main.py
   ```

4. **Test the worker**:
   ```bash
   python test_worker.py
   ```

### Docker

1. **Build the image**:
   ```bash
   docker build -t python-worker .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8000:8000 \
     -e WORKER_ID=worker-1 \
     -e BACKEND_URL=http://host.docker.internal:3000 \
     -e REDIS_URL=redis://host.docker.internal:6379 \
     python-worker
   ```

### Docker Compose

The worker is included in the main docker-compose.yml:

```bash
cd backend
docker-compose up worker
```

## 📊 Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "worker_id": "worker-1",
  "version": "1.0.0",
  "uptime": 12345.67,
  "memory_usage": 45.2,
  "cpu_usage": 12.5,
  "active_jobs": {
    "training": 2,
    "prediction": 1
  },
  "redis_connected": true,
  "backend_connected": true
}
```

### Metrics

```bash
curl http://localhost:8000/metrics
```

Prometheus metrics include:
- `worker_requests_total` - Total requests by method, endpoint, and status
- `worker_request_duration_seconds` - Request duration histogram
- `worker_active_training_jobs` - Number of active training jobs
- `worker_active_prediction_jobs` - Number of active prediction jobs
- `worker_model_cache_size` - Number of models in cache

## 🤖 Model Training

### Start Training Job

```bash
curl -X POST http://localhost:8000/api/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "training_123",
    "organization_id": "org_456",
    "model_type": "classification",
    "training_data": {
      "dataSource": "data.csv",
      "features": ["feature1", "feature2"],
      "target": "label",
      "testSize": 0.2
    },
    "hyperparameters": {
      "max_depth": 10,
      "n_estimators": 100
    }
  }'
```

### Monitor Training Progress

```bash
curl http://localhost:8000/api/jobs/training_123/status
```

## 🔮 Model Predictions

### Start Prediction Job

```bash
curl -X POST http://localhost:8000/api/prediction/start \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "prediction_123",
    "organization_id": "org_456",
    "model_id": "model_789",
    "input_data": {
      "features": [1.0, 2.0, 3.0]
    }
  }'
```

### Get Prediction Results

```bash
curl http://localhost:8000/api/jobs/prediction_123/status
```

## 🔄 Worker Communication

### Backend to Worker

The backend communicates with workers through HTTP requests:

```typescript
// Submit training job
const response = await fetch('http://worker:8000/api/training/start', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(trainingData)
});
```

### Worker to Backend

Workers notify the backend of job status changes:

```python
# Notify backend of job completion
await notify_backend(job_id, "completed", result)
```

## 🧪 Testing

### Run Test Suite

```bash
python test_worker.py
```

The test suite includes:
- Health check validation
- Metrics endpoint testing
- Training job lifecycle
- Prediction job lifecycle
- Job listing and management
- Internal message handling

### Manual Testing

1. **Health Check**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Start Training**:
   ```bash
   curl -X POST http://localhost:8000/api/training/start \
     -H "Content-Type: application/json" \
     -d '{"job_id": "test", "organization_id": "test", "model_type": "classification", "training_data": {}}'
   ```

3. **Check Job Status**:
   ```bash
   curl http://localhost:8000/api/jobs/test/status
   ```

## 🔧 Development

### Project Structure

```
worker/
├── main.py              # FastAPI application
├── config.py            # Configuration management
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container configuration
├── test_worker.py      # Test suite
└── README.md           # This file
```

### Adding New Model Types

1. **Update capabilities** in `config.py`
2. **Add model type** to validation schemas
3. **Implement training logic** in `train_model()` function
4. **Add prediction logic** in `predict_model()` function

### Adding New Endpoints

1. **Define route** in `main.py`
2. **Add validation schema** using Pydantic
3. **Implement business logic**
4. **Add tests** in `test_worker.py`

## 🚨 Troubleshooting

### Common Issues

1. **Worker not registering with backend**:
   - Check `BACKEND_URL` environment variable
   - Verify backend is running and accessible
   - Check network connectivity

2. **Redis connection failed**:
   - Verify `REDIS_URL` is correct
   - Check Redis server is running
   - Verify network connectivity

3. **Jobs stuck in pending state**:
   - Check worker health status
   - Verify job queue is not full
   - Check for worker errors in logs

4. **High memory usage**:
   - Reduce `MAX_CONCURRENT_JOBS`
   - Decrease `MODEL_CACHE_TTL`
   - Monitor model cache size

### Logs

Worker logs include:
- Worker registration status
- Job start/completion events
- Health check results
- Error messages and stack traces

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python main.py
```

## 📈 Performance

### Optimization Tips

1. **Model Caching**: Models are cached in memory for faster predictions
2. **Async Processing**: All jobs run asynchronously
3. **Connection Pooling**: HTTP client uses connection pooling
4. **Resource Limits**: Configurable concurrent job limits

### Scaling

- **Horizontal Scaling**: Run multiple worker instances
- **Load Balancing**: Use load balancer to distribute jobs
- **Worker Types**: Deploy specialized workers for different model types

## 🔒 Security

- **API Key Authentication**: Optional API key validation
- **Internal Communication**: Secure internal message handling
- **Input Validation**: All inputs validated with Pydantic
- **Error Handling**: Secure error messages without sensitive data

## 📝 License

This project is licensed under the ISC License.
