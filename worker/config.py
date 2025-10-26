"""
Configuration module for Python Worker Service
"""

import os
from typing import List

class Config:
    """Configuration class for the worker service"""
    
    def __init__(self):
        # Worker identification
        self.WORKER_ID = os.getenv('WORKER_ID', 'worker-1')
        self.WORKER_TYPE = os.getenv('WORKER_TYPE', 'python_ml')
        
        # Service URLs
        self.BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:3000')
        self.REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
        
        # Logging
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        
        # Job configuration
        self.MAX_CONCURRENT_JOBS = int(os.getenv('MAX_CONCURRENT_JOBS', '5'))
        self.MODEL_CACHE_TTL = int(os.getenv('MODEL_CACHE_TTL', '3600'))  # 1 hour
        self.HEALTH_CHECK_INTERVAL = int(os.getenv('HEALTH_CHECK_INTERVAL', '30'))  # 30 seconds
        
        # Worker capabilities
        self.CAPABILITIES = os.getenv('CAPABILITIES', 'training,prediction').split(',')
        
        # Model configuration
        self.MODEL_STORAGE_PATH = os.getenv('MODEL_STORAGE_PATH', '/app/models')
        self.DEFAULT_MODEL_TYPES = ['classification', 'regression', 'clustering', 'neural_network']
        
        # Security
        self.API_KEY = os.getenv('API_KEY', '')
        self.INTERNAL_API_KEY = os.getenv('INTERNAL_API_KEY', '')
        
        # Performance
        self.REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '30'))
        self.JOB_TIMEOUT = int(os.getenv('JOB_TIMEOUT', '3600'))  # 1 hour
        
        # Monitoring
        self.METRICS_ENABLED = os.getenv('METRICS_ENABLED', 'true').lower() == 'true'
        self.METRICS_PORT = int(os.getenv('METRICS_PORT', '9090'))

# Global config instance
config = Config()
