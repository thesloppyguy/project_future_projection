#!/usr/bin/env python3
"""
Simple runner script for the forecasting pipeline.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from forecast_implementation.main import ForecastingPipeline

if __name__ == "__main__":
    print("Starting Forecasting Pipeline...")
    pipeline = ForecastingPipeline()
    pipeline.run_full_pipeline()
    print("\nPipeline completed!")

