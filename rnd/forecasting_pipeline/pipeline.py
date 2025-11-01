"""
Main Pipeline Orchestrator

Orchestrates all 6 steps of the forecasting pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

try:
    from . import config
    from .step1_data_aggregation import main as step1_main
    from .step2_feature_engineering import main as step2_main
    from .step3_outlier_detection import main as step3_main
    from .step4_multi_model_training import main as step4_main
    from .step5_multi_model_validation import main as step5_main
    from .step6_forecast_generation import main as step6_main
except ImportError:
    import config
    from step1_data_aggregation import main as step1_main
    from step2_feature_engineering import main as step2_main
    from step3_outlier_detection import main as step3_main
    try:
        from step4_multi_model_training import main as step4_main
        from step5_multi_model_validation import main as step5_main
    except ImportError:
        # Fallback to old single-model steps
        from step4_model_training import main as step4_main
        from step5_validation import main as step5_main
    from step6_forecast_generation import main as step6_main

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.OUTPUT_DIR / 'pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Custom exception for pipeline errors."""
    pass


def run_step(step_name: str, step_func, required_files: list = None):
    """Run a pipeline step with error handling."""
    logger.info(f"\n{'='*70}")
    logger.info(f"RUNNING: {step_name}")
    logger.info(f"{'='*70}")
    
    # Check required files
    if required_files:
        for file_path in required_files:
            if not Path(file_path).exists():
                raise PipelineError(f"Required file not found: {file_path}. Please run previous steps first.")
    
    try:
        result = step_func()
        logger.info(f"✓ {step_name} completed successfully")
        return result
    except Exception as e:
        logger.error(f"✗ {step_name} failed with error: {str(e)}")
        logger.exception(e)
        raise PipelineError(f"{step_name} failed: {str(e)}") from e


def run_full_pipeline(skip_steps: list = None, exclude_deep_learning: bool = None, exclude_tonnage: bool = None):
    """
    Run the complete forecasting pipeline.
    
    Args:
        skip_steps: List of step numbers (1-6) to skip
        exclude_deep_learning: If True, excludes LSTM, GRU, and SimpleRNN models
                              If None, uses config.INCLUDE_DEEP_LEARNING setting
        exclude_tonnage: If True, aggregates only by Branch (excludes Tonnage)
                        If None, uses config.EXCLUDE_TONNAGE setting
    """
    if skip_steps is None:
        skip_steps = []
    
    # Override config setting if argument provided
    if exclude_deep_learning is not None:
        config.INCLUDE_DEEP_LEARNING = not exclude_deep_learning
        if exclude_deep_learning:
            logger.info("Deep learning models (LSTM, GRU, SimpleRNN): EXCLUDED")
        else:
            logger.info("Deep learning models (LSTM, GRU, SimpleRNN): INCLUDED")
    
    # Override config setting for Tonnage exclusion
    if exclude_tonnage is not None:
        config.EXCLUDE_TONNAGE = exclude_tonnage
        if exclude_tonnage:
            config.GROUP_BY_COLS = ["Date", "Branch"]
            config.CATEGORICAL_COLS = ["Branch"]
            logger.info("Aggregation: BY BRANCH ONLY (Tonnage EXCLUDED)")
        else:
            config.GROUP_BY_COLS = ["Date", "Branch", "Tonnage"]
            config.CATEGORICAL_COLS = ["Branch", "Tonnage"]
            logger.info("Aggregation: BY BRANCH AND TONNAGE")
    
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("HVAC FORECASTING ML PIPELINE")
    logger.info(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    results = {}
    
    try:
        # Step 1: Data Aggregation
        if 1 not in skip_steps:
            results['step1'] = run_step(
                "Step 1: Data Aggregation",
                step1_main,
                required_files=[config.SOURCE_DATA_FILE]
            )
        else:
            logger.info("Skipping Step 1: Data Aggregation")
        
        # Step 2: Feature Engineering
        if 2 not in skip_steps:
            results['step2'] = run_step(
                "Step 2: Feature Engineering",
                step2_main,
                required_files=[config.AGGREGATED_DATA_FILE]
            )
        else:
            logger.info("Skipping Step 2: Feature Engineering")
        
        # Step 3: Outlier Detection
        if 3 not in skip_steps:
            results['step3'] = run_step(
                "Step 3: Outlier Detection & Treatment",
                step3_main,
                required_files=[config.FEATURED_DATA_FILE]
            )
        else:
            logger.info("Skipping Step 3: Outlier Detection & Treatment")
        
        # Step 4: Model Training
        if 4 not in skip_steps:
            results['step4'] = run_step(
                "Step 4: Model Training",
                step4_main,
                required_files=[config.CLEANED_DATA_FILE]
            )
        else:
            logger.info("Skipping Step 4: Model Training")
        
        # Step 5: Validation
        if 5 not in skip_steps:
            results['step5'] = run_step(
                "Step 5: Validation",
                step5_main,
                required_files=[config.CLEANED_DATA_FILE]
            )
        else:
            logger.info("Skipping Step 5: Validation")
        
        # Step 6: Forecast Generation
        if 6 not in skip_steps:
            results['step6'] = run_step(
                "Step 6: Forecast Generation",
                step6_main,
                required_files=[config.CLEANED_DATA_FILE, config.MODEL_DIR / "latest_final_model.pkl"]
            )
        else:
            logger.info("Skipping Step 6: Forecast Generation")
        
        # Summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total duration: {duration}")
        logger.info("=" * 70)
        
        logger.info("\nOutput files:")
        logger.info(f"  - Aggregated data: {config.AGGREGATED_DATA_FILE}")
        logger.info(f"  - Featured data: {config.FEATURED_DATA_FILE}")
        logger.info(f"  - Cleaned data: {config.CLEANED_DATA_FILE}")
        logger.info(f"  - Models: {config.MODEL_DIR}")
        logger.info(f"  - Forecasts: {config.FORECAST_FILE}")
        logger.info(f"  - Validation results: {config.OUTPUT_DIR / 'validation_results.json'}")
        
        return results
        
    except PipelineError as e:
        logger.error(f"\n{'='*70}")
        logger.error("PIPELINE FAILED")
        logger.error(f"Error: {str(e)}")
        logger.error(f"{'='*70}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n{'='*70}")
        logger.error("PIPELINE FAILED WITH UNEXPECTED ERROR")
        logger.error(f"Error: {str(e)}")
        logger.error(f"{'='*70}")
        logger.exception(e)
        sys.exit(1)


def main():
    """Main entry point with CLI."""
    parser = argparse.ArgumentParser(
        description="HVAC Forecasting ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python pipeline.py

  # Run only forecast generation (requires previous steps to be completed)
  python pipeline.py --steps 6

  # Skip validation step
  python pipeline.py --skip 5

  # Run only steps 1-3
  python pipeline.py --steps 1 2 3
        """
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        nargs='+',
        choices=[1, 2, 3, 4, 5, 6],
        help='Run only specific steps (default: run all steps)'
    )
    
    parser.add_argument(
        '--skip',
        type=int,
        nargs='+',
        choices=[1, 2, 3, 4, 5, 6],
        help='Skip specific steps'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--exclude-deep-learning',
        action='store_true',
        help='Exclude deep learning models (LSTM, GRU, SimpleRNN) from training and validation'
    )
    
    parser.add_argument(
        '--exclude-tonnage',
        action='store_true',
        help='Aggregate only by Branch (exclude Tonnage from combination)'
    )
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Determine which steps to run
    if args.steps:
        # Run only specified steps
        all_steps = [1, 2, 3, 4, 5, 6]
        skip_steps = [s for s in all_steps if s not in args.steps]
    elif args.skip:
        skip_steps = args.skip
    else:
        skip_steps = []
    
    # Run pipeline
    run_full_pipeline(skip_steps=skip_steps, 
                     exclude_deep_learning=args.exclude_deep_learning,
                     exclude_tonnage=args.exclude_tonnage)


if __name__ == "__main__":
    main()

