"""
Example Usage of the Final Sales Forecasting Model
=================================================

This script demonstrates how to use the SalesForecastingModel class
for different scenarios and configurations.
"""

from final_forecasting_model import SalesForecastingModel
import pandas as pd


def example_1_basic_usage():
    """Example 1: Basic usage with default parameters"""
    print("Example 1: Basic Usage")
    print("-" * 30)

    # Initialize model with default parameters
    forecaster = SalesForecastingModel()

    # Run the complete pipeline
    results = forecaster.run_full_pipeline()

    if results:
        print("Basic forecasting completed successfully!")
        return results
    else:
        print("Basic forecasting failed!")
        return None


def example_2_custom_parameters():
    """Example 2: Custom parameters for different product configurations"""
    print("\nExample 2: Custom Parameters")
    print("-" * 30)

    # Try different product configurations
    configurations = [
        {'state': 'Tamil Nadu', 'star_rating': '5 Star', 'tonnage': 1.5},
        {'state': 'Karnataka', 'star_rating': '3 Star', 'tonnage': 1.0},
        {'state': 'Kerala', 'star_rating': '3 Star', 'tonnage': 1.5},
    ]

    results_list = []

    for config in configurations:
        print(f"\nTesting configuration: {config}")
        forecaster = SalesForecastingModel(**config)
        results = forecaster.run_full_pipeline()

        if results:
            results_list.append({
                'config': config,
                'results': results,
                'mape': results['metrics']['MAPE'] if results['metrics'] else None
            })
            print(
                f"Success! MAPE: {results['metrics']['MAPE']:.2f}%" if results['metrics'] else "Success!")
        else:
            print("Failed!")

    return results_list


def example_3_load_saved_model():
    """Example 3: Load and use a previously saved model"""
    print("\nExample 3: Load Saved Model")
    print("-" * 30)

    import pickle

    try:
        # Load the saved model
        with open('./outputs/models/final_sales_forecasting_model.pkl', 'rb') as f:
            model_data = pickle.load(f)

        print("Loaded model information:")
        print(f"  State: {model_data['state']}")
        print(f"  Star Rating: {model_data['star_rating']}")
        print(f"  Tonnage: {model_data['tonnage']}")
        print(f"  Best Parameters: {model_data['best_params']}")
        print(f"  Saved on: {model_data['timestamp']}")

        # You can use the loaded model for predictions
        loaded_model = model_data['results']
        print("Model loaded successfully!")

        return model_data

    except FileNotFoundError:
        print("No saved model found. Run the main forecasting script first.")
        return None


def example_4_batch_forecasting():
    """Example 4: Batch forecasting for multiple configurations"""
    print("\nExample 4: Batch Forecasting")
    print("-" * 30)

    # Define multiple configurations to test
    configurations = [
        {'state': 'Tamil Nadu', 'star_rating': '3 Star', 'tonnage': 1.5},
        {'state': 'Tamil Nadu', 'star_rating': '5 Star', 'tonnage': 1.5},
        {'state': 'Karnataka', 'star_rating': '3 Star', 'tonnage': 1.5},
    ]

    batch_results = []

    for i, config in enumerate(configurations, 1):
        print(
            f"\nProcessing configuration {i}/{len(configurations)}: {config}")

        try:
            forecaster = SalesForecastingModel(**config)
            results = forecaster.run_full_pipeline()

            if results and results['metrics']:
                batch_results.append({
                    'config': config,
                    'mape': results['metrics']['MAPE'],
                    'mae': results['metrics']['MAE'],
                    'rmse': results['metrics']['RMSE']
                })
                print(f"  MAPE: {results['metrics']['MAPE']:.2f}%")
            else:
                print("  No results or metrics available")

        except Exception as e:
            print(f"  Error: {e}")

    # Summary of batch results
    if batch_results:
        print(f"\nBatch Forecasting Summary:")
        print("-" * 40)
        for result in batch_results:
            config_str = f"{result['config']['state']} - {result['config']['star_rating']} - {result['config']['tonnage']}T"
            print(
                f"{config_str:30} | MAPE: {result['mape']:6.2f}% | MAE: {result['mae']:8.2f}")

    return batch_results


def main():
    """Run all examples"""
    print("Sales Forecasting Model - Example Usage")
    print("=" * 50)

    # Example 1: Basic usage
    basic_results = example_1_basic_usage()

    # Example 2: Custom parameters
    custom_results = example_2_custom_parameters()

    # Example 3: Load saved model
    saved_model = example_3_load_saved_model()

    # Example 4: Batch forecasting
    batch_results = example_4_batch_forecasting()

    print("\n" + "=" * 50)
    print("All examples completed!")

    return {
        'basic': basic_results,
        'custom': custom_results,
        'saved_model': saved_model,
        'batch': batch_results
    }


if __name__ == "__main__":
    results = main()
