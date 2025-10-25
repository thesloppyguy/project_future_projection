#!/usr/bin/env python3
"""
Comprehensive Sales Forecasting Evaluation
==========================================

This script runs the final forecasting model pipeline for all combinations of 
State, Star_Rating, and Tonnage from the monthly sales summary data.

It evaluates each combination and logs:
- Model Evaluation metrics (MAE, MSE, RMSE, MAPE)
- Predicted Sales vs Actual values
- Weather data integration results

Author: AI Assistant
Date: 2025
"""

from final_forecasting_model import SalesForecastingModel
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import the final forecasting model


class ComprehensiveEvaluator:
    """Comprehensive evaluator for all sales forecasting combinations"""

    def __init__(self):
        self.results = []
        self.state_weather_mapping = {
            'Andhra Pradesh': 'AP',
            'Karnataka': 'KA',
            'Kerala': 'KL',
            'Telangana': 'TL',
            'Tamil Nadu': 'TN'
        }

    def load_sales_data(self):
        """Load and analyze sales data"""
        print("Loading sales data...")
        df = pd.read_csv('./data/monthly_sales_summary.csv')

        # Get all unique combinations
        combinations = df.groupby(
            ['State', 'Star_Rating', 'Tonnage']).size().reset_index()
        combinations.columns = [
            'State', 'Star_Rating', 'Tonnage', 'Data_Points']

        print(f"Found {len(combinations)} unique combinations")
        print(f"States: {df['State'].unique()}")
        print(f"Star Ratings: {df['Star_Rating'].unique()}")
        print(f"Tonnages: {sorted(df['Tonnage'].unique())}")

        return combinations

    def get_weather_file_path(self, state):
        """Get the correct weather file path for a state"""
        state_code = self.state_weather_mapping.get(state)
        if state_code:
            return f'./outputs/processed_weather_data/{state_code}_weather_timeseries.csv'
        return None

    def evaluate_combination(self, state, star_rating, tonnage, combination_id, total_combinations):
        """Evaluate a single combination using the final forecasting model"""
        print(f"\n{'='*80}")
        print(f"EVALUATING COMBINATION {combination_id}/{total_combinations}")
        print(f"State: {state}")
        print(f"Star Rating: {star_rating}")
        print(f"Tonnage: {tonnage}")
        print(f"{'='*80}")

        try:
            # Initialize the forecasting model
            forecaster = SalesForecastingModel(
                state=state,
                star_rating=star_rating,
                tonnage=tonnage
            )

            # Load sales data
            sales_data = forecaster.load_sales_data()
            if sales_data is None:
                return self._create_error_result(state, star_rating, tonnage, "Failed to load sales data")

            # Load weather data
            weather_file = self.get_weather_file_path(state)
            if not weather_file or not os.path.exists(weather_file):
                return self._create_error_result(state, star_rating, tonnage, f"Weather file not found: {weather_file}")

            # Override the weather data loading method to use the correct file
            def load_weather_data():
                try:
                    df = pd.read_csv(weather_file)
                    df['Date'] = pd.to_datetime(df['Date'])
                    print(
                        f"Loaded weather data: {df.shape[0]} rows from {weather_file}")
                    return df
                except Exception as e:
                    print(f"Error loading weather data: {e}")
                    return None

            forecaster.load_weather_data = load_weather_data

            # Format sales data
            sales_formatted = forecaster.format_sales_data(sales_data)
            if sales_formatted is None:
                return self._create_error_result(state, star_rating, tonnage, "No sales data found for this combination")

            # Load weather data
            weather_data = forecaster.load_weather_data()
            if weather_data is None:
                return self._create_error_result(state, star_rating, tonnage, "Failed to load weather data")

            # Create features
            merged_data = forecaster.create_features(
                sales_formatted, weather_data)

            # Check if we have enough data
            if len(merged_data) < 12:  # Need at least 12 months for seasonal models
                return self._create_error_result(state, star_rating, tonnage, f"Insufficient data: {len(merged_data)} months")

            # Check stationarity
            print("\nChecking stationarity...")
            is_stationary = forecaster.check_stationarity(
                merged_data['Monthly_Total_Sales'])

            # Prepare data
            train_data, test_data, selected_features = forecaster.prepare_data(
                merged_data)
            if train_data is None:
                return self._create_error_result(state, star_rating, tonnage, "No training data available")

            # Fit model
            forecaster.fit_model(train_data, selected_features)

            # Make predictions if test data exists
            if len(test_data) > 0:
                pred_mean, pred_ci = forecaster.make_predictions(
                    test_data, selected_features)

                # Get actual and predicted values
                actual_values = test_data['Monthly_Total_Sales'].values
                predicted_values = pred_mean.values

                # Remove NaN values for evaluation
                mask = ~(np.isnan(actual_values) | np.isnan(predicted_values))
                if np.any(mask):
                    metrics = forecaster.evaluate_model(
                        actual_values[mask], predicted_values[mask])

                    # Create detailed results
                    result = {
                        'combination_id': combination_id,
                        'state': state,
                        'star_rating': star_rating,
                        'tonnage': tonnage,
                        'status': 'success',
                        'data_points': len(merged_data),
                        'training_points': len(train_data),
                        'test_points': len(test_data),
                        'is_stationary': is_stationary,
                        'best_params': forecaster.best_params,
                        'mae': metrics['MAE'],
                        'mse': metrics['MSE'],
                        'rmse': metrics['RMSE'],
                        'mape': metrics['MAPE'],
                        'predictions': []
                    }

                    # Add prediction details
                    for i, (date, pred_val) in enumerate(zip(test_data['Date'], pred_mean)):
                        actual_val = test_data['Monthly_Total_Sales'].iloc[i] if i < len(
                            test_data) else np.nan
                        result['predictions'].append({
                            'date': date.strftime('%Y-%m'),
                            'predicted': float(pred_val),
                            'actual': float(actual_val) if not np.isnan(actual_val) else None,
                            'error': float(pred_val - actual_val) if not np.isnan(actual_val) else None
                        })

                    print(f"\n[SUCCESS] Model evaluation completed")
                    print(f"   MAE: {metrics['MAE']:.2f}")
                    print(f"   RMSE: {metrics['RMSE']:.2f}")
                    print(f"   MAPE: {metrics['MAPE']:.2f}%")

                    return result
                else:
                    return self._create_error_result(state, star_rating, tonnage, "No valid predictions for evaluation")
            else:
                # No test data - model trained but not evaluated
                result = {
                    'combination_id': combination_id,
                    'state': state,
                    'star_rating': star_rating,
                    'tonnage': tonnage,
                    'status': 'trained_no_test',
                    'data_points': len(merged_data),
                    'training_points': len(train_data),
                    'test_points': 0,
                    'is_stationary': is_stationary,
                    'best_params': forecaster.best_params,
                    'mae': None,
                    'mse': None,
                    'rmse': None,
                    'mape': None,
                    'predictions': []
                }

                print(
                    f"\n[WARNING] Model trained but no test data for evaluation")
                return result

        except Exception as e:
            error_msg = f"Exception during evaluation: {str(e)}"
            print(f"\n[ERROR] {error_msg}")
            return self._create_error_result(state, star_rating, tonnage, error_msg)

    def _create_error_result(self, state, star_rating, tonnage, error_message):
        """Create an error result entry"""
        return {
            'combination_id': 0,
            'state': state,
            'star_rating': star_rating,
            'tonnage': tonnage,
            'status': 'error',
            'error_message': error_message,
            'data_points': 0,
            'training_points': 0,
            'test_points': 0,
            'is_stationary': None,
            'best_params': None,
            'mae': None,
            'mse': None,
            'rmse': None,
            'mape': None,
            'predictions': []
        }

    def run_comprehensive_evaluation(self):
        """Run evaluation for all combinations"""
        print("COMPREHENSIVE SALES FORECASTING EVALUATION")
        print("=" * 80)
        print(f"Started at: {datetime.now()}")

        # Load combinations
        combinations = self.load_sales_data()
        total_combinations = len(combinations)

        print(f"\nWill evaluate {total_combinations} combinations...")

        # Create output directory
        os.makedirs('outputs/evaluations', exist_ok=True)

        # Evaluate each combination
        for idx, row in combinations.iterrows():
            combination_id = idx + 1
            result = self.evaluate_combination(
                row['State'],
                row['Star_Rating'],
                row['Tonnage'],
                combination_id,
                total_combinations
            )

            self.results.append(result)

            # Save intermediate results every 10 combinations
            if combination_id % 10 == 0:
                self._save_intermediate_results(combination_id)

        # Save final results
        self._save_final_results()

        # Generate summary
        self._generate_summary()

        print(f"\n{'='*80}")
        print("COMPREHENSIVE EVALUATION COMPLETED!")
        print(f"Completed at: {datetime.now()}")
        print(f"Total combinations evaluated: {len(self.results)}")
        print(f"Results saved to: outputs/evaluations/")
        print(f"{'='*80}")

    def _save_intermediate_results(self, combination_id):
        """Save intermediate results"""
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(
            f'outputs/evaluations/intermediate_results_{combination_id}.csv', index=False)
        print(
            f"[SAVED] Intermediate results for {combination_id} combinations")

    def _save_final_results(self):
        """Save final comprehensive results"""
        # Save main results
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(
            'outputs/evaluations/comprehensive_evaluation_results.csv', index=False)

        # Save detailed predictions
        all_predictions = []
        for result in self.results:
            if result['predictions']:
                for pred in result['predictions']:
                    pred_record = {
                        'state': result['state'],
                        'star_rating': result['star_rating'],
                        'tonnage': result['tonnage'],
                        'date': pred['date'],
                        'predicted': pred['predicted'],
                        'actual': pred['actual'],
                        'error': pred['error']
                    }
                    all_predictions.append(pred_record)

        if all_predictions:
            predictions_df = pd.DataFrame(all_predictions)
            predictions_df.to_csv(
                'outputs/evaluations/detailed_predictions.csv', index=False)

        print("[SAVED] Final results saved:")
        print("   - outputs/evaluations/comprehensive_evaluation_results.csv")
        print("   - outputs/evaluations/detailed_predictions.csv")

    def _generate_summary(self):
        """Generate evaluation summary"""
        results_df = pd.DataFrame(self.results)

        # Summary statistics
        total_combinations = len(results_df)
        successful_evaluations = len(
            results_df[results_df['status'] == 'success'])
        trained_no_test = len(
            results_df[results_df['status'] == 'trained_no_test'])
        errors = len(results_df[results_df['status'] == 'error'])

        # Performance statistics for successful evaluations
        successful_results = results_df[results_df['status'] == 'success']

        summary_content = f"""# Comprehensive Sales Forecasting Evaluation Summary

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
- **Total Combinations Evaluated**: {total_combinations}
- **Successful Evaluations**: {successful_evaluations}
- **Trained (No Test Data)**: {trained_no_test}
- **Errors**: {errors}
- **Success Rate**: {(successful_evaluations/total_combinations)*100:.1f}%

## Performance Summary (Successful Evaluations Only)
"""

        if len(successful_results) > 0:
            summary_content += f"""
### Model Performance Metrics
- **Average MAE**: {successful_results['mae'].mean():.2f}
- **Average RMSE**: {successful_results['rmse'].mean():.2f}
- **Average MAPE**: {successful_results['mape'].mean():.2f}%
- **Best MAE**: {successful_results['mae'].min():.2f}
- **Worst MAE**: {successful_results['mae'].max():.2f}

### Top 10 Best Performing Combinations (by MAE)
| State | Star Rating | Tonnage | MAE | RMSE | MAPE |
|-------|-------------|---------|-----|------|------|
"""

            top_10 = successful_results.nsmallest(10, 'mae')
            for _, row in top_10.iterrows():
                summary_content += f"| {row['state']} | {row['star_rating']} | {row['tonnage']} | {row['mae']:.2f} | {row['rmse']:.2f} | {row['mape']:.2f}% |\n"

            summary_content += f"""
### Performance by State
"""
            state_performance = successful_results.groupby('state').agg({
                'mae': ['mean', 'std', 'count'],
                'rmse': 'mean',
                'mape': 'mean'
            }).round(2)

            for state in state_performance.index:
                mae_mean = state_performance.loc[state, ('mae', 'mean')]
                mae_std = state_performance.loc[state, ('mae', 'std')]
                count = state_performance.loc[state, ('mae', 'count')]
                summary_content += f"- **{state}**: MAE {mae_mean:.2f} ± {mae_std:.2f} (n={count})\n"

            summary_content += f"""
### Performance by Star Rating
"""
            rating_performance = successful_results.groupby('star_rating').agg({
                'mae': ['mean', 'std', 'count'],
                'rmse': 'mean',
                'mape': 'mean'
            }).round(2)

            for rating in rating_performance.index:
                mae_mean = rating_performance.loc[rating, ('mae', 'mean')]
                mae_std = rating_performance.loc[rating, ('mae', 'std')]
                count = rating_performance.loc[rating, ('mae', 'count')]
                summary_content += f"- **{rating}**: MAE {mae_mean:.2f} ± {mae_std:.2f} (n={count})\n"

        # Error analysis
        if errors > 0:
            summary_content += f"""
## Error Analysis
"""
            error_results = results_df[results_df['status'] == 'error']
            error_summary = error_results['error_message'].value_counts()

            for error_msg, count in error_summary.items():
                summary_content += f"- **{error_msg}**: {count} combinations\n"

        summary_content += f"""
## Files Generated
- `outputs/evaluations/comprehensive_evaluation_results.csv` - Main results
- `outputs/evaluations/detailed_predictions.csv` - Detailed predictions
- `outputs/evaluations/evaluation_summary.md` - This summary

## Recommendations
1. **Focus on High-Performing Combinations**: Prioritize the top 10 combinations for business planning
2. **Address Data Quality Issues**: Review combinations with errors for data quality improvements
3. **Model Selection**: Consider ensemble methods for combinations with high MAPE
4. **Weather Integration**: Analyze weather correlation for better feature engineering

---
*This evaluation provides comprehensive insights into sales forecasting performance across all product combinations.*
"""

        # Save summary
        with open('outputs/evaluations/evaluation_summary.md', 'w') as f:
            f.write(summary_content)

        print(
            "[SAVED] Evaluation summary saved to: outputs/evaluations/evaluation_summary.md")


def main():
    """Main execution function"""
    evaluator = ComprehensiveEvaluator()
    evaluator.run_comprehensive_evaluation()


if __name__ == "__main__":
    main()
