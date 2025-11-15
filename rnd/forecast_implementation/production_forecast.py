"""
Production forecasting script with dynamic SOB calculation and forecasting.
Uses best models (CatBoost) for BLR and MAA branches to forecast next 12 months.
Calculates SOB from historical full financial years and forecasts SOB for future months.
Estimates total market sales using forecasted SOB and weighted mean.
Calculates yearly and branch-wise sales for the financial year.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import logging
from datetime import datetime
import pickle

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from forecast_implementation.config import (
    DATA_FILE, MODEL_DIR, CALIBRATION_END_DATE, FORECAST_HORIZON_MONTHS,
    DATA_DIR
)
from forecast_implementation.data_preparation import DataPreparator
from forecast_implementation.feature_engineering import FeatureEngineer
from forecast_implementation.models import ModelTrainer
from forecast_implementation.utils import detect_frequency, create_future_dates
from forecast_implementation.config import LAG_PERIODS, ROLLING_WINDOWS, INCLUDE_SEASONALITY, INCLUDE_TREND

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SOBCalculator:
    """Calculate and forecast Share of Business (SOB) from historical data."""
    
    def __init__(self, market_share_file: Path):
        """
        Initialize SOB calculator.
        
        Args:
            market_share_file: Path to market share CSV file
        """
        self.market_share_file = market_share_file
        self.market_share_df = None
        self.historical_sob = {}
    
    def get_financial_year(self, date: pd.Timestamp) -> int:
        """
        Get financial year for a date.
        FY starts in April (month >= 4 means next year's FY).
        
        Args:
            date: Date timestamp
            
        Returns:
            Financial year (e.g., 2024 for FY 2024-2025)
        """
        if pd.isnull(date):
            return None
        return date.year + 1 if date.month >= 4 else date.year
    
    def load_market_share_data(self) -> pd.DataFrame:
        """Load and prepare market share data."""
        logger.info("Loading market share data...")
        
        df = pd.read_csv(self.market_share_file)
        df['YearMonth'] = pd.to_datetime(df['YearMonth'], format='%Y-%m')
        df['FinancialYear'] = df['YearMonth'].apply(self.get_financial_year)
        df['Month'] = df['YearMonth'].dt.month
        
        # Sort by date
        df = df.sort_values('YearMonth').reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} months of market share data")
        logger.info(f"Date range: {df['YearMonth'].min()} to {df['YearMonth'].max()}")
        
        self.market_share_df = df
        return df
    
    def calculate_historical_sob_by_fy(self, branch: str) -> pd.DataFrame:
        """
        Calculate average SOB for each complete financial year.
        
        Args:
            branch: Branch name (e.g., 'BLR', 'MAA')
            
        Returns:
            DataFrame with FY, average SOB, and statistics
        """
        if self.market_share_df is None:
            self.load_market_share_data()
        
        df = self.market_share_df.copy()
        
        # Get SOB column for branch
        sob_col = branch
        if sob_col not in df.columns:
            raise ValueError(f"Branch {branch} not found in market share data")
        
        # Group by financial year and calculate statistics
        fy_stats = df.groupby('FinancialYear').agg({
            sob_col: ['mean', 'std', 'min', 'max', 'count']
        }).reset_index()
        
        fy_stats.columns = ['FinancialYear', 'avg_sob', 'std_sob', 'min_sob', 'max_sob', 'month_count']
        
        # Filter to complete FYs (12 months)
        complete_fys = fy_stats[fy_stats['month_count'] == 12].copy()
        
        # Calculate overall statistics from complete FYs
        if len(complete_fys) > 0:
            overall_mean = complete_fys['avg_sob'].mean()
            overall_std = complete_fys['avg_sob'].std()
            overall_min = complete_fys['min_sob'].min()
            overall_max = complete_fys['max_sob'].max()
            
            logger.info(f"\n{branch} SOB Statistics (from {len(complete_fys)} complete FYs):")
            logger.info(f"  Mean: {overall_mean:.4f} ({overall_mean*100:.2f}%)")
            logger.info(f"  Std: {overall_std:.4f} ({overall_std*100:.2f}%)")
            logger.info(f"  Range: {overall_min:.4f} - {overall_max:.4f} ({overall_min*100:.2f}% - {overall_max*100:.2f}%)")
            logger.info(f"  Complete FYs: {sorted(complete_fys['FinancialYear'].tolist())}")
            
            self.historical_sob[branch] = {
                'mean': overall_mean,
                'std': overall_std,
                'min': overall_min,
                'max': overall_max,
                'complete_fys': complete_fys
            }
        else:
            logger.warning(f"No complete FYs found for {branch}")
            self.historical_sob[branch] = {
                'mean': df[sob_col].mean(),
                'std': df[sob_col].std(),
                'min': df[sob_col].min(),
                'max': df[sob_col].max(),
                'complete_fys': pd.DataFrame()
            }
        
        return complete_fys
    
    def forecast_sob(
        self,
        branch: str,
        n_periods: int = 12,
        model_type: str = "prophet"
    ) -> pd.Series:
        """
        Forecast SOB for next N months.
        
        Args:
            branch: Branch name
            n_periods: Number of months to forecast
            model_type: Model to use ('prophet', 'holt_winters', 'auto_arima')
            
        Returns:
            Forecasted SOB series
        """
        if self.market_share_df is None:
            self.load_market_share_data()
        
        df = self.market_share_df.copy()
        sob_col = branch
        
        # Create time series
        sob_series = pd.Series(
            df[sob_col].values,
            index=df['YearMonth']
        )
        
        # Filter to data up to calibration end date
        cutoff_date = pd.to_datetime(CALIBRATION_END_DATE)
        sob_series = sob_series[sob_series.index <= cutoff_date]
        
        if len(sob_series) < 12:
            logger.warning(f"Insufficient data for {branch} SOB forecasting, using historical mean")
            historical_mean = self.historical_sob.get(branch, {}).get('mean', sob_series.mean())
            last_date = sob_series.index.max()
            future_dates = create_future_dates(last_date, n_periods, 'MS')
            return pd.Series([historical_mean] * n_periods, index=future_dates)
        
        logger.info(f"Forecasting {branch} SOB for {n_periods} months using {model_type}...")
        
        # Train model
        model_trainer = ModelTrainer(random_state=42)
        
        if model_type == "prophet":
            model = model_trainer.train_prophet(sob_series)
            if model is None:
                logger.warning(f"Prophet failed for {branch}, using historical mean")
                historical_mean = self.historical_sob.get(branch, {}).get('mean', sob_series.mean())
                last_date = sob_series.index.max()
                future_dates = create_future_dates(last_date, n_periods, 'MS')
                return pd.Series([historical_mean] * n_periods, index=future_dates)
            
            # Forecast
            last_date = sob_series.index.max()
            future_dates = create_future_dates(last_date, n_periods, 'MS')
            forecast = model_trainer.forecast_prophet(model, n_periods, last_date, freq='MS')
            
        elif model_type == "holt_winters":
            model = model_trainer.train_holt_winters(sob_series)
            if model is None:
                logger.warning(f"Holt-Winters failed for {branch}, using historical mean")
                historical_mean = self.historical_sob.get(branch, {}).get('mean', sob_series.mean())
                last_date = sob_series.index.max()
                future_dates = create_future_dates(last_date, n_periods, 'MS')
                return pd.Series([historical_mean] * n_periods, index=future_dates)
            
            last_date = sob_series.index.max()
            forecast = model_trainer.forecast_holt_winters(model, n_periods, last_date, freq='MS')
            
        elif model_type == "auto_arima":
            model = model_trainer.train_auto_arima(sob_series)
            if model is None:
                logger.warning(f"Auto-ARIMA failed for {branch}, using historical mean")
                historical_mean = self.historical_sob.get(branch, {}).get('mean', sob_series.mean())
                last_date = sob_series.index.max()
                future_dates = create_future_dates(last_date, n_periods, 'MS')
                return pd.Series([historical_mean] * n_periods, index=future_dates)
            
            last_date = sob_series.index.max()
            forecast = model_trainer.forecast_auto_arima(model, n_periods, last_date, freq='MS')
            
        else:
            # Default: use historical mean
            logger.warning(f"Unknown model type {model_type}, using historical mean")
            historical_mean = self.historical_sob.get(branch, {}).get('mean', sob_series.mean())
            last_date = sob_series.index.max()
            future_dates = create_future_dates(last_date, n_periods, 'MS')
            return pd.Series([historical_mean] * n_periods, index=future_dates)
        
        # Ensure SOB is between 0 and 1
        forecast = forecast.clip(lower=0, upper=1)
        
        logger.info(f"  {branch} SOB forecast range: {forecast.min():.4f} - {forecast.max():.4f}")
        logger.info(f"  {branch} SOB forecast mean: {forecast.mean():.4f} ({forecast.mean()*100:.2f}%)")
        
        return forecast


class ProductionForecaster:
    """Production forecasting using best models and dynamic SOB-based market estimation."""
    
    def __init__(
        self,
        model_name: str = "catboost",
        sob_model_type: str = "prophet"
    ):
        """
        Initialize production forecaster.
        
        Args:
            model_name: Model to use for sales forecasting (default: 'catboost')
            sob_model_type: Model to use for SOB forecasting (default: 'prophet')
        """
        self.model_name = model_name
        self.sob_model_type = sob_model_type
        
        # Initialize components
        self.data_preparator = DataPreparator(DATA_FILE, missing_month="2020-04")
        self.feature_engineer = FeatureEngineer(
            lag_periods=LAG_PERIODS,
            rolling_windows=ROLLING_WINDOWS,
            include_seasonality=INCLUDE_SEASONALITY,
            include_trend=INCLUDE_TREND,
        )
        self.model_trainer = ModelTrainer(random_state=42)
        
        # SOB calculator
        market_share_file = DATA_DIR / "market_share_by_month_year.csv"
        self.sob_calculator = SOBCalculator(market_share_file)
        
        # Load models and data
        self.blr_model = None
        self.maa_model = None
        self.blr_series = None
        self.maa_series = None
        self.blr_feature_names = None
        self.maa_feature_names = None
        
        # Forecasted SOB
        self.blr_sob_forecast = None
        self.maa_sob_forecast = None
    
    def get_financial_year(self, date: pd.Timestamp) -> int:
        """Get financial year for a date (FY starts in April)."""
        if pd.isnull(date):
            return None
        return date.year + 1 if date.month >= 4 else date.year
    
    def load_models_and_data(self):
        """Load trained models and historical data."""
        logger.info("Loading models and historical data...")
        
        # Load aggregated data
        aggregated_data = self.data_preparator.prepare_aggregated_data()
        monthly_branch_data = aggregated_data['monthly']['branch_wise']
        
        # Get data up to calibration end date
        historical_data = monthly_branch_data[
            monthly_branch_data['Date'] <= CALIBRATION_END_DATE
        ].copy()
        
        # Prepare series for BLR and MAA
        series_dict = self.data_preparator.prepare_series(
            historical_data, value_col='Quantity', group_by='Branch'
        )
        
        self.blr_series = series_dict.get('BLR')
        self.maa_series = series_dict.get('MAA')
        
        if self.blr_series is None:
            raise ValueError("BLR branch data not found")
        if self.maa_series is None:
            raise ValueError("MAA branch data not found")
        
        logger.info(f"BLR series: {len(self.blr_series)} periods, {self.blr_series.index.min()} to {self.blr_series.index.max()}")
        logger.info(f"MAA series: {len(self.maa_series)} periods, {self.maa_series.index.min()} to {self.maa_series.index.max()}")
        
        # Load models
        blr_model_path = MODEL_DIR / f"monthly_branch_wise_BLR_{self.model_name}.pkl"
        maa_model_path = MODEL_DIR / f"monthly_branch_wise_MAA_{self.model_name}.pkl"
        
        if not blr_model_path.exists():
            raise FileNotFoundError(f"BLR model not found: {blr_model_path}")
        if not maa_model_path.exists():
            raise FileNotFoundError(f"MAA model not found: {maa_model_path}")
        
        self.blr_model = self.model_trainer.load_model(self.model_name, blr_model_path)
        self.maa_model = self.model_trainer.load_model(self.model_name, maa_model_path)
        
        if self.blr_model is None or self.maa_model is None:
            raise ValueError("Failed to load models")
        
        # Get feature names (needed for ML models)
        _, _, self.blr_feature_names = self.feature_engineer.prepare_ml_features(self.blr_series)
        _, _, self.maa_feature_names = self.feature_engineer.prepare_ml_features(self.maa_series)
        
        logger.info("Models and data loaded successfully")
    
    def calculate_and_forecast_sob(self, n_periods: int = 12):
        """Calculate historical SOB and forecast for next N months."""
        logger.info("=" * 80)
        logger.info("CALCULATING AND FORECASTING SOB")
        logger.info("=" * 80)
        
        # Calculate historical SOB from complete FYs for all branches
        logger.info("\nCalculating historical SOB from complete financial years...")
        # Get all branches from market share data
        if self.sob_calculator.market_share_df is None:
            self.sob_calculator.load_market_share_data()
        
        all_branches = [col for col in self.sob_calculator.market_share_df.columns 
                       if col not in ['YearMonth', 'FinancialYear', 'Month']]
        
        for branch in all_branches:
            try:
                self.sob_calculator.calculate_historical_sob_by_fy(branch)
            except Exception as e:
                logger.warning(f"Failed to calculate SOB for {branch}: {e}")
        
        # Forecast SOB for next N months
        logger.info(f"\nForecasting SOB for next {n_periods} months...")
        self.blr_sob_forecast = self.sob_calculator.forecast_sob(
            'BLR', n_periods, self.sob_model_type
        )
        self.maa_sob_forecast = self.sob_calculator.forecast_sob(
            'MAA', n_periods, self.sob_model_type
        )
    
    def generate_forecast(
        self,
        model: Any,
        series: pd.Series,
        feature_names: list,
        n_periods: int = 12
    ) -> pd.Series:
        """
        Generate forecast for a branch.
        
        Args:
            model: Trained model
            series: Historical time series
            feature_names: Feature names for ML models
            n_periods: Number of periods to forecast
            
        Returns:
            Forecast series
        """
        freq, _ = detect_frequency(series)
        last_date = series.index.max()
        
        # Get last values for lag features
        max_lag_needed = max(self.feature_engineer.lag_periods) if self.feature_engineer.lag_periods else 12
        last_values = series.tail(max(12, max_lag_needed)).values
        
        # Generate future dates
        future_dates = create_future_dates(last_date, n_periods, freq)
        
        forecasts = []
        current_values = list(last_values)
        
        for i, future_date in enumerate(future_dates):
            # Create features
            features_dict = self.feature_engineer.create_features_for_forecast(
                current_values, future_date, len(series) + i
            )
            
            # Create feature vector
            feature_vector = np.array(
                [features_dict.get(name, 0) for name in feature_names]
            ).reshape(1, -1)
            
            # Predict
            if self.model_name == "catboost":
                pred = self.model_trainer.forecast_catboost(model, feature_vector)[0]
            elif self.model_name == "lightgbm":
                pred = self.model_trainer.forecast_lightgbm(model, feature_vector)[0]
            elif self.model_name == "xgboost":
                pred = self.model_trainer.forecast_xgboost(model, feature_vector)[0]
            elif self.model_name == "random_forest":
                pred = self.model_trainer.forecast_random_forest(model, feature_vector)[0]
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
            
            pred = max(0, pred)  # Ensure non-negative
            forecasts.append(pred)
            current_values.append(pred)
            
            # Keep enough values for lag features
            max_lag = max(self.feature_engineer.lag_periods) if self.feature_engineer.lag_periods else 12
            if len(current_values) > max_lag:
                current_values.pop(0)
        
        return pd.Series(forecasts, index=future_dates)
    
    def calculate_yearly_sob(self) -> Dict[str, float]:
        """
        Calculate yearly SOB for BLR and MAA from historical complete FYs.
        
        Returns:
            Dictionary with yearly SOB for each branch
        """
        logger.info("\nCalculating yearly SOB from historical complete FYs...")
        
        yearly_sob = {}
        
        # Get yearly SOB for BLR and MAA
        for branch in ['BLR', 'MAA']:
            if branch in self.sob_calculator.historical_sob:
                # Use mean SOB from complete FYs as yearly SOB
                yearly_sob[branch] = self.sob_calculator.historical_sob[branch]['mean']
                logger.info(f"  {branch} Yearly SOB: {yearly_sob[branch]:.4f} ({yearly_sob[branch]*100:.2f}%)")
            else:
                logger.warning(f"  {branch} SOB not found, calculating...")
                self.sob_calculator.calculate_historical_sob_by_fy(branch)
                yearly_sob[branch] = self.sob_calculator.historical_sob[branch]['mean']
        
        return yearly_sob
    
    def calculate_monthly_distribution(self, branch: str) -> pd.Series:
        """
        Calculate historical monthly distribution pattern for a branch.
        Returns percentage of yearly sales for each month (1-12).
        
        Args:
            branch: Branch name
            
        Returns:
            Series with month (1-12) as index and percentage as values
        """
        if self.sob_calculator.market_share_df is None:
            self.sob_calculator.load_market_share_data()
        
        df = self.sob_calculator.market_share_df.copy()
        
        # Get complete FYs only
        complete_fys = self.sob_calculator.historical_sob.get(branch, {}).get('complete_fys', pd.DataFrame())
        if len(complete_fys) == 0:
            # Calculate if not already done
            self.sob_calculator.calculate_historical_sob_by_fy(branch)
            complete_fys = self.sob_calculator.historical_sob.get(branch, {}).get('complete_fys', pd.DataFrame())
        
        if len(complete_fys) == 0:
            logger.warning(f"No complete FYs for {branch}, using all data")
            complete_fy_list = df['FinancialYear'].unique()
        else:
            complete_fy_list = complete_fys['FinancialYear'].tolist()
        
        # Filter to complete FYs
        df_complete = df[df['FinancialYear'].isin(complete_fy_list)].copy()
        
        # Group by month and calculate average SOB
        monthly_sob = df_complete.groupby('Month')[branch].mean()
        
        # Normalize to sum to 1 (percentage distribution)
        monthly_distribution = monthly_sob / monthly_sob.sum()
        
        logger.info(f"  {branch} monthly distribution calculated from {len(complete_fy_list)} complete FYs")
        
        return monthly_distribution
    
    def estimate_total_market_sales_yearly(
        self,
        blr_yearly_sales: float,
        maa_yearly_sales: float,
        blr_yearly_sob: float,
        maa_yearly_sob: float,
        use_weighted_mean: bool = True
    ) -> Dict[str, float]:
        """
        Estimate total market sales using yearly SOB.
        
        Formula: Total Market = (BLR Sales / BLR SOB + MAA Sales / MAA SOB) / 2
        Or weighted: Total Market = weighted average
        
        Args:
            blr_yearly_sales: BLR yearly sales
            maa_yearly_sales: MAA yearly sales
            blr_yearly_sob: BLR yearly SOB
            maa_yearly_sob: MAA yearly SOB
            use_weighted_mean: If True, use weighted mean
            
        Returns:
            Dictionary with total market estimate and bounds
        """
        # Estimate total market from each branch
        total_from_blr = blr_yearly_sales / blr_yearly_sob
        total_from_maa = maa_yearly_sales / maa_yearly_sob
        
        if use_weighted_mean:
            # Weight by SOB
            weights_blr = blr_yearly_sob / (blr_yearly_sob + maa_yearly_sob)
            weights_maa = maa_yearly_sob / (blr_yearly_sob + maa_yearly_sob)
            total_market = total_from_blr * weights_blr + total_from_maa * weights_maa
        else:
            # Simple average
            total_market = (total_from_blr + total_from_maa) / 2
        
        # Calculate bounds using historical variance
        blr_std = self.sob_calculator.historical_sob.get('BLR', {}).get('std', 0.01)
        maa_std = self.sob_calculator.historical_sob.get('MAA', {}).get('std', 0.01)
        
        blr_sob_min = max(0.001, blr_yearly_sob - blr_std)
        blr_sob_max = min(1.0, blr_yearly_sob + blr_std)
        maa_sob_min = max(0.001, maa_yearly_sob - maa_std)
        maa_sob_max = min(1.0, maa_yearly_sob + maa_std)
        
        total_from_blr_min = blr_yearly_sales / blr_sob_max
        total_from_blr_max = blr_yearly_sales / blr_sob_min
        total_from_maa_min = maa_yearly_sales / maa_sob_max
        total_from_maa_max = maa_yearly_sales / maa_sob_min
        
        if use_weighted_mean:
            total_min = total_from_blr_min * weights_blr + total_from_maa_min * weights_maa
            total_max = total_from_blr_max * weights_blr + total_from_maa_max * weights_maa
        else:
            total_min = (total_from_blr_min + total_from_maa_min) / 2
            total_max = (total_from_blr_max + total_from_maa_max) / 2
        
        return {
            'total_market': total_market,
            'total_market_min': total_min,
            'total_market_max': total_max,
            'total_from_blr': total_from_blr,
            'total_from_maa': total_from_maa
        }
    
    def estimate_total_market_sales(
        self,
        blr_forecast: pd.Series,
        maa_forecast: pd.Series,
        use_weighted_mean: bool = True
    ) -> pd.DataFrame:
        """
        Estimate total market sales using forecasted SOB.
        
        Formula: Total Market = Branch Sales / SOB
        
        Args:
            blr_forecast: BLR branch forecast
            maa_forecast: MAA branch forecast
            use_weighted_mean: If True, use weighted mean (weighted by SOB)
            
        Returns:
            DataFrame with total market estimates and bounds
        """
        # Align forecasts on common dates
        common_dates = blr_forecast.index.intersection(maa_forecast.index)
        common_dates_sob = self.blr_sob_forecast.index.intersection(self.maa_sob_forecast.index)
        common_dates = common_dates.intersection(common_dates_sob)
        
        if len(common_dates) == 0:
            logger.warning("No common dates between forecasts and SOB")
            return pd.DataFrame()
        
        blr_aligned = blr_forecast.loc[common_dates]
        maa_aligned = maa_forecast.loc[common_dates]
        blr_sob_aligned = self.blr_sob_forecast.loc[common_dates]
        maa_sob_aligned = self.maa_sob_forecast.loc[common_dates]
        
        # Estimate total market from each branch using forecasted SOB
        total_from_blr = blr_aligned / blr_sob_aligned
        total_from_maa = maa_aligned / maa_sob_aligned
        
        if use_weighted_mean:
            # Weight by SOB (larger SOB = more reliable estimate)
            # Normalize weights so they sum to 1
            weights_blr = blr_sob_aligned / (blr_sob_aligned + maa_sob_aligned)
            weights_maa = maa_sob_aligned / (blr_sob_aligned + maa_sob_aligned)
            
            # Weighted mean
            total_market = total_from_blr * weights_blr + total_from_maa * weights_maa
        else:
            # Simple average
            total_market = (total_from_blr + total_from_maa) / 2
        
        # Calculate variance from historical SOB
        blr_historical_std = self.sob_calculator.historical_sob.get('BLR', {}).get('std', 0.01)
        maa_historical_std = self.sob_calculator.historical_sob.get('MAA', {}).get('std', 0.01)
        
        # Bounds using ±1 std from forecasted SOB
        blr_sob_min = (blr_sob_aligned - blr_historical_std).clip(lower=0.001)
        blr_sob_max = (blr_sob_aligned + blr_historical_std).clip(upper=1.0)
        maa_sob_min = (maa_sob_aligned - maa_historical_std).clip(lower=0.001)
        maa_sob_max = (maa_sob_aligned + maa_historical_std).clip(upper=1.0)
        
        total_from_blr_min = blr_aligned / blr_sob_max
        total_from_blr_max = blr_aligned / blr_sob_min
        total_from_maa_min = maa_aligned / maa_sob_max
        total_from_maa_max = maa_aligned / maa_sob_min
        
        if use_weighted_mean:
            total_min = (
                total_from_blr_min * weights_blr + total_from_maa_min * weights_maa
            )
            total_max = (
                total_from_blr_max * weights_blr + total_from_maa_max * weights_maa
            )
        else:
            total_min = (total_from_blr_min + total_from_maa_min) / 2
            total_max = (total_from_blr_max + total_from_maa_max) / 2
        
        # Create result DataFrame
        result = pd.DataFrame({
            'date': common_dates,
            'total_market_estimate': total_market.values,
            'total_market_min': total_min.values,
            'total_market_max': total_max.values,
            'blr_forecast': blr_aligned.values,
            'maa_forecast': maa_aligned.values,
            'blr_sob_forecast': blr_sob_aligned.values,
            'maa_sob_forecast': maa_sob_aligned.values,
            'total_from_blr': total_from_blr.values,
            'total_from_maa': total_from_maa.values,
        })
        result.set_index('date', inplace=True)
        
        return result
    
    def get_yearly_sob(self, branch: str) -> float:
        """
        Get yearly SOB for a branch (average from complete FYs).
        
        Args:
            branch: Branch name
            
        Returns:
            Yearly SOB (average)
        """
        if branch in self.sob_calculator.historical_sob:
            return self.sob_calculator.historical_sob[branch]['mean']
        else:
            logger.warning(f"No historical SOB found for {branch}, using 0")
            return 0.0
    
    def get_monthly_distribution_pattern(self, branch: str) -> Dict[int, float]:
        """
        Get historical monthly distribution pattern for a branch.
        Returns percentage of yearly sales for each month (1-12).
        
        Args:
            branch: Branch name
            
        Returns:
            Dictionary mapping month (1-12) to percentage of yearly sales
        """
        if self.sob_calculator.market_share_df is None:
            self.sob_calculator.load_market_share_data()
        
        df = self.sob_calculator.market_share_df.copy()
        
        # Get complete FYs only
        fy_counts = df.groupby('FinancialYear').size()
        complete_fys = fy_counts[fy_counts == 12].index.tolist()
        
        if len(complete_fys) == 0:
            logger.warning(f"No complete FYs found for monthly distribution, using equal distribution")
            return {i: 1.0/12 for i in range(1, 13)}
        
        # Calculate monthly distribution patterns from complete FYs
        # For each FY, get monthly SOB values and normalize to sum to 1.0
        monthly_patterns = []
        for fy in complete_fys:
            fy_data = df[df['FinancialYear'] == fy].copy()
            if len(fy_data) == 12:
                # Get monthly SOB values for this FY
                monthly_sob = fy_data.set_index('Month')[branch].sort_index()
                # Normalize to sum to 1.0 (percentage of yearly sales per month)
                monthly_sob_normalized = monthly_sob / monthly_sob.sum()
                monthly_patterns.append(monthly_sob_normalized)
        
        if len(monthly_patterns) == 0:
            logger.warning(f"No monthly patterns found, using equal distribution")
            return {i: 1.0/12 for i in range(1, 13)}
        
        # Average normalized patterns across all FYs
        monthly_avg = pd.concat(monthly_patterns, axis=1).mean(axis=1)
        
        # Ensure it sums to 1.0 (should already, but just in case)
        monthly_avg_normalized = monthly_avg / monthly_avg.sum()
        
        return {int(month): float(pct) for month, pct in monthly_avg_normalized.items()}
    
    def calculate_fy_sales(
        self,
        blr_forecast: pd.Series,
        maa_forecast: pd.Series
    ) -> Dict[str, Any]:
        """
        Calculate yearly and branch-wise sales for the financial year using yearly SOB.
        
        Process:
        1. Calculate yearly SOB for BLR and MAA from complete FYs
        2. Sum BLR and MAA yearly sales
        3. Use yearly SOB to calculate total market sales
        4. Use branch-wise SOB to distribute total sales to each branch
        5. Calculate monthly distributions for each branch
        
        Args:
            blr_forecast: BLR branch forecast (monthly)
            maa_forecast: MAA branch forecast (monthly)
            
        Returns:
            Dictionary with FY sales summaries and monthly distributions
        """
        logger.info("\n" + "=" * 80)
        logger.info("CALCULATING FINANCIAL YEAR SALES USING YEARLY SOB")
        logger.info("=" * 80)
        
        # Prepare dataframes with financial year
        blr_df = pd.DataFrame({
            'date': blr_forecast.index,
            'forecast': blr_forecast.values
        })
        blr_df['date'] = pd.to_datetime(blr_df['date'])
        blr_df['FinancialYear'] = blr_df['date'].apply(self.get_financial_year)
        blr_df['Month'] = blr_df['date'].dt.month
        
        maa_df = pd.DataFrame({
            'date': maa_forecast.index,
            'forecast': maa_forecast.values
        })
        maa_df['date'] = pd.to_datetime(maa_df['date'])
        maa_df['FinancialYear'] = maa_df['date'].apply(self.get_financial_year)
        maa_df['Month'] = maa_df['date'].dt.month
        
        # Get unique financial years in forecast
        forecast_fys = sorted(set(blr_df['FinancialYear'].unique()) | set(maa_df['FinancialYear'].unique()))
        
        # Get yearly SOB for BLR and MAA
        blr_yearly_sob = self.get_yearly_sob('BLR')
        maa_yearly_sob = self.get_yearly_sob('MAA')
        
        logger.info(f"\nYearly SOB (from complete FYs):")
        logger.info(f"  BLR: {blr_yearly_sob:.4f} ({blr_yearly_sob*100:.2f}%)")
        logger.info(f"  MAA: {maa_yearly_sob:.4f} ({maa_yearly_sob*100:.2f}%)")
        
        fy_summaries = {}
        all_monthly_distributions = {}
        
        for fy in forecast_fys:
            logger.info(f"\nFinancial Year {fy}-{fy+1}:")
            
            # Filter data for this FY
            fy_blr = blr_df[blr_df['FinancialYear'] == fy]
            fy_maa = maa_df[maa_df['FinancialYear'] == fy]
            
            # Calculate yearly sales for BLR and MAA
            blr_yearly_sales = fy_blr['forecast'].sum()
            maa_yearly_sales = fy_maa['forecast'].sum()
            
            logger.info(f"  BLR Yearly Sales: {blr_yearly_sales:,.0f} units")
            logger.info(f"  MAA Yearly Sales: {maa_yearly_sales:,.0f} units")
            
            # Calculate total market sales using yearly SOB
            # Formula: Total Market = (BLR Sales / BLR SOB + MAA Sales / MAA SOB) / 2
            # Or use weighted mean
            total_from_blr = blr_yearly_sales / blr_yearly_sob if blr_yearly_sob > 0 else 0
            total_from_maa = maa_yearly_sales / maa_yearly_sob if maa_yearly_sob > 0 else 0
            
            # Weighted mean by SOB
            if blr_yearly_sob > 0 and maa_yearly_sob > 0:
                weights_blr = blr_yearly_sob / (blr_yearly_sob + maa_yearly_sob)
                weights_maa = maa_yearly_sob / (blr_yearly_sob + maa_yearly_sob)
                total_market_sales = total_from_blr * weights_blr + total_from_maa * weights_maa
            else:
                total_market_sales = (total_from_blr + total_from_maa) / 2
            
            # Calculate bounds using historical variance
            blr_std = self.sob_calculator.historical_sob.get('BLR', {}).get('std', 0.01)
            maa_std = self.sob_calculator.historical_sob.get('MAA', {}).get('std', 0.01)
            
            blr_sob_min = max(0.001, blr_yearly_sob - blr_std)
            blr_sob_max = min(1.0, blr_yearly_sob + blr_std)
            maa_sob_min = max(0.001, maa_yearly_sob - maa_std)
            maa_sob_max = min(1.0, maa_yearly_sob + maa_std)
            
            total_from_blr_min = blr_yearly_sales / blr_sob_max
            total_from_blr_max = blr_yearly_sales / blr_sob_min
            total_from_maa_min = maa_yearly_sales / maa_sob_max
            total_from_maa_max = maa_yearly_sales / maa_sob_min
            
            if blr_yearly_sob > 0 and maa_yearly_sob > 0:
                total_market_min = total_from_blr_min * weights_blr + total_from_maa_min * weights_maa
                total_market_max = total_from_blr_max * weights_blr + total_from_maa_max * weights_maa
            else:
                total_market_min = (total_from_blr_min + total_from_maa_min) / 2
                total_market_max = (total_from_blr_max + total_from_maa_max) / 2
            
            logger.info(f"  Total Market Sales: {total_market_sales:,.0f} units")
            logger.info(f"  Range: {total_market_min:,.0f} - {total_market_max:,.0f} units")
            
            # Calculate branch-wise sales using branch SOB
            branch_sales = {}
            branch_sales['BLR'] = blr_yearly_sales  # Already calculated
            branch_sales['MAA'] = maa_yearly_sales  # Already calculated
            
            # Calculate other branches using their yearly SOB
            for branch, sob_data in self.sob_calculator.historical_sob.items():
                if branch not in ['BLR', 'MAA']:
                    branch_yearly_sob = sob_data['mean']
                    branch_yearly_sales = total_market_sales * branch_yearly_sob
                    branch_sales[branch] = branch_yearly_sales
                    logger.info(f"  {branch} Yearly Sales: {branch_yearly_sales:,.0f} units (SOB: {branch_yearly_sob*100:.2f}%)")
            
            # Calculate monthly distributions for each branch
            monthly_distributions = {}
            for branch in branch_sales.keys():
                monthly_pattern = self.get_monthly_distribution_pattern(branch)
                branch_yearly_total = branch_sales[branch]
                
                # Calculate monthly sales
                branch_monthly = {}
                for month in range(1, 13):
                    month_pct = monthly_pattern.get(month, 1.0/12)
                    month_sales = branch_yearly_total * month_pct
                    branch_monthly[month] = month_sales
                
                monthly_distributions[branch] = branch_monthly
                logger.info(f"  {branch} monthly distribution calculated")
            
            fy_summaries[f"FY_{fy}"] = {
                'financial_year': f"{fy}-{fy+1}",
                'total_market_sales': total_market_sales,
                'total_market_min': total_market_min,
                'total_market_max': total_market_max,
                'branch_sales': branch_sales,
                'monthly_distributions': monthly_distributions,
                'month_count': len(fy_blr) + len(fy_maa)
            }
            
            all_monthly_distributions[f"FY_{fy}"] = monthly_distributions
        
        return {
            'fy_summaries': fy_summaries,
            'monthly_distributions': all_monthly_distributions
        }
    
    def run_production_forecast(
        self,
        n_periods: int = 12,
        output_path: Optional[Path] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Run production forecast for next N months.
        
        Args:
            n_periods: Number of months to forecast (default: 12)
            output_path: Path to save results (optional)
            
        Returns:
            Dictionary with forecasts and total market estimates
        """
        logger.info("=" * 80)
        logger.info("PRODUCTION FORECASTING")
        logger.info("=" * 80)
        
        # Calculate and forecast SOB
        self.calculate_and_forecast_sob(n_periods)
        
        # Load models and data
        self.load_models_and_data()
        
        # Generate forecasts
        logger.info(f"\nGenerating {n_periods}-month sales forecasts...")
        
        logger.info("  Forecasting BLR...")
        blr_forecast = self.generate_forecast(
            self.blr_model, self.blr_series, self.blr_feature_names, n_periods
        )
        logger.info(f"    BLR forecast: {blr_forecast.sum():.0f} total units over {len(blr_forecast)} months")
        
        logger.info("  Forecasting MAA...")
        maa_forecast = self.generate_forecast(
            self.maa_model, self.maa_series, self.maa_feature_names, n_periods
        )
        logger.info(f"    MAA forecast: {maa_forecast.sum():.0f} total units over {len(maa_forecast)} months")
        
        # Estimate total market sales
        logger.info("\nEstimating total market sales using forecasted SOB...")
        total_market_estimate = self.estimate_total_market_sales(
            blr_forecast, maa_forecast, use_weighted_mean=True
        )
        
        logger.info(f"\nTotal Market Estimate:")
        logger.info(f"  Total: {total_market_estimate['total_market_estimate'].sum():.0f} units")
        logger.info(f"  Range: {total_market_estimate['total_market_min'].sum():.0f} - {total_market_estimate['total_market_max'].sum():.0f} units")
        
        # Calculate FY sales using yearly SOB
        fy_results = self.calculate_fy_sales(blr_forecast, maa_forecast)
        fy_summaries = fy_results['fy_summaries']
        monthly_distributions = fy_results['monthly_distributions']
        
        # Prepare results
        results = {
            'blr_forecast': pd.DataFrame({
                'date': blr_forecast.index,
                'forecast': blr_forecast.values
            }),
            'maa_forecast': pd.DataFrame({
                'date': maa_forecast.index,
                'forecast': maa_forecast.values
            }),
            'blr_sob_forecast': pd.DataFrame({
                'date': self.blr_sob_forecast.index,
                'sob_forecast': self.blr_sob_forecast.values
            }),
            'maa_sob_forecast': pd.DataFrame({
                'date': self.maa_sob_forecast.index,
                'sob_forecast': self.maa_sob_forecast.values
            }),
            'total_market': total_market_estimate.reset_index(),
            'fy_summaries': fy_summaries,
            'monthly_distributions': monthly_distributions
        }
        
        # Create FY summary DataFrame
        fy_summary_rows = []
        for fy_key, fy_data in fy_summaries.items():
            row = {
                'FinancialYear': fy_data['financial_year'],
                'TotalMarketSales': fy_data['total_market_sales'],
                'TotalMarketMin': fy_data['total_market_min'],
                'TotalMarketMax': fy_data['total_market_max'],
                'MonthCount': fy_data['month_count']
            }
            # Add branch sales
            for branch, sales in fy_data['branch_sales'].items():
                row[f'{branch}_Sales'] = sales
            fy_summary_rows.append(row)
        
        results['fy_summary'] = pd.DataFrame(fy_summary_rows)
        
        # Create monthly distribution DataFrames for each FY
        monthly_dist_dfs = {}
        for fy_key, fy_monthly in monthly_distributions.items():
            monthly_rows = []
            for branch, branch_monthly in fy_monthly.items():
                for month, sales in branch_monthly.items():
                    monthly_rows.append({
                        'FinancialYear': fy_key,
                        'Branch': branch,
                        'Month': month,
                        'Sales': sales
                    })
            monthly_dist_dfs[fy_key] = pd.DataFrame(monthly_rows)
        
        results['monthly_distributions_df'] = monthly_dist_dfs
        
        # Save results
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            results['blr_forecast'].to_csv(
                output_path.parent / f"{output_path.stem}_blr_forecast.csv",
                index=False
            )
            results['maa_forecast'].to_csv(
                output_path.parent / f"{output_path.stem}_maa_forecast.csv",
                index=False
            )
            results['blr_sob_forecast'].to_csv(
                output_path.parent / f"{output_path.stem}_blr_sob_forecast.csv",
                index=False
            )
            results['maa_sob_forecast'].to_csv(
                output_path.parent / f"{output_path.stem}_maa_sob_forecast.csv",
                index=False
            )
            results['total_market'].to_csv(
                output_path.parent / f"{output_path.stem}_total_market.csv",
                index=False
            )
            results['fy_summary'].to_csv(
                output_path.parent / f"{output_path.stem}_fy_summary.csv",
                index=False
            )
            
            # Save monthly distributions for each FY
            for fy_key, monthly_df in monthly_dist_dfs.items():
                monthly_df.to_csv(
                    output_path.parent / f"{output_path.stem}_{fy_key}_monthly_distribution.csv",
                    index=False
                )
            
            logger.info(f"\nResults saved to {output_path.parent}")
        
        return results


def main():
    """Main function for production forecasting."""
    forecaster = ProductionForecaster(
        model_name="catboost",
        sob_model_type="prophet"  # Use Prophet for SOB forecasting
    )
    
    # Generate 12-month forecast
    results = forecaster.run_production_forecast(
        n_periods=12,
        output_path=Path("forecast_outputs/production") / f"production_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("PRODUCTION FORECAST SUMMARY")
    print("=" * 80)
    print(f"\nBLR Forecast (next 12 months):")
    print(results['blr_forecast'].to_string(index=False))
    print(f"\nMAA Forecast (next 12 months):")
    print(results['maa_forecast'].to_string(index=False))
    print(f"\nBLR SOB Forecast (next 12 months):")
    print(results['blr_sob_forecast'].to_string(index=False))
    print(f"\nMAA SOB Forecast (next 12 months):")
    print(results['maa_sob_forecast'].to_string(index=False))
    print(f"\nTotal Market Estimate (next 12 months):")
    print(results['total_market'][['date', 'total_market_estimate', 'total_market_min', 'total_market_max', 'blr_sob_forecast', 'maa_sob_forecast']].to_string(index=False))
    print(f"\nFinancial Year Summary:")
    print(results['fy_summary'].to_string(index=False))
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

