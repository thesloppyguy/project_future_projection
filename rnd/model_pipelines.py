"""
Unified model pipelines for classical and deep learning time series models.

Supported models:
- Prophet
- ETS (Exponential Smoothing)
- TBATS
- LSTM (Keras)
- DeepAR (via Darts' RNN/DeepAR-like)
- Transformer (via Darts' TransformerModel)

All pipelines expose a consistent interface: fit(ts, past_covariates, future_covariates) -> self,
forecast(n, past_covariates, future_covariates) -> pd.Series, and save/load helpers.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# Classical models
from statsmodels.tsa.holtwinters import ExponentialSmoothing
try:
    from tbats import TBATS
    _TBATS_IMPORT_ERROR = None
except Exception as _e:  # Defer hard failure until TBATS is actually used
    TBATS = None  # type: ignore
    _TBATS_IMPORT_ERROR = _e

# Prophet
from prophet import Prophet

# Darts for deep learning/transformers and unified interfaces
from darts import TimeSeries
from darts.models import (
    RNNModel,
    TransformerModel,
)

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


def to_timeseries(df: pd.DataFrame, time_col: str, value_col: str) -> TimeSeries:
    series = TimeSeries.from_dataframe(df, time_col, value_col)
    return series


def ensure_datetime_index(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    out = out.sort_values(time_col).reset_index(drop=True)
    return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {"MAE": float(mae), "MSE": float(mse), "RMSE": rmse, "MAPE": float(mape)}


class BasePipeline:
    def fit(self, train_df: pd.DataFrame, time_col: str, value_col: str,
            past_covariates: Optional[pd.DataFrame] = None,
            future_covariates: Optional[pd.DataFrame] = None) -> "BasePipeline":
        raise NotImplementedError

    def forecast(self, n: int, time_col: str, value_col: str,
                 past_covariates: Optional[pd.DataFrame] = None,
                 future_covariates: Optional[pd.DataFrame] = None) -> pd.Series:
        raise NotImplementedError

    def evaluate(self, test_df: pd.DataFrame, preds: pd.Series, value_col: str) -> Dict[str, float]:
        mask = (~preds.isna()) & (~test_df[value_col].isna())
        return compute_metrics(test_df.loc[mask, value_col].values, preds.loc[mask].values)


class ProphetPipeline(BasePipeline):
    def __init__(self, yearly_seasonality: str | bool = "auto",
                 weekly_seasonality: str | bool = False,
                 daily_seasonality: str | bool = False,
                 changepoint_prior_scale: float = 0.05):
        self.model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            changepoint_prior_scale=changepoint_prior_scale,
        )

    def fit(self, train_df: pd.DataFrame, time_col: str, value_col: str,
            past_covariates: Optional[pd.DataFrame] = None,
            future_covariates: Optional[pd.DataFrame] = None) -> "ProphetPipeline":
        df = train_df[[time_col, value_col]].rename(columns={time_col: "ds", value_col: "y"}).copy()
        if past_covariates is not None:
            for col in past_covariates.columns:
                self.model.add_regressor(col)
            df = df.merge(past_covariates, left_on="ds", right_on=past_covariates.columns[0], how="left") if past_covariates.columns[0] == "ds" else df.join(past_covariates.set_index(past_covariates.columns[0]), on="ds")
        self.model.fit(df)
        return self

    def forecast(self, n: int, time_col: str, value_col: str,
                 past_covariates: Optional[pd.DataFrame] = None,
                 future_covariates: Optional[pd.DataFrame] = None) -> pd.Series:
        # Expect future covariates to include the future dates
        if future_covariates is not None and "ds" in future_covariates.columns:
            future_df = future_covariates.copy()
        else:
            raise ValueError("Prophet requires future_covariates with a 'ds' column for future dates")
        forecast = self.model.predict(future_df)
        return pd.Series(forecast["yhat"].values, index=future_df["ds"])


class ETSPipeline(BasePipeline):
    def __init__(self, trend: str | None = "add", seasonal: str | None = "add", seasonal_periods: int = 12):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model = None

    def fit(self, train_df: pd.DataFrame, time_col: str, value_col: str,
            past_covariates: Optional[pd.DataFrame] = None,
            future_covariates: Optional[pd.DataFrame] = None) -> "ETSPipeline":
        y = train_df[value_col].astype(float).values
        self.model = ExponentialSmoothing(
            y, trend=self.trend, seasonal=self.seasonal, seasonal_periods=self.seasonal_periods
        ).fit(optimized=True)
        return self

    def forecast(self, n: int, time_col: str, value_col: str,
                 past_covariates: Optional[pd.DataFrame] = None,
                 future_covariates: Optional[pd.DataFrame] = None) -> pd.Series:
        preds = self.model.forecast(n)
        # Construct future index from last date
        last_date = pd.to_datetime(past_covariates["Date"].max()) if past_covariates is not None and "Date" in past_covariates.columns else None
        if last_date is None:
            raise ValueError("ETSPipeline.forecast requires past_covariates with Date column to build future index.")
        idx = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=n, freq="MS")
        return pd.Series(preds, index=idx)


class TBATSPipeline(BasePipeline):
    def __init__(self, seasonal_periods: Tuple[int, ...] = (12,)):
        self.seasonal_periods = seasonal_periods
        self.model = None

    def fit(self, train_df: pd.DataFrame, time_col: str, value_col: str,
            past_covariates: Optional[pd.DataFrame] = None,
            future_covariates: Optional[pd.DataFrame] = None) -> "TBATSPipeline":
        if TBATS is None:
            raise ImportError(
                f"TBATS is unavailable due to an import error: {_TBATS_IMPORT_ERROR}. "
                "Fix your NumPy/pmdarima/tbats installation or skip 'tbats' in --models."
            )
        estimator = TBATS(seasonal_periods=list(self.seasonal_periods))
        self.model = estimator.fit(train_df[value_col].astype(float).values)
        return self

    def forecast(self, n: int, time_col: str, value_col: str,
                 past_covariates: Optional[pd.DataFrame] = None,
                 future_covariates: Optional[pd.DataFrame] = None) -> pd.Series:
        preds = self.model.forecast(steps=n)
        last_date = pd.to_datetime(past_covariates["Date"].max()) if past_covariates is not None and "Date" in past_covariates.columns else None
        if last_date is None:
            raise ValueError("TBATSPipeline.forecast requires past_covariates with Date column to build future index.")
        idx = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=n, freq="MS")
        return pd.Series(preds, index=idx)


class LSTMPipeline(BasePipeline):
    def __init__(self, input_chunk_length: int = 24, output_chunk_length: int = 6, n_rnn_layers: int = 2,
                 hidden_dim: int = 64, dropout: float = 0.1, random_state: int = 42):
        self.model = RNNModel(
            model="LSTM",
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            hidden_dim=hidden_dim,
            n_rnn_layers=n_rnn_layers,
            dropout=dropout,
            random_state=random_state,
            training_length=input_chunk_length + output_chunk_length,
            n_epochs=50,
            pl_trainer_kwargs={"enable_progress_bar": False},
        )
        self.series: Optional[TimeSeries] = None

    def fit(self, train_df: pd.DataFrame, time_col: str, value_col: str,
            past_covariates: Optional[pd.DataFrame] = None,
            future_covariates: Optional[pd.DataFrame] = None) -> "LSTMPipeline":
        self.series = to_timeseries(ensure_datetime_index(train_df, time_col), time_col, value_col)
        past_cov = None
        fut_cov = None
        if past_covariates is not None and len(past_covariates.columns) > 1:
            past_cov = TimeSeries.from_dataframe(past_covariates, time_col, past_covariates.columns[1:])
        if future_covariates is not None and len(future_covariates.columns) > 1:
            fut_cov = TimeSeries.from_dataframe(future_covariates, time_col, future_covariates.columns[1:])
        self.model.fit(self.series, past_covariates=past_cov, future_covariates=fut_cov, verbose=False)
        return self

    def forecast(self, n: int, time_col: str, value_col: str,
                 past_covariates: Optional[pd.DataFrame] = None,
                 future_covariates: Optional[pd.DataFrame] = None) -> pd.Series:
        past_cov = None
        fut_cov = None
        if past_covariates is not None and len(past_covariates.columns) > 1:
            past_cov = TimeSeries.from_dataframe(past_covariates, time_col, past_covariates.columns[1:])
        if future_covariates is not None and len(future_covariates.columns) > 1:
            fut_cov = TimeSeries.from_dataframe(future_covariates, time_col, future_covariates.columns[1:])
        fcst = self.model.predict(n=n, past_covariates=past_cov, future_covariates=fut_cov)
        return pd.Series(fcst.values().ravel(), index=fcst.time_index)


class DeepARPipeline(BasePipeline):
    def __init__(self, input_chunk_length: int = 24, output_chunk_length: int = 6, hidden_dim: int = 64,
                 n_rnn_layers: int = 2, dropout: float = 0.1, random_state: int = 42):
        # Using Darts RNNModel with likelihood to mimic DeepAR-style probabilistic forecasting
        self.model = RNNModel(
            model="LSTM",
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            hidden_dim=hidden_dim,
            n_rnn_layers=n_rnn_layers,
            dropout=dropout,
            random_state=random_state,
            training_length=input_chunk_length + output_chunk_length,
            n_epochs=50,
            likelihood="gaussian",
            pl_trainer_kwargs={"enable_progress_bar": False},
        )
        self.series: Optional[TimeSeries] = None

    def fit(self, train_df: pd.DataFrame, time_col: str, value_col: str,
            past_covariates: Optional[pd.DataFrame] = None,
            future_covariates: Optional[pd.DataFrame] = None) -> "DeepARPipeline":
        self.series = to_timeseries(ensure_datetime_index(train_df, time_col), time_col, value_col)
        self.model.fit(self.series, verbose=False)
        return self

    def forecast(self, n: int, time_col: str, value_col: str,
                 past_covariates: Optional[pd.DataFrame] = None,
                 future_covariates: Optional[pd.DataFrame] = None) -> pd.Series:
        fcst = self.model.predict(n=n)
        return pd.Series(fcst.values().ravel(), index=fcst.time_index)


class TransformerPipeline(BasePipeline):
    def __init__(self, input_chunk_length: int = 24, output_chunk_length: int = 6, d_model: int = 64,
                 nhead: int = 4, num_encoder_layers: int = 2, num_decoder_layers: int = 2, dropout: float = 0.1,
                 random_state: int = 42):
        self.model = TransformerModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            random_state=random_state,
            n_epochs=60,
            pl_trainer_kwargs={"enable_progress_bar": False},
        )
        self.series: Optional[TimeSeries] = None

    def fit(self, train_df: pd.DataFrame, time_col: str, value_col: str,
            past_covariates: Optional[pd.DataFrame] = None,
            future_covariates: Optional[pd.DataFrame] = None) -> "TransformerPipeline":
        self.series = to_timeseries(ensure_datetime_index(train_df, time_col), time_col, value_col)
        past_cov = None
        fut_cov = None
        if past_covariates is not None and len(past_covariates.columns) > 1:
            past_cov = TimeSeries.from_dataframe(past_covariates, time_col, past_covariates.columns[1:])
        if future_covariates is not None and len(future_covariates.columns) > 1:
            fut_cov = TimeSeries.from_dataframe(future_covariates, time_col, future_covariates.columns[1:])
        self.model.fit(self.series, past_covariates=past_cov, future_covariates=fut_cov, verbose=False)
        return self

    def forecast(self, n: int, time_col: str, value_col: str,
                 past_covariates: Optional[pd.DataFrame] = None,
                 future_covariates: Optional[pd.DataFrame] = None) -> pd.Series:
        past_cov = None
        fut_cov = None
        if past_covariates is not None and len(past_covariates.columns) > 1:
            past_cov = TimeSeries.from_dataframe(past_covariates, time_col, past_covariates.columns[1:])
        if future_covariates is not None and len(future_covariates.columns) > 1:
            fut_cov = TimeSeries.from_dataframe(future_covariates, time_col, future_covariates.columns[1:])
        fcst = self.model.predict(n=n, past_covariates=past_cov, future_covariates=fut_cov)
        return pd.Series(fcst.values().ravel(), index=fcst.time_index)


PIPELINES = {
    "prophet": ProphetPipeline,
    "ets": ETSPipeline,
    "tbats": TBATSPipeline,
    "lstm": LSTMPipeline,
    "deepar": DeepARPipeline,
    "transformer": TransformerPipeline,
}


def train_and_forecast(model_name: str,
                       train_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       time_col: str,
                       value_col: str,
                       past_covariates_train: Optional[pd.DataFrame] = None,
                       future_covariates_test: Optional[pd.DataFrame] = None,
                       n_forecast: Optional[int] = None,
                       **model_kwargs: Any) -> Tuple[pd.Series, Dict[str, float]]:
    if model_name not in PIPELINES:
        raise ValueError(f"Unknown model: {model_name}")
    n = n_forecast if n_forecast is not None else len(test_df)
    pipeline = PIPELINES[model_name](**model_kwargs)
    pipeline.fit(train_df, time_col, value_col, past_covariates=past_covariates_train,
                 future_covariates=future_covariates_test)
    preds = pipeline.forecast(n, time_col, value_col, past_covariates=past_covariates_train,
                              future_covariates=future_covariates_test)
    # Align to test index if possible
    if len(preds) != n and len(test_df) > 0:
        preds = preds.reindex(pd.to_datetime(test_df[time_col].values))
    metrics = pipeline.evaluate(test_df, preds, value_col)
    return preds, metrics


def save_series(series: pd.Series, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = series.rename("prediction").to_frame()
    df.to_csv(path, index_label="Date")


