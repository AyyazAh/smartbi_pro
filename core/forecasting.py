"""Forecasting: Prophet with linear-regression + rolling-mean fallbacks."""
import pandas as pd
import numpy as np
from utils.logger import get_logger

log = get_logger("forecasting")


def _month_freq() -> str:
    """Return 'ME' for pandas >= 2.2, else 'M'."""
    major, minor = map(int, pd.__version__.split(".")[:2])
    return "ME" if (major, minor) >= (2, 2) else "M"


def _month_start_freq() -> str:
    """Return 'MS' — month start (unchanged in pandas 2.2)."""
    return "MS"


def _fit_linear(ts: pd.Series, periods: int) -> pd.DataFrame:
    """Simple linear trend forecast with ±10% band."""
    x = np.arange(len(ts))
    y = ts.values.astype(float)
    coef = np.polyfit(x, y, 1)

    future_x = np.arange(len(ts), len(ts) + periods)
    pred = np.polyval(coef, future_x)

    last_date = ts.index[-1]
    future_idx = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=periods,
        freq=_month_freq(),
    )

    # simple band: 10% of prediction
    band = np.abs(pred) * 0.10
    return pd.DataFrame({
        "ds": future_idx,
        "yhat": pred,
        "yhat_lower": pred - band,
        "yhat_upper": pred + band,
    })


def _fit_rolling_mean(ts: pd.Series, periods: int) -> pd.DataFrame:
    """Last resort: extend with rolling mean."""
    mean_val = float(ts.tail(3).mean())
    last_date = ts.index[-1]
    future_idx = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=periods,
        freq=_month_freq(),
    )
    band = abs(mean_val) * 0.15
    return pd.DataFrame({
        "ds": future_idx,
        "yhat": [mean_val] * periods,
        "yhat_lower": [mean_val - band] * periods,
        "yhat_upper": [mean_val + band] * periods,
    })


def _fit_prophet(ts: pd.Series, periods: int, interval_width: float) -> pd.DataFrame:
    """Prophet forecast."""
    from prophet import Prophet

    train = ts.reset_index()
    train.columns = ["ds", "y"]
    train["ds"] = pd.to_datetime(train["ds"])
    train["y"] = train["y"].astype(float)

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=interval_width,
    )
    m.fit(train)

    future = m.make_future_dataframe(periods=periods, freq=_month_start_freq())
    fc = m.predict(future)

    # Return only the future periods
    out = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods).reset_index(drop=True)
    return out


def forecast_series(ts: pd.Series, periods: int = 6,
                    config: dict = None) -> dict:
    """
    Forecast a monthly time series.
    Returns:
      {
        "method": 'prophet' | 'linear' | 'rolling_mean',
        "forecast": DataFrame[ds, yhat, yhat_lower, yhat_upper],
        "history": original ts,
        "n_history_points": int,
        "periods": int,
      }
    """
    config = config or {}
    fcfg = config.get("forecasting", {})
    interval_width = fcfg.get("confidence_interval", 0.90)
    min_prophet = fcfg.get("min_points_for_prophet", 8)

    if ts is None or len(ts) == 0:
        raise ValueError("Empty time series")

    # Ensure monthly frequency, fill small gaps
    ts = ts.asfreq(_month_freq()).interpolate(method="linear").ffill().bfill()

    n = len(ts)
    log.info(f"Forecasting {n} monthly points, horizon={periods}")

    forecast_df = None
    method = None

    # Try Prophet
    if n >= min_prophet:
        try:
            forecast_df = _fit_prophet(ts, periods, interval_width)
            method = "prophet"
            log.info(f"Prophet forecast successful ({periods} periods)")
        except Exception as e:
            log.warning(f"Prophet failed: {e}")

    # Fallback: linear
    if forecast_df is None:
        try:
            forecast_df = _fit_linear(ts, periods)
            method = "linear"
            log.info(f"Linear fallback used")
        except Exception as e:
            log.warning(f"Linear fallback failed: {e}")

    # Last resort: rolling mean
    if forecast_df is None:
        forecast_df = _fit_rolling_mean(ts, periods)
        method = "rolling_mean"
        log.info(f"Rolling-mean fallback used")

    # Ensure non-negative (revenue can't be negative)
    forecast_df["yhat"] = forecast_df["yhat"].clip(lower=0)
    forecast_df["yhat_lower"] = forecast_df["yhat_lower"].clip(lower=0)
    forecast_df["yhat_upper"] = forecast_df["yhat_upper"].clip(lower=0)

    return {
        "method": method,
        "forecast": forecast_df,
        "history": ts,
        "n_history_points": n,
        "periods": periods,
    }


def print_forecast_report(result: dict) -> None:
    """Pretty-print forecast summary."""
    print(f"\n{'='*70}")
    print(f"🔮 FORECAST REPORT")
    print(f"{'='*70}")
    print(f"Method used:      {result['method'].upper()}")
    print(f"History points:   {result['n_history_points']}")
    print(f"Forecast horizon: {result['periods']} months")
    print(f"{'-'*70}")

    fc = result["forecast"]
    print(f"\n{'Month':<14} {'Forecast':>14} {'Lower':>14} {'Upper':>14}")
    print(f"{'-'*70}")
    for _, row in fc.iterrows():
        ds = pd.to_datetime(row["ds"]).strftime("%Y-%m")
        print(f"{ds:<14} {row['yhat']:>14,.2f} {row['yhat_lower']:>14,.2f} {row['yhat_upper']:>14,.2f}")

    # Summary
    total = fc["yhat"].sum()
    avg = fc["yhat"].mean()
    first = fc["yhat"].iloc[0]
    last = fc["yhat"].iloc[-1]
    trend = ((last - first) / first * 100) if first else 0

    print(f"\n📊 SUMMARY")
    print(f"   Total projected:  {total:>14,.2f}")
    print(f"   Avg per month:    {avg:>14,.2f}")
    print(f"   Trend (first→last): {trend:>13.2f}%")

    print(f"{'='*70}\n")


__all__ = ["forecast_series", "print_forecast_report"]