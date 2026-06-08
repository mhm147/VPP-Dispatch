import pandas as pd

def persistenceForecast(series: pd.Series, horizon: int = 24) -> pd.Series:
    """
    Predicts the next `horizon` hours by repeating the last known value.
    The dumbest possible forecast — our lower baseline.
    """
    # Grab the last known value in the series
    lastValue = series.iloc[-1]
    
    # Create a datetime index for the next `horizon` hours
    # We start from the last timestamp, generate horizon+1 points, then skip the first (which is the last known time)
    futureIndex = pd.date_range(start=series.index[-1], periods=horizon + 1, freq="h")[1:]
    
    # Return a Series filled with the same value repeated across the future index
    return pd.Series(lastValue, index=futureIndex)

def seasonalNaiveForecast(series: pd.Series, horizon: int = 24) -> pd.Series:
    """
    Predicts the next `horizon` hours using the same hours from 7 days ago.
    Captures weekly seasonality — smarter than plain persistence.
    """
    # 7 days ago = 168 hours ago
    seasonalLag = 168
    
    # Grab the last 168+horizon values so we have enough history to look back
    history = series.iloc[-(seasonalLag + horizon):]
    
    # For each future hour, find the value from exactly 7 days prior
    futureIndex = pd.date_range(start=series.index[-1], periods=horizon + 1, freq="h")[1:]
    forecastValues = [series.iloc[-(seasonalLag - i)] for i in range(horizon)]
    
    return pd.Series(forecastValues, index=futureIndex)