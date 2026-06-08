from data.aeso_loader import load_raw
from forecasting.baselines import persistenceForecast

df = load_raw("data/raw/Hourly_Metered_Volumes_and_Pool_Price_and_AIL_2020-Jul2025.csv")

# Use the pool price series as input
priceSeries = df["ACTUAL_POOL_PRICE"]

# Forecast the next 24 hours
forecast = persistenceForecast(priceSeries, horizon=24)

print(forecast)

from forecasting.baselines import persistenceForecast, seasonalNaiveForecast

seasonalForecast = seasonalNaiveForecast(priceSeries, horizon=24)
print(seasonalForecast)