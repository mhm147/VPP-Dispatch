from data.aeso_loader import load_raw, trainValTestSplit
from forecasting.baselines import historicalAverageForecast

df = load_raw("data/raw/Hourly_Metered_Volumes_and_Pool_Price_and_AIL_2020-Jul2025.csv")

# Split the data temporally
train, val, test = trainValTestSplit(df)

# Use pool price series from train set as input
trainPrice = train["ACTUAL_POOL_PRICE"]

# Forecast for the first 24 hours of the test set
forecastIndex = test.index[:24]
forecast = historicalAverageForecast(trainPrice, forecastIndex)

print(forecast)