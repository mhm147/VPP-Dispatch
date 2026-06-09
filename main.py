# Entry point for testing during development.
# Run with: python main.py
# This file wires together the loader functions and prints a quick sanity check.

from data.entsoe_loader import getApiKey, loadMarketData, trainValTestSplit

# --- Config: change these dates once the smoke test works ---
# Start with one week to verify the API token and connection.
# Then widen to "2020-01-01" through "2025-06-01" for the full project history.
START_DATE = "2024-01-01"
END_DATE = "2024-01-08"

# After the first successful API pull, data is saved here as CSV.
# Next runs read from disk instead of calling ENTSO-E again.
CACHE_PATH = "data/raw/de_lu_hourly.csv"

# Step 1: read API token from .env (see .env.example for setup).
apiKey = getApiKey()

# Step 2: load prices + load, merge, add time features, print data-quality audit.
df = loadMarketData(
    apiKey,
    startDate=START_DATE,
    endDate=END_DATE,
    cachePath=CACHE_PATH,
)

# Step 3: peek at the first few rows to confirm the data looks right.
print("\nFirst rows:")
print(df.head())

# Step 4: temporal train/val/test split — only when we have multi-year history.
# With just one week of 2024 data, train and val would be empty, so we skip for now.
if df.index.min().year <= 2020 and df.index.max().year >= 2024:
    print("\nTemporal split:")
    train, val, test = trainValTestSplit(df)
else:
    print(
        "\nSkipping train/val/test split for now — widen START_DATE/END_DATE "
        "to 2020-01-01 through 2025-06-01 once the API pull works."
    )