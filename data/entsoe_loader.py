# This module pulls German electricity market data from ENTSO-E (the European grid operator).
# ENTSO-E publishes transparency data for free — prices, load, generation, etc.
# We use the `entsoe-py` library so we don't have to build raw HTTP requests ourselves.

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv  # reads key=value pairs from a .env file on disk
from entsoe import EntsoePandasClient  # wrapper around the ENTSO-E API; returns pandas objects

# DE_LU = Germany + Luxembourg bidding zone. Since 2018 they share one day-ahead price.
# EPEX (the exchange) clears the day-ahead auction for this zone every day at ~12:00 Berlin time.
DE_LU = "DE_LU"

# All German market timestamps must be in Europe/Berlin (CET in winter, CEST in summer).
# ENTSO-E requires timezone-aware timestamps — a naive date string will error.
BERLIN_TZ = "Europe/Berlin"


def getApiKey(envPath: str = ".env") -> str:
    """
    Load the ENTSO-E API token from a .env file or the ENTSOE_API_KEY environment variable.
    Never hardcode the token in source code — it is a secret.
    """
    # load_dotenv reads .env into environment variables so os.getenv can find them.
    load_dotenv(envPath)

    apiKey = os.getenv("ENTSOE_API_KEY")

    # Fail fast with a helpful message if the token is missing or still the placeholder.
    if not apiKey or apiKey == "your-token-here":
        raise ValueError(
            "Missing ENTSOE_API_KEY. Copy .env.example to .env and paste your token.\n"
            "ENTSO-E hides the token until you email transparency@entsoe.eu with subject "
            "'RESTful API access' and your registered email. Approval usually takes ~3 working days. "
            "Then: log in -> My Account Settings -> Generate token."
        )
    return apiKey


def _toBerlinTimestamp(dateString: str) -> pd.Timestamp:
    """
    Convert a date string like '2024-01-01' into a timezone-aware Berlin timestamp.
    The leading underscore means "internal helper" — only used inside this file.
    """
    return pd.Timestamp(dateString, tz=BERLIN_TZ)


def fetchDayAheadPrices(
    apiKey: str,
    startDate: str,
    endDate: str,
    countryCode: str = DE_LU,
) -> pd.Series:
    """
    Pull day-ahead electricity prices (EUR/MWh) from the ENTSO-E Transparency API.
    These are the prices from the EPEX day-ahead auction — the main product traders use.
    """
    # Create a client object — think of it as our authenticated connection to ENTSO-E.
    client = EntsoePandasClient(api_key=apiKey)

    # Convert string dates to proper timestamps ENTSO-E understands.
    start = _toBerlinTimestamp(startDate)
    end = _toBerlinTimestamp(endDate)

    # query_day_ahead_prices returns a pandas Series: index = datetime, values = EUR/MWh.
    # entsoe-py auto-splits requests longer than one year (API limit).
    prices = client.query_day_ahead_prices(countryCode, start=start, end=end)

    # .name labels the Series so we know what it is when we merge with load later.
    prices.name = "dayAheadPrice"
    return prices


def fetchActualLoad(
    apiKey: str,
    startDate: str,
    endDate: str,
    countryCode: str = DE_LU,
) -> pd.Series:
    """
    Pull actual total load (MW) for the bidding zone.
    Load is a key fundamental — high demand often pushes prices up on the merit order.
    """
    client = EntsoePandasClient(api_key=apiKey)
    start = _toBerlinTimestamp(startDate)
    end = _toBerlinTimestamp(endDate)

    # query_load returns total electricity consumption in MW for each hour.
    load = client.query_load(countryCode, start=start, end=end)
    load.name = "actualLoad"
    return load


def _hourlyMean(series: pd.Series) -> pd.Series:
    """
    Resample to hourly means if ENTSO-E returns sub-hourly points.
    We want one price per hour to match our forecasting setup later.
    """
    if series.empty:
        return series

    # pd.infer_freq guesses the time step (e.g. 'H' = hourly). If already hourly, skip resampling.
    inferredFreq = pd.infer_freq(series.index)
    if inferredFreq is not None and inferredFreq.startswith("H"):
        return series

    # resample("h") groups data into hourly buckets; .mean() averages within each bucket.
    return series.resample("h").mean()


def buildMarketFrame(prices: pd.Series, load: pd.Series) -> pd.DataFrame:
    """
    Merge price and load on the datetime index into one tidy DataFrame.
    One row per hour, columns = price + load. This is our working dataset.
    """
    # Make sure both series are hourly before merging.
    pricesHourly = _hourlyMean(prices)
    loadHourly = _hourlyMean(load)

    # pd.concat stacks Series side-by-side (axis=1) aligned on the datetime index.
    df = pd.concat([pricesHourly, loadHourly], axis=1)

    # sort_index ensures rows are in chronological order.
    df = df.sort_index()

    # Drop duplicate timestamps (can happen around DST clock changes).
    # keep="first" keeps the earlier row when two rows share the same timestamp.
    df = df[~df.index.duplicated(keep="first")]
    return df


def addTimeFeatures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar features traders use to spot patterns (hour-of-day, seasonality).
    Prices follow daily and weekly cycles — same calendar features every forecaster uses.
    """
    # .copy() so we don't accidentally modify the original DataFrame passed in.
    df = df.copy()

    # These are extracted from the datetime index, not fetched from the API.
    df["hour"] = df.index.hour          # 0-23: captures the 5pm demand peak
    df["month"] = df.index.month        # 1-12: captures winter heating / summer AC
    df["dayOfWeek"] = df.index.dayofweek  # Monday=0, Sunday=6
    df["isWeekend"] = (df.index.dayofweek >= 5).astype(int)  # 1 if Sat/Sun, else 0
    return df


def auditDataQuality(df: pd.DataFrame) -> None:
    """
    Print a quick data-quality summary before we trust the series for forecasting.
    Analysts always sanity-check: negatives, gaps, duplicates.
    """
    priceCol = "dayAheadPrice"

    # Count hours where price went negative — normal in Germany when renewables oversupply the grid.
    negativeHours = (df[priceCol] < 0).sum()
    zeroHours = (df[priceCol] == 0).sum()
    missingPrice = df[priceCol].isna().sum()

    # Build a perfect hourly index from first to last timestamp.
    # Any hour missing from our data = a gap we need to know about.
    fullIndex = pd.date_range(df.index.min(), df.index.max(), freq="h", tz=BERLIN_TZ)
    missingTimestamps = len(fullIndex.difference(df.index))
    duplicateTimestamps = df.index.duplicated().sum()

    print(f"Shape: {df.shape}")
    print(f"Range: {df.index.min()} -> {df.index.max()}")
    print(f"Negative prices: {negativeHours:,} hours (common in Germany when wind/solar floods the grid)")
    print(f"Zero prices: {zeroHours:,} hours")
    print(f"Missing price values: {missingPrice:,}")
    print(f"Missing hourly timestamps: {missingTimestamps:,}")
    print(f"Duplicate timestamps: {duplicateTimestamps:,}")

    # Mean vs median gap is a quick check for right-skew / fat tails in the price distribution.
    if priceCol in df.columns and df[priceCol].notna().any():
        print(
            f"Price stats (EUR/MWh): mean={df[priceCol].mean():.2f}, "
            f"median={df[priceCol].median():.2f}, max={df[priceCol].max():.2f}"
        )


def loadMarketData(
    apiKey: str,
    startDate: str,
    endDate: str,
    cachePath: str | None = None,
    countryCode: str = DE_LU,
) -> pd.DataFrame:
    """
    Load DE-LU day-ahead prices and actual load.
    If cachePath is set and the file exists, read from disk instead of calling the API again.
    """
    # --- Cache path: skip the API if we already saved this data to CSV ---
    if cachePath is not None and Path(cachePath).exists():
        # index_col=0 means the first column (datetime) becomes the row index.
        df = pd.read_csv(cachePath, index_col=0, parse_dates=True)

        # CSVs lose timezone info — re-attach Berlin timezone so timestamps stay correct.
        if df.index.tz is None:
            df.index = df.index.tz_localize(BERLIN_TZ)

        print(f"Loaded cached market data from {cachePath}")
        auditDataQuality(df)
        return addTimeFeatures(df)

    # --- Fresh pull from ENTSO-E ---
    prices = fetchDayAheadPrices(apiKey, startDate, endDate, countryCode)
    load = fetchActualLoad(apiKey, startDate, endDate, countryCode)
    df = buildMarketFrame(prices, load)
    df = addTimeFeatures(df)

    # Save to CSV so the next run is instant (API is slow and rate-limited).
    if cachePath is not None:
        Path(cachePath).parent.mkdir(parents=True, exist_ok=True)  # create data/raw/ if missing
        df.to_csv(cachePath)
        print(f"Saved cache to {cachePath}")

    auditDataQuality(df)
    return df


def trainValTestSplit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Temporal split — never shuffle time series.
    Train: 2020-2022, Validation: 2023, Test: 2024 onward.
    """
    # .loc["start":"end"] slices by date label on the datetime index.
    # Past data trains the model; future data tests it — mimics real trading (you can't see tomorrow).
    train = df.loc["2020-01-01":"2022-12-31"]
    val = df.loc["2023-01-01":"2023-12-31"]
    test = df.loc["2024-01-01":]

    print(f"Train: {len(train):,} hours | {train.index.min().date()} -> {train.index.max().date()}")
    print(f"Val:   {len(val):,} hours | {val.index.min().date()} -> {val.index.max().date()}")
    print(f"Test:  {len(test):,} hours | {test.index.min().date()} -> {test.index.max().date()}")
    return train, val, test
