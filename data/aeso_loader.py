import pandas as pd

COLS = [
    "Date_Begin_Local", "ACTUAL_POOL_PRICE", "ACTUAL_AIL",
    "HOUR_AHEAD_POOL_PRICE_FORECAST", "EXPORT_BC", "EXPORT_MT",
    "EXPORT_SK", "IMPORT_BC", "IMPORT_MT", "IMPORT_SK"
]

def load_raw(path:str) -> pd.DataFrame:
    #Loading only the columns we ccare about, skipping the 200+ generator columns

    df = pd.read_csv(path, usecols=COLS)
    
    #Converting the date column from string to date for filtering purposes

    df["Date_Begin_Local"] = pd.to_datetime(df["Date_Begin_Local"])
    df = df.set_index("Date_Begin_Local")
    
    #Converting ends

    #Finding duplicate timestamps
    #duplicated() returns true for every duplicate row
    #keep = "first" means we only keep the first occurance of duplicates

    df = df[~df.index.duplicated(keep="first")]

    #Extracting time features (Hour, month, day of week, is weekend)

    df["hour"] = df.index.hour
    df["month"] = df.index.month
    df["dayOfWeek"] = df.index.dayofweek #Monday=0 , Sunday = 6
    df["isWeekend"] = (df.index.dayofweek>=5).astype(int) #True if Saturday/Sunday

    print(f"Shape after cleaning: {df.shape}")
    print(f"Loaded {len(df):,} hours | {df.index.min().date()} → {df.index.max().date()}")
    return df