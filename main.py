from data.aeso_loader import load_raw

df = load_raw("data/raw/Hourly_Metered_Volumes_and_Pool_Price_and_AIL_2020-Jul2025.csv")
print(df.head())