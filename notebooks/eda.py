import sys
# Add project root to path so we can import from data/
sys.path.insert(0, ".")

import matplotlib.pyplot as plt
from data.aeso_loader import load_raw

df = load_raw("data/raw/Hourly_Metered_Volumes_and_Pool_Price_and_AIL_2020-Jul2025.csv")

# Calculate summary statistics
meanPrice   = df["ACTUAL_POOL_PRICE"].mean()
medianPrice = df["ACTUAL_POOL_PRICE"].median()
maxPrice    = df["ACTUAL_POOL_PRICE"].max()
above200    = (df["ACTUAL_POOL_PRICE"] > 200).sum()
above500    = (df["ACTUAL_POOL_PRICE"] > 500).sum()

# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 5))

# Plot histogram — 100 bins across the full price range
ax.hist(df["ACTUAL_POOL_PRICE"], bins=100, color="steelblue", edgecolor="white", linewidth=0.3)

# Vertical lines showing mean and median — mean > median confirms right skew
ax.axvline(meanPrice,   color="red",    linestyle="--", linewidth=1.5, label=f"Mean: ${meanPrice:.2f}")
ax.axvline(medianPrice, color="orange", linestyle="--", linewidth=1.5, label=f"Median: ${medianPrice:.2f}")

# Text box with spike stats positioned in top right (0–1 are axes coordinates, not price)
statsText = f"Max: ${maxPrice:.0f}/MWh\nHours > $200: {above200}\nHours > $500: {above500}"
ax.text(0.72, 0.95, statsText,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

ax.legend()
ax.set_title("AESO Pool Price Distribution (2020–2025)")
ax.set_xlabel("Pool Price ($/MWh)")
ax.set_ylabel("Hours")

plt.tight_layout()
plt.show()

# Calculate average price for each hour of the day (0-23)
hourlyAvgPrice = df.groupby("hour")["ACTUAL_POOL_PRICE"].mean()

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(hourlyAvgPrice.index, hourlyAvgPrice.values, color="steelblue", edgecolor="white", linewidth=0.3)

ax.set_title("Average Pool Price by Hour of Day (2020–2025)")
ax.set_xlabel("Hour of Day (Mountain Time)")
ax.set_ylabel("Avg Price ($/MWh)")
ax.set_xticks(range(24))

plt.tight_layout()
plt.show()

# Average load by hour of day
hourlyAvgLoad = df.groupby("hour")["ACTUAL_AIL"].mean()

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(hourlyAvgLoad.index, hourlyAvgLoad.values, color="darkorange", edgecolor="white", linewidth=0.3)

ax.set_title("Average Alberta Internal Load by Hour of Day (2020–2025)")
ax.set_xlabel("Hour of Day (Mountain Time)")
ax.set_ylabel("Avg Load (MW)")
ax.set_xticks(range(24))

plt.tight_layout()
plt.show()

# Average price by month — reveals seasonal patterns
monthlyAvgPrice = df.groupby("month")["ACTUAL_POOL_PRICE"].mean()

# Month names for x-axis labels
monthNames = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(monthlyAvgPrice.index, monthlyAvgPrice.values, color="steelblue", edgecolor="white", linewidth=0.3)

ax.set_title("Average Pool Price by Month (2020–2025)")
ax.set_xlabel("Month")
ax.set_ylabel("Avg Price ($/MWh)")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(monthNames)

plt.tight_layout()
plt.show()

# Separate weekday and weekend data
weekday = df[df["isWeekend"] == 0]
weekend = df[df["isWeekend"] == 1]

# Average price by hour for each group
weekdayHourly = weekday.groupby("hour")["ACTUAL_POOL_PRICE"].mean()
weekendHourly = weekend.groupby("hour")["ACTUAL_POOL_PRICE"].mean()

fig, ax = plt.subplots(figsize=(10, 5))

# Plot both lines on the same axes
ax.plot(weekdayHourly.index, weekdayHourly.values, color="steelblue", linewidth=2, label="Weekday")
ax.plot(weekendHourly.index, weekendHourly.values, color="darkorange", linewidth=2, label="Weekend")

ax.set_title("Average Pool Price by Hour — Weekday vs Weekend (2020–2025)")
ax.set_xlabel("Hour of Day (Mountain Time)")
ax.set_ylabel("Avg Price ($/MWh)")
ax.set_xticks(range(24))
ax.legend()

plt.tight_layout()
plt.show()