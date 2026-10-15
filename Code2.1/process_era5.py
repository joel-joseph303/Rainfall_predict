import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# FIND FILE1 ... FILE6
# --------------------------------------------------

folders = sorted(
    [p for p in BASE_DIR.glob("file*") if p.is_dir()],
    key=lambda x: x.name
)

print("Found folders:")
for folder in folders:
    print(" ", folder)

# --------------------------------------------------
# PROCESS EACH YEAR
# --------------------------------------------------

daily_data = []

for folder in folders:

    print("\n" + "=" * 60)
    print("Processing:", folder.name)
    print("=" * 60)

    instant_file = folder / "data_stream-oper_stepType-instant.nc"
    accum_file = folder / "data_stream-oper_stepType-accum.nc"

    if not instant_file.exists():
        print("❌ Instant file missing")
        continue

    if not accum_file.exists():
        print("❌ Accumulated file missing")
        continue

    # Open datasets
    instant = xr.open_dataset(instant_file)
    accum = xr.open_dataset(accum_file)

    # --------------------------------------------------
    # COMBINE VARIABLES
    # --------------------------------------------------

    ds = xr.merge([
        instant,
        accum
    ])

    print("Variables:", list(ds.data_vars))

    # --------------------------------------------------
    # SPATIAL AVERAGE
    # --------------------------------------------------
    #
    # This creates ONE Kerala-region value per timestamp.
    # We are using the downloaded Kerala bounding box.
    #

    ds = ds.mean(dim=["latitude", "longitude"])

    # --------------------------------------------------
    # CONVERT TO DATAFRAME
    # --------------------------------------------------

    df = ds.to_dataframe().reset_index()

    # Remove unnecessary coordinate columns
    columns_to_drop = [
        "number",
        "expver"
    ]

    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=col)

    # --------------------------------------------------
    # UNIT CONVERSIONS
    # --------------------------------------------------

    # Temperature: Kelvin → Celsius
    df["temperature"] = df["t2m"] - 273.15
    df["dew_point"] = df["d2m"] - 273.15

    # Pressure: Pa → hPa
    df["pressure"] = df["msl"] / 100

    # Wind speed
    df["wind_speed"] = np.sqrt(
        df["u10"] ** 2 +
        df["v10"] ** 2
    )

 

    # Precipitation: metres → millimetres
    df["rainfall"] = df["tp"] * 1000

    # --------------------------------------------------
    # SELECT FINAL HOURLY FEATURES
    # --------------------------------------------------

    df = df[
        [
            "valid_time",
            "temperature",
            "dew_point",
            "pressure",
            "u10",
            "v10",
            "wind_speed",
            "rainfall"
        ]
    ]

    # --------------------------------------------------
    # DAILY AGGREGATION
    # --------------------------------------------------

    df["date"] = pd.to_datetime(df["valid_time"]).dt.date

    daily = df.groupby("date").agg({

        "temperature": "mean",

        "dew_point": "mean",

        "pressure": "mean",

        "u10": "mean",

        "v10": "mean",

        "wind_speed": "mean",

        "rainfall": "sum"

    }).reset_index()

    daily_data.append(daily)

    print(
        "Daily records:",
        len(daily)
    )

# --------------------------------------------------
# COMBINE ALL YEARS
# --------------------------------------------------

if not daily_data:
    raise RuntimeError("No data was processed.")

final_df = pd.concat(
    daily_data,
    ignore_index=True
)

# Sort chronologically
final_df = final_df.sort_values("date")

# Remove duplicates
final_df = final_df.drop_duplicates(
    subset="date"
)

# --------------------------------------------------
# SAVE
# --------------------------------------------------

output_file = OUTPUT_DIR / "daily_weather.csv"

final_df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 60)
print("PROCESSING COMPLETE")
print("=" * 60)

print("Rows:", len(final_df))
print("Columns:", len(final_df.columns))

print("\nFirst rows:")
print(final_df.head())

print("\nLast rows:")
print(final_df.tail())

print("\nSaved to:")
print(output_file)