"""
Run this ONCE locally before deploying:

    python convert_to_parquet.py

It converts data/master_songs.csv -> data/master_songs.parquet using the
same dtypes app.py uses at runtime. Commit the .parquet file (it will be
noticeably smaller than the CSV) and the app will skip CSV parsing
entirely on Streamlit Cloud, so cold-start / reboot times drop a lot.

You can keep or delete the original CSV from the repo afterwards --
app.py will use the Parquet file automatically if it exists.
"""
import os
import pandas as pd

CSV_PATH     = "data/master_songs.csv"
PARQUET_PATH = "data/master_songs.parquet"

FEATURE_DTYPES = {
    "danceability": "float32", "energy": "float32", "loudness": "float32",
    "speechiness": "float32", "acousticness": "float32", "instrumentalness": "float32",
    "liveness": "float32", "valence": "float32", "tempo": "float32",
}
CATEGORY_COLS = ["genre", "language", "region"]

# NOTE: we intentionally do NOT downcast any feature columns to float16.
# float16 was causing "Audio Feature Distributions" charts to render blank/
# broken (many plotting libs mishandle float16 during histogram binning,
# and Arrow/Parquet readers can round-trip float16 inconsistently).
# All numeric features stay float32. Size is instead controlled purely via
# Parquet's columnar layout + zstd max compression + category dtype for
# the string columns, which is normally enough to take a 300+ MB CSV to
# well under 100 MB without touching precision.

if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"Could not find {CSV_PATH}. Run prepare_data.py first.")

    dtype_map = {**FEATURE_DTYPES, **{c: "category" for c in CATEGORY_COLS}}
    print(f"Reading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH, dtype=dtype_map)

    print(f"Writing {PARQUET_PATH} (zstd compression) ...")
    df.to_parquet(PARQUET_PATH, index=False, compression="zstd", compression_level=19)

    csv_mb = os.path.getsize(CSV_PATH) / 1e6
    pq_mb  = os.path.getsize(PARQUET_PATH) / 1e6
    print(f"Done. {csv_mb:.1f} MB CSV -> {pq_mb:.1f} MB Parquet "
          f"({len(df):,} rows).")

    LIMIT_MB = 100
    if pq_mb >= LIMIT_MB:
        print(f"WARNING: Parquet file is {pq_mb:.1f} MB, at/over the "
              f"{LIMIT_MB} MB target. Consider dropping unused columns, "
              f"reducing string-column cardinality (category dtype), or "
              f"splitting the dataset before committing it.")