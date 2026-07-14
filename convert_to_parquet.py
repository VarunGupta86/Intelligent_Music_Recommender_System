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

# These 7 features are bounded 0-1, so float16 (~3 decimal digits of
# precision) loses nothing that matters for similarity scoring or display,
# while halving their storage vs float32. loudness (dB) and tempo (BPM)
# have wider ranges, so they stay float32 for safety.
FLOAT16_SAFE_COLS = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence",
]

if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"Could not find {CSV_PATH}. Run prepare_data.py first.")

    dtype_map = {**FEATURE_DTYPES, **{c: "category" for c in CATEGORY_COLS}}
    print(f"Reading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH, dtype=dtype_map)

    for col in FLOAT16_SAFE_COLS:
        if col in df.columns:
            df[col] = df[col].astype("float16")

    print(f"Writing {PARQUET_PATH} (zstd compression) ...")
    df.to_parquet(PARQUET_PATH, index=False, compression="zstd", compression_level=19)

    csv_mb = os.path.getsize(CSV_PATH) / 1e6
    pq_mb  = os.path.getsize(PARQUET_PATH) / 1e6
    print(f"Done. {csv_mb:.1f} MB CSV -> {pq_mb:.1f} MB Parquet "
          f"({len(df):,} rows).")