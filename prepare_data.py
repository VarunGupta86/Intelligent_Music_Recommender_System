import pandas as pd
import numpy as np
import os, glob

RAW_DIR  = "data/raw"
OUT_PATH = "data/master_songs.csv"

FEATURE_COLS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]
FINAL_COLS = [
    "track_name", "artist_name", "genre",
    "language", "region",
] + FEATURE_COLS


WESTERN_EXACT = {
    "comedy", "classical", "opera", "soundtrack", "anime", "folk",
    "reggaeton", "electronic", "hip-hop", "movie", "jazz", "reggae",
    "world", "ska", "country", "blues", "dance", "r&b",
    "children's music", "soul", "indie", "pop", "rock",
    "alternative", "rap", "a capella", "unknown", "indian",
}

# Genre substring → language (first match wins)
GENRE_TO_LANG = [
    ({"k-pop", "korean", "kpop"},           "Korean"),
    ({"j-pop", "japanese", "anime"},        "Japanese"),
    ({"latin", "reggaeton", "salsa",
      "bachata", "cumbia", "flamenco"},      "Spanish"),
    ({"afrobeats", "afro", "afropop"},      "Afrobeats"),
    ({"c-pop", "chinese", "mandopop"},      "Chinese"),
    ({"french", "francophone"},             "French"),
    ({"portuguese", "samba", "bossa",
      "mpb", "forro", "axe"},               "Portuguese"),
    ({"arabic", "khaleeji", "rai"},         "Arabic"),
    ({"turkish"},                           "Turkish"),
    ({"bollywood", "filmi", "hindi",
      "hindustani", "ghazal", "qawwali",
      "bhajan"},                            "Hindi"),
    ({"tamil", "kollywood"},                "Tamil"),
    ({"telugu", "tollywood"},               "Telugu"),
    ({"kannada", "sandalwood"},             "Kannada"),
    ({"malayalam", "mollywood"},            "Malayalam"),
    ({"punjabi"},                           "Punjabi"),
    ({"bengali", "bangla", "rabindra"},     "Bengali"),
    ({"indian"},                            "English"),
    ({"pop", "rock", "hip-hop", "jazz", "blues", "country", "electronic",
      "reggae", "soul", "r&b", "classical", "folk", "indie", "rap",
      "dance", "opera", "ska", "reggaeton", "comedy", "soundtrack"},  "English"),
]

INDIAN_GENRE_KW = {
    "bollywood", "filmi", "hindi", "desi", "lata",
    "punjabi", "gujarati", "marathi", "rajasthani", "bhojpuri",
    "odia", "assamese", "garhwali", "haryanvi", "himachali",
    "kashmiri", "maithili", "konkani", "manipuri",
    "tamil", "telugu", "kannada", "malayalam",
    "kollywood", "tollywood", "mollywood", "sandalwood", "carnatic",
    "hindustani", "bhajan", "qawwali", "ghazal", "sufi",
    "hare krishna", "sarangi", "sarod", "sitar", "rabindra",
    "bangla", "bengali", "koligeet",
    "northeast indian", "arunachal",
    "pakistani", "urdu",
}


def classify_region(genre: str, language: str, raw_region: str) -> str:
    r = raw_region.strip().lower()
    if r and r not in ("nan", "none", "unknown", "", "global", "international"):
        if "india" in r or r == "indian":
            return "Indian"

    g = genre.strip().lower()
    l = language.strip().lower()

    if g in WESTERN_EXACT:
        return "Global"

    for kw in INDIAN_GENRE_KW:
        if kw in g:
            return "Indian"

    for kw in ("hindi", "tamil", "telugu", "kannada", "malayalam", "punjabi",
               "bengali", "marathi", "gujarati", "odia", "assamese", "bhojpuri",
               "rajasthani", "haryanvi", "urdu"):
        if kw in l:
            return "Indian"

    return "Global"


def infer_language(genre: str, existing: str) -> str:
    e = existing.strip()
    e_low = e.lower()
    if e and e_low not in ("nan", "none", "unknown", ""):
        return e

    g = genre.strip().lower()

    for keywords, lang in GENRE_TO_LANG:
        if any(kw in g for kw in keywords):
            return lang

    return "English"


DATASET_CONFIGS = [
    {
        "keywords": ["cleaned_tracks_global"],
        "rename":   {"track_name": "track_name", "artist_name": "artist_name",
                     "genre": "genre", "language": "language", "region": "region"},
        "language_hint": None,
    },
    {
        "keywords": ["2m", "2m_songs"],
        "rename":   {"track_name": "track_name", "artist_name": "artist_name",
                     "genre": "genre", "region": "region"},
        "language_hint": None,
    },
    {
        "keywords": ["amitansh", "1m"],
        "rename":   {"track_name": "track_name", "artist_name": "artist_name",
                     "genre": "genre"},
        "language_hint": None,
    },
    {
        "keywords": ["maharshi", "spotify_tracks"],
        "rename":   {"track_name": "track_name", "artists": "artist_name",
                     "track_genre": "genre"},
        "language_hint": None,
    },
]


def _get_config(filename: str):
    fname = filename.lower()
    for cfg in DATASET_CONFIGS:
        if any(kw in fname for kw in cfg["keywords"]):
            return cfg
    return None


def _coerce_features(df: pd.DataFrame) -> pd.DataFrame:
    for c in FEATURE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def standardise(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df.columns = [c.lower().strip() for c in df.columns]
    df = df.rename(columns=cfg["rename"])

    for col in ("track_name", "artist_name", "genre", "language", "region"):
        if col not in df.columns:
            df[col] = ""

    hint = cfg.get("language_hint")
    if hint:
        mask = df["language"].fillna("").str.strip().str.lower().isin(("", "nan", "none"))
        df.loc[mask, "language"] = hint

    df = _coerce_features(df)

    feat_present = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=feat_present, how="all")

    keep = [c for c in FINAL_COLS if c in df.columns]
    return df[keep].copy()


def load_file(path: str) -> pd.DataFrame | None:
    fname = os.path.basename(path)
    try:
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip", low_memory=False)
        if df.empty:
            print(f"  ⚠ {fname}: empty file, skipped")
            return None

        cfg = _get_config(fname)
        if cfg is None:
            print(f"  ⚠ {fname}: unrecognised filename — skipped (rename to match a known keyword)")
            return None

        df = standardise(df, cfg)
        print(f"  ✓ {fname:55s} → {len(df):>8,} rows")
        return df

    except Exception as e:
        print(f"  ✗ {fname}: {e}")
        return None


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    files = sorted(set(glob.glob(f"{RAW_DIR}/**/*.csv", recursive=True)))

    if not files:
        print(f"\n⚠  No CSV files found in {RAW_DIR}/")
        print("   Place your CSVs there and re-run.\n")
        return

    print(f"\n📂 Found {len(files)} file(s) — loading ALL rows (no cap)…\n")
    frames = [load_file(f) for f in files]
    frames = [f for f in frames if f is not None and not f.empty]

    if not frames:
        print("\n✗ No usable data. Check file names match the expected keywords.")
        return

    master = pd.concat(frames, ignore_index=True)
    print(f"\n🔗 Total rows before dedup: {len(master):,}")

    master["track_name"]  = master["track_name"].astype(str).str.strip()
    master["artist_name"] = master["artist_name"].astype(str).str.strip()
    master["_key"] = (master["track_name"].str.lower() + "||" +
                      master["artist_name"].str.lower())
    master = master.drop_duplicates(subset="_key").drop(columns="_key")
    print(f"   After dedup:              {len(master):,}")

    bad = {"nan", "none", "null", "", "unknown"}
    master = master[~master["track_name"].str.lower().isin(bad)]
    master = master[~master["artist_name"].str.lower().isin(bad)]
    print(f"   After removing bad rows:  {len(master):,}")

    print("\n🌍 Classifying region & language for every song…")

    genre_col    = master["genre"].fillna("").astype(str)
    language_col = master["language"].fillna("").astype(str)
    region_col   = master["region"].fillna("").astype(str)

    master["region"] = [
        classify_region(g, l, r)
        for g, l, r in zip(genre_col, language_col, region_col)
    ]
    master["language"] = [
        infer_language(g, l)
        for g, l in zip(genre_col, language_col)
    ]

    master["genre"] = master["genre"].astype(str).str.strip()
    master.loc[master["genre"].str.lower().isin({"nan", "none", "unknown", ""}), "genre"] = ""

    for c in FEATURE_COLS:
        if c in master.columns:
            master[c] = master[c].astype("float32")

    master = master.reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    master[FINAL_COLS].to_csv(OUT_PATH, index=False)

    print(f"\n✅ Saved {OUT_PATH}  —  {len(master):,} songs")
    print(f"\n📊 Region:\n{master['region'].value_counts().to_string()}")
    print(f"\n🌍 Language (top 15):\n{master['language'].value_counts().head(15).to_string()}")
    print(f"\n🎵 Top genres (top 15):\n{master['genre'].value_counts().head(15).to_string()}")
    print(f"\nColumns saved: {FINAL_COLS}")
    print(f"Approx size in memory: {master.memory_usage(deep=True).sum() / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
