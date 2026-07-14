import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]

FEATURE_BUCKETS = {
    "danceability":     {
        "Low (0–0.4)":    (0.0,  0.4),
        "Medium (0.4–0.7)":(0.4,  0.7),
        "High (0.7–1.0)": (0.7,  1.0),
    },
    "energy":           {
        "Low (0–0.4)":    (0.0,  0.4),
        "Medium (0.4–0.7)":(0.4,  0.7),
        "High (0.7–1.0)": (0.7,  1.0),
    },
    "valence":          {
        "Sad (0–0.35)":   (0.0,  0.35),
        "Neutral (0.35–0.65)":(0.35, 0.65),
        "Happy (0.65–1.0)":(0.65, 1.0),
    },
    "acousticness":     {
        "Electric (0–0.4)": (0.0, 0.4),
        "Mixed (0.4–0.7)":  (0.4, 0.7),
        "Acoustic (0.7–1.0)":(0.7, 1.0),
    },
    "instrumentalness": {
        "Vocal (0–0.5)":        (0.0, 0.5),
        "Instrumental (0.5–1.0)":(0.5, 1.0),
    },
    "liveness":         {
        "Studio (0–0.3)":   (0.0, 0.3),
        "Live Feel (0.3–1.0)":(0.3, 1.0),
    },
    "speechiness":      {
        "Melodic (0–0.1)":   (0.0,  0.1),
        "Mixed (0.1–0.4)":   (0.1,  0.4),
        "Rap/Spoken (0.4–1.0)":(0.4, 1.0),
    },
    "tempo":            {
        "Slow (<90 BPM)":   (0,    90),
        "Mid (90–130 BPM)": (90,   130),
        "Fast (>130 BPM)":  (130,  9999),
    },
}

FEATURE_LABELS = {
    "danceability":     "💃 Danceability",
    "energy":           "⚡ Energy",
    "valence":          "😊 Mood (Valence)",
    "acousticness":     "🎸 Acousticness",
    "instrumentalness": "🎹 Instrumentalness",
    "liveness":         "🎤 Liveness",
    "speechiness":      "🗣 Speechiness",
    "tempo":            "🥁 Tempo",
}


class MusicRecommender:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy().reset_index(drop=True)
        self._clean()
        self._build_matrix()

    def _clean(self):
        BAD = {"nan", "none", "null", "", "unknown", "n/a", "na"}

        for col in ["track_name", "artist_name"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip()
                self.df = self.df[~self.df[col].str.lower().isin(BAD)]

        for col in ["genre", "language"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip()
                self.df.loc[self.df[col].str.lower().isin(BAD), col] = ""

        if "region" not in self.df.columns:
            self.df["region"] = "Global"

        self.df = self.df.reset_index(drop=True)

    def _build_matrix(self):
        available = [c for c in FEATURE_COLS if c in self.df.columns]
        feature_df = self.df[available].apply(pd.to_numeric, errors="coerce").fillna(0)
        scaler = MinMaxScaler()
        self.scaled = scaler.fit_transform(feature_df).astype(np.float32)
        self.feature_names = available

    def search(self, query: str, top_n: int = 40) -> pd.DataFrame:
        q = query.lower().strip()
        mask = (
            self.df["track_name"].str.lower().str.contains(q, na=False) |
            self.df["artist_name"].str.lower().str.contains(q, na=False)
        )
        results = self.df[mask].drop_duplicates(subset=["track_name", "artist_name"])
        cols = ["track_name", "artist_name", "genre", "language", "region"] + \
               [f for f in self.feature_names if f in results.columns]
        cols = [c for c in cols if c in results.columns]
        return results[cols].head(top_n)

    def recommend(
        self,
        track_name: str,
        artist_name: str = "",
        n: int = 10,
        language_filter: list = None,
        genre_filter: list = None,
        region_filter: str = "All",
        feature_filters: dict = None,
    ) -> pd.DataFrame:
        name_mask = self.df["track_name"].str.lower().str.strip() == track_name.lower().strip()
        if artist_name:
            name_mask &= self.df["artist_name"].str.lower().str.strip() == artist_name.lower().strip()

        idxs = self.df[name_mask].index.tolist()
        if not idxs:
            return pd.DataFrame()

        seed_idx = idxs[0]
        seed_vec = self.scaled[seed_idx]

        norms = np.linalg.norm(self.scaled, axis=1)
        norms[norms == 0] = 1e-9
        seed_norm = seed_vec / (np.linalg.norm(seed_vec) + 1e-9)
        mat_normed = self.scaled / norms[:, None]
        sims = mat_normed @ seed_norm

        sims[seed_idx] = -1

        fmask = pd.Series(True, index=self.df.index)

        if region_filter and region_filter != "All":
            fmask &= self.df["region"] == region_filter

        if language_filter:
            fmask &= self.df["language"].isin(language_filter)

        if genre_filter:
            pattern = "|".join([g.lower() for g in genre_filter])
            fmask &= self.df["genre"].str.lower().str.contains(pattern, na=False)

        if feature_filters:
            for feat, (fmin, fmax) in feature_filters.items():
                if feat in self.df.columns:
                    col = pd.to_numeric(self.df[feat], errors="coerce")
                    fmask &= (col >= fmin) & (col <= fmax)

        filtered_indices = self.df.index[fmask].tolist()
        if not filtered_indices:
            return pd.DataFrame()

        filtered_sims = sims[filtered_indices]
        top_order = np.argsort(filtered_sims)[::-1][:n]
        top_indices = [filtered_indices[i] for i in top_order]

        result = self.df.iloc[top_indices].copy()
        result["similarity_score"] = np.round(sims[top_indices], 4)

        cols = ["track_name", "artist_name", "genre", "language", "region",
                "similarity_score"] + self.feature_names
        cols = [c for c in cols if c in result.columns]
        return result[cols].reset_index(drop=True)

    def get_seed_features(self, track_name: str, artist_name: str = "") -> dict:
        mask = self.df["track_name"].str.lower().str.strip() == track_name.lower().strip()
        if artist_name:
            mask &= self.df["artist_name"].str.lower().str.strip() == artist_name.lower().strip()
        rows = self.df[mask]
        if rows.empty:
            return {}
        row = rows.iloc[0]
        return {f: round(float(row[f]), 4) for f in self.feature_names
                if f in row and pd.notna(row[f])}

    def feature_ranges(self) -> dict:
        ranges = {}
        for f in self.feature_names:
            col = pd.to_numeric(self.df[f], errors="coerce").dropna()
            if len(col):
                ranges[f] = (float(col.min()), float(col.max()))
        return ranges

    @property
    def languages(self):
        BAD = {"nan","none","null","","unknown","n/a","na"}
        return sorted({
            str(l).strip() for l in self.df["language"].dropna().unique()
            if str(l).strip().lower() not in BAD
        })

    @property
    def genres(self):
        BAD = {"nan","none","null","","unknown","n/a","na"}
        return sorted({
            str(g).strip() for g in self.df["genre"].dropna().unique()
            if str(g).strip().lower() not in BAD
        })

    def stats(self):
        return {
            "total_songs": len(self.df),
            "languages":   len(self.languages),
            "genres":      len(self.genres),
            "artists":     int(self.df["artist_name"].nunique()),
        }
