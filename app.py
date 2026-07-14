import os
import re
import urllib.parse
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from utils.recommender import FEATURE_BUCKETS, FEATURE_LABELS

st.set_page_config(
    page_title="Music Recommender · ML Engine",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

    [data-testid="stStatusWidget"], #MainMenu, footer { visibility: hidden !important; display: none !important; }

    @keyframes auroraMove { 0%,100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }
    .stApp {
        background: linear-gradient(-45deg, #080810, #0f0a1a, #0a1020, #080810) !important;
        background-size: 400% 400% !important;
        animation: auroraMove 18s ease infinite !important;
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
    }

    .hero-wrap { padding: 2.5rem 0 1.5rem; border-bottom: 1px solid rgba(255,255,255,0.06); margin-bottom: 1.5rem; }
    .hero-title {
        font-family: 'Inter', sans-serif; font-size: 3.2rem; font-weight: 700; line-height: 1.1;
        background: linear-gradient(100deg, #e2c4ff 0%, #a78bfa 35%, #60a5fa 70%, #38bdf8 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    .hero-sub { color: #6b7280; font-size: 0.95rem; font-weight: 300; letter-spacing: 0.04em; }

    .academic-card {
        background: rgba(20,15,35,0.6) !important; backdrop-filter: blur(24px) saturate(150%) !important;
        border: 1px solid rgba(192,132,252,0.15) !important; border-radius: 20px;
        box-shadow: 0 8px 40px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.06);
        padding: 2.5rem 2rem; margin-bottom: 2rem; position: relative; overflow: hidden;
    }
    .academic-card::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, #c084fc, #818cf8, #38bdf8, transparent); opacity: 0.7;
    }
    .academic-title {
        font-family: 'Inter', sans-serif; font-weight: 800; font-size: 1.6rem; text-align: center; margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #c084fc 0%, #38bdf8 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; display: block;
    }
    .academic-subtitle { font-weight: 600; color: #e2e8f0 !important; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px; text-align: center; margin-bottom: 0.2rem; }
    .academic-dept     { color: #6b7280 !important; font-size: 0.85rem; text-align: center; margin-bottom: 2rem; }
    .grid-container    { display: flex; justify-content: space-around; flex-wrap: wrap; gap: 2rem; text-align: left; }
    .grid-col h4       { font-family: 'Inter', sans-serif; font-weight: 700; color: #e2e8f0 !important; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 0.75rem; border-bottom: 1px solid rgba(192,132,252,0.2); padding-bottom: 0.4rem; }
    .grid-col p        { margin: 0.35rem 0; color: #9ca3af !important; font-size: 0.9rem; }
    .prof-title        { font-size: 0.75rem !important; color: #4b5563 !important; margin-top: -0.1rem !important; margin-bottom: 0.8rem !important; }

    @keyframes eq { 0%,100% { transform: scaleY(0.2); } 50% { transform: scaleY(1); } }
    .eq-container { display: flex; align-items: flex-end; justify-content: center; gap: 4px; height: 28px; margin-bottom: 1.5rem; }
    .eq-bar       { width: 5px; background: linear-gradient(180deg, #38bdf8, #c084fc); border-radius: 3px; transform-origin: bottom; }
    .eq-bar:nth-child(1) { height:100%; animation: eq 1.2s ease-in-out infinite; }
    .eq-bar:nth-child(2) { height: 80%; animation: eq 0.8s ease-in-out infinite 0.15s; }
    .eq-bar:nth-child(3) { height:100%; animation: eq 1.4s ease-in-out infinite 0.3s; }
    .eq-bar:nth-child(4) { height: 60%; animation: eq 0.9s ease-in-out infinite 0.05s; }
    .eq-bar:nth-child(5) { height: 90%; animation: eq 1.1s ease-in-out infinite 0.25s; }
    .eq-bar:nth-child(6) { height: 70%; animation: eq 1.3s ease-in-out infinite 0.1s; }
    .eq-bar:nth-child(7) { height: 85%; animation: eq 1.0s ease-in-out infinite 0.35s; }

    p, label { color: #9ca3af !important; }
    div[data-baseweb="input"], div[data-baseweb="select"] {
        background: rgba(15,10,30,0.7) !important; backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 10px !important; transition: all 0.3s ease !important;
    }
    div[data-baseweb="input"]:focus-within, div[data-baseweb="select"]:focus-within {
        border-color: #c084fc !important; box-shadow: 0 0 20px rgba(192,132,252,0.2) !important; background: rgba(20,15,40,0.9) !important;
    }
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div { background-color: transparent !important; border: none !important; box-shadow: none !important; }
    div[data-baseweb="input"] input, div[data-baseweb="select"] * { color: #f9fafb !important; -webkit-text-fill-color: #f9fafb !important; font-weight: 500 !important; }
    div[data-baseweb="input"] input::placeholder { color: #4b5563 !important; -webkit-text-fill-color: #4b5563 !important; }

    .stButton { display: flex !important; justify-content: center !important; }
    button[kind="primary"] {
        background: linear-gradient(135deg, #7c3aed 0%, #2563eb 100%) !important;
        border: none !important; border-radius: 8px !important; padding: 0.75rem 2.5rem !important;
        box-shadow: 0 6px 20px rgba(124,58,237,0.3) !important; transition: all 0.3s !important;
    }
    button[kind="primary"], button[kind="primary"] *, button[kind="primary"] p,
    button[kind="primary"] span, button[kind="primary"] div {
        color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;
        font-family: 'Inter', sans-serif !important; font-weight: 800 !important; font-size: 0.95rem !important;
    }
    button[kind="primary"]:hover { box-shadow: 0 10px 30px rgba(124,58,237,0.5) !important; transform: translateY(-2px) scale(1.01) !important; filter: brightness(1.1); }

    button[kind="secondary"] {
        background: rgba(255,255,255,0.06) !important; border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 8px !important; padding: 0.75rem 2.5rem !important;
        backdrop-filter: blur(8px) !important; transition: all 0.3s ease !important;
    }
    button[kind="secondary"] *, button[kind="secondary"] p {
        color: #e2e8f0 !important; -webkit-text-fill-color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important; font-weight: 500 !important; font-size: 0.92rem !important;
    }
    button[kind="secondary"]:hover {
        background: rgba(192,132,252,0.12) !important;
        border-color: rgba(192,132,252,0.4) !important;
        transform: translateY(-2px) !important;
    }

    [data-testid="stExpander"] {
        background: rgba(15,12,30,0.6) !important; backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255,255,255,0.06) !important; border-radius: 14px !important;
        box-shadow: 0 6px 24px rgba(0,0,0,0.35) !important; margin-bottom: 1rem !important; transition: all 0.3s !important;
    }
    [data-testid="stExpander"]:hover { transform: translateY(-3px) !important; border-color: rgba(192,132,252,0.35) !important; box-shadow: 0 12px 30px rgba(192,132,252,0.12) !important; }
    [data-testid="stExpander"] details summary { background-color: transparent !important; padding: 0.4rem 0.2rem !important; }
    [data-testid="stExpander"] details summary p { color: #f1f5f9 !important; -webkit-text-fill-color: #f1f5f9 !important; font-family: 'Inter', sans-serif !important; font-weight: 700 !important; font-size: 1rem !important; }

    [data-testid="stProgress"] > div { background-color: rgba(255,255,255,0.04) !important; border-radius: 50px !important; }
    [data-testid="stProgress"] > div > div > div { background: linear-gradient(90deg, #7c3aed, #38bdf8) !important; border-radius: 50px !important; box-shadow: 0 0 8px rgba(124,58,237,0.4) !important; }
    [data-testid="stProgress"] > div > div > p { color: #6b7280 !important; font-size: 0.8rem !important; }

    button[data-baseweb="tab"] { color: #6b7280 !important; -webkit-text-fill-color: #6b7280 !important; font-family: 'Inter', sans-serif !important; font-weight: 500 !important; font-size: 0.9rem !important; }
    button[data-baseweb="tab"][aria-selected="true"] { color: #c084fc !important; -webkit-text-fill-color: #c084fc !important; border-bottom-color: #c084fc !important; font-weight: 600 !important; }
    [data-testid="stTabsContent"] { background: transparent !important; }

    [data-testid="stSidebar"] { background: rgba(8,6,20,0.85) !important; border-right: 1px solid rgba(255,255,255,0.05) !important; }

    .stat-card   { background: rgba(255,255,255,0.03); border: 1px solid rgba(192,132,252,0.15); border-radius: 14px; padding: 1rem 1.1rem; text-align: center; margin-bottom: 0.5rem; }
    .stat-number { font-family: 'Inter', sans-serif; font-size: 1.7rem; font-weight: 700; color: #c084fc; letter-spacing: -0.5px; }
    .stat-label  { font-family: 'Plus Jakarta Sans', sans-serif; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.15em; color: #6b7280; margin-top: 0.25rem; font-weight: 500; }

    .song-row   { background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 0.75rem 1rem; margin-bottom: 0.4rem; display: flex; align-items: center; gap: 0.75rem; transition: all 0.2s; }
    .song-row:hover { border-color: rgba(192,132,252,0.35); background: rgba(192,132,252,0.04); }
    .song-rank   { color: #374151; font-size: 0.78rem; font-weight: 700; min-width: 22px; font-family: 'Inter', sans-serif; }
    .song-title  { font-family: 'Inter', sans-serif; font-weight: 700; color: #f1f5f9; font-size: 0.88rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 220px; }
    .song-artist { color: #6b7280; font-size: 0.76rem; }
    .song-badge  { font-size: 0.65rem; padding: 2px 7px; border-radius: 999px; font-weight: 600; white-space: nowrap; }
    .badge-genre { background: rgba(52,211,153,0.1); color: #34d399; border: 1px solid rgba(52,211,153,0.2); }
    .sim-score   { color: #c084fc; font-family: 'Inter', sans-serif; font-weight: 800; font-size: 0.85rem; min-width: 38px; text-align: right; }
    .yt-link     { display: inline-flex; align-items: center; gap: 3px; background: rgba(248,113,113,0.1); color: #f87171; border: 1px solid rgba(248,113,113,0.25); border-radius: 999px; padding: 2px 9px; font-size: 0.65rem; font-weight: 600; text-decoration: none !important; white-space: nowrap; }
    .yt-link:hover { background: rgba(248,113,113,0.2); }

    .selector-card  { background: rgba(124,58,237,0.07); border: 1px solid rgba(192,132,252,0.25); border-radius: 12px; padding: 1rem 1.3rem; margin-bottom: 0.75rem; }
    .selector-title { font-family: 'Inter', sans-serif; font-weight: 700; color: #f1f5f9; font-size: 1rem; }
    .selector-meta  { color: #6b7280; font-size: 0.8rem; margin-top: 0.15rem; }

    div[data-testid="stMetricValue"] { color: #c084fc !important; }
</style>
"""

st.html(DARK_CSS)


def yt_url(track: str, artist: str) -> str:
    q = urllib.parse.quote_plus(f"{track} {artist} official audio")
    return f"https://www.youtube.com/results?search_query={q}"

def fmt_feat(val, feat: str) -> str:
    if feat == "tempo":    return f"{val:.0f} BPM"
    if feat == "loudness": return f"{val:.1f} dB"
    return f"{val:.2f}"

def dark_chart_layout(**extra) -> dict:
    base = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#9ca3af')
    base.update(extra)
    return base

RADAR_FEATS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]

def normalise_for_radar(feat_dict: dict, df_ranges: dict) -> dict:
    out = {}
    for f, v in feat_dict.items():
        if f in df_ranges:
            lo, hi = df_ranges[f]
            out[f] = float((v - lo) / (hi - lo)) if hi > lo else 0.0
        else:
            out[f] = float(v)
    return out


FEATURE_DTYPES = {
    "danceability": "float32", "energy": "float32", "loudness": "float32",
    "speechiness": "float32", "acousticness": "float32", "instrumentalness": "float32",
    "liveness": "float32", "valence": "float32", "tempo": "float32",
}
CATEGORY_COLS = ["genre", "language", "region"]
CSV_PATH     = "data/master_songs.csv"
PARQUET_PATH = "data/master_songs.parquet"


@st.cache_resource(show_spinner="Loading music library…")
def load_recommender():
    from utils.recommender import MusicRecommender

    class FastMusicRecommender(MusicRecommender):
        """
        Same class as utils/recommender.py, unchanged — this only adds a
        cached, precomputed normalised matrix so recommend() doesn't have
        to redo that O(N*9) work on every single button click. Overriding
        here means utils/recommender.py itself stays 100% as uploaded.
        """
        def __init__(self, df):
            super().__init__(df)
            norms = np.linalg.norm(self.scaled, axis=1)
            norms[norms == 0] = 1e-9
            self._normed_cache = (self.scaled / norms[:, None]).astype(np.float32)

        def recommend(self, track_name, artist_name="", n=10,
                       language_filter=None, genre_filter=None,
                       region_filter="All", feature_filters=None):
            name_mask = self.df["track_name"].str.lower().str.strip() == track_name.lower().strip()
            if artist_name:
                name_mask &= self.df["artist_name"].str.lower().str.strip() == artist_name.lower().strip()

            idxs = self.df[name_mask].index.tolist()
            if not idxs:
                return pd.DataFrame()

            seed_idx  = idxs[0]
            seed_norm = self._normed_cache[seed_idx]
            sims = self._normed_cache @ seed_norm
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
            top_order   = np.argsort(filtered_sims)[::-1][:n]
            top_indices = [filtered_indices[i] for i in top_order]

            result = self.df.iloc[top_indices].copy()
            result["similarity_score"] = np.round(sims[top_indices], 4)

            cols = ["track_name", "artist_name", "genre", "language", "region",
                    "similarity_score"] + self.feature_names
            cols = [c for c in cols if c in result.columns]
            return result[cols].reset_index(drop=True)

    if os.path.exists(PARQUET_PATH):
        # Parquet is columnar + compressed: on a ~350MB CSV this is typically
        # 3-6x smaller on disk and several times faster to read than
        # pd.read_csv, because it stores dtypes natively (no text parsing)
        # and only decodes the columns we actually ask for.
        df = pd.read_parquet(PARQUET_PATH)
        source = f"Master Dataset · {len(df):,} songs"
    elif os.path.exists(CSV_PATH):
        # Explicit dtypes avoid pandas' slow, memory-hungry type-sniffing
        # pass over the whole file, and category dtype for the repeated
        # text columns (genre/language/region) cuts their memory footprint
        # dramatically versus plain object/string columns.
        dtype_map = {**FEATURE_DTYPES, **{c: "category" for c in CATEGORY_COLS}}
        df = pd.read_csv(CSV_PATH, dtype=dtype_map)
        source = f"Master Dataset · {len(df):,} songs"

        # Cache a Parquet copy so every subsequent app restart / redeploy
        # loads in a fraction of the time instead of re-parsing the CSV.
        try:
            df.to_parquet(PARQUET_PATH, index=False)
        except Exception:
            pass  # pyarrow/fastparquet not installed yet — safe to skip
    else:
        st.error(
            "⚠️ No dataset found. Run `prepare_data.py` to generate `data/master_songs.csv`.",
            icon="🚫",
        )
        st.stop()

    return FastMusicRecommender(df), source

rec, data_source = load_recommender()
stats     = rec.stats()
feat_rngs = rec.feature_ranges()


with st.sidebar:

    st.html('<div style="font-family:\'Inter\',sans-serif;font-size:1.2rem;font-weight:700;color:#a78bfa;letter-spacing:0.18em;text-transform:uppercase;padding-top:6px;">🎵 &nbsp;Filters</div>')

    st.markdown("")
    n_recs = st.slider("Number of recommendations", 5, 25, 10)
    st.markdown("---")

    lang_options    = ["All"] + rec.languages
    lang_filter_sel = st.multiselect("**🗣 Language**", lang_options, placeholder="All languages")
    lang_filter     = [] if (not lang_filter_sel or "All" in lang_filter_sel) else lang_filter_sel

    sidebar_genre_options = ["All"] + rec.genres
    genre_selection  = st.multiselect("**🎼 Genre**", sidebar_genre_options, placeholder="All genres")
    genre_filter = [] if ("All" in genre_selection or not genre_selection) else genre_selection

    st.markdown("---")

    with st.expander("🎚 Advanced Audio Features"):
        st.caption("Filter recommendations by audio characteristics.")
        feature_filters = {}
        for feat, label in FEATURE_LABELS.items():
            if feat not in rec.feature_names:
                continue
            buckets = FEATURE_BUCKETS.get(feat, {})
            if not buckets:
                continue
            chosen = st.multiselect(label, options=list(buckets.keys()), placeholder="Any", key=f"feat_{feat}")
            if chosen:
                feature_filters[feat] = (
                    min(buckets[c][0] for c in chosen),
                    max(buckets[c][1] for c in chosen),
                )
        if feature_filters:
            st.success(f"Active: {', '.join(FEATURE_LABELS[f] for f in feature_filters)}")

    st.markdown("---")
    st.html('<div style="font-family:\'Inter\',sans-serif;font-size:1.2rem;font-weight:700;color:#a78bfa;margin-bottom:0.5rem;letter-spacing:0.18em;text-transform:uppercase;">📊 &nbsp;Dataset Stats</div>')
    st.html(f"""
    <div class="stat-card"><div class="stat-number">{stats['total_songs']:,}</div><div class="stat-label">Total Songs</div></div>
    <div class="stat-card"><div class="stat-number">{stats['artists']:,}</div><div class="stat-label">Unique Artists</div></div>
    <div class="stat-card"><div class="stat-number">{stats['genres']:,}</div><div class="stat-label">Genres</div></div>
    <div class="stat-card"><div class="stat-number">{stats['languages']:,}</div><div class="stat-label">Languages</div></div>
    """)


st.html("""
<div class="hero-wrap">
    <div class="hero-title">Intelligent Music Recommender</div>
    <div class="hero-sub">Discover songs that feel just like your favourite track — powered by acoustic feature analysis &amp; cosine similarity</div>
</div>
""")

st.html("""
<div class="academic-card">
    <div class="eq-container">
        <div class="eq-bar"></div><div class="eq-bar"></div><div class="eq-bar"></div>
        <div class="eq-bar"></div><div class="eq-bar"></div><div class="eq-bar"></div>
        <div class="eq-bar"></div>
    </div>
    <div class="academic-title">INTELLIGENT MUSIC RECOMMENDATION USING MACHINE LEARNING AND ACOUSTIC FEATURE ANALYSIS</div>
    <div class="academic-subtitle">Bachelor of Technology</div>
    <div class="academic-dept">Computer Science Engineering – Data Science</div>
    <div class="grid-container">
        <div class="grid-col">
            <h4>Developed By</h4>
            <p>Abhay Pandit (0002CD231001)</p>
            <p>Aman Sen (0002CD231007)</p>
            <p>Anany Mishra (0002CD231008)</p>
            <p>Varun Gupta (0002CD231075)</p>
        </div>
        <div class="grid-col">
            <h4>Under Guidance Of</h4>
            <p>Mr. Madhav Chaturvedi</p><p class="prof-title">Assistant Professor</p>
            <p>Mr. Gajendra Kumar Ahirwar</p><p class="prof-title">Assistant Professor</p>
        </div>
    </div>
</div>
""")

tab_search, tab_explore, tab_about = st.tabs(["🔍 Recommend", "📈 Explore Dataset", "📖 About"])


with tab_search:

    col_inp, col_btn = st.columns([4, 1])
    with col_inp:
        query = st.text_input("Search", placeholder="Type a song name or artist…",
                              label_visibility="collapsed", key="search_input")
    with col_btn:
        st.button("Search 🔎", use_container_width=True)

    seed_name   = None
    seed_artist = ""

    _BAD = {"nan", "none", "null", "unknown", "n/a", "na", ""}

    if query:
        results = rec.search(query, top_n=40)

        if results.empty:
            st.warning("No songs found. Try a different name.")
        else:
            st.markdown(f"**{len(results)} match(es) found — select your seed song:**")

            def make_label(row):
                lang  = str(row.get("language", "")).strip()
                genre = str(row.get("genre", "")).strip()
                parts = [f"{row['track_name']}  —  {row.get('artist_name', '?')}"]
                meta  = []
                if lang  and lang.lower()  not in _BAD: meta.append(lang)
                if genre and genre.lower() not in _BAD: meta.append(genre[:18])
                if meta: parts.append(f"({', '.join(meta)})")
                return "  ".join(parts)

            labels       = [make_label(results.iloc[i]) for i in range(len(results))]
            chosen_label = st.selectbox("Pick song", labels, label_visibility="collapsed")
            chosen_row   = results.iloc[labels.index(chosen_label)]

            feat_prev = " · ".join([
                f"{f}: {fmt_feat(chosen_row[f], f)}"
                for f in ["danceability", "energy", "valence", "tempo"]
                if f in chosen_row and pd.notna(chosen_row.get(f))
            ])

            lang_disp  = str(chosen_row.get("language", "")).strip()
            genre_disp = str(chosen_row.get("genre", "")).strip()
            if lang_disp.lower()  in _BAD: lang_disp  = "—"
            if genre_disp.lower() in _BAD: genre_disp = "—"

            with st.container(border=True):
                c_info, c_yt = st.columns([3, 1])
                with c_info:
                    st.markdown(f"### 🎵 {chosen_row['track_name']}")
                    meta_parts = [str(chosen_row.get('artist_name', 'Unknown'))]
                    if lang_disp  != "—": meta_parts.append(lang_disp)
                    if genre_disp != "—": meta_parts.append(genre_disp)
                    st.caption("  ·  ".join(meta_parts))
                with c_yt:
                    st.link_button("▶ YouTube",
                                   yt_url(chosen_row['track_name'], str(chosen_row.get('artist_name', ''))),
                                   use_container_width=True)
                if feat_prev:
                    st.markdown(f"<div style='font-size:0.78rem;color:#6b7280;margin-top:0.25rem'>{feat_prev}</div>",
                                unsafe_allow_html=True)

            seed_name   = chosen_row["track_name"]
            seed_artist = str(chosen_row.get("artist_name", ""))

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🎲 Shuffle Random Track", use_container_width=True):
            random_track = rec.df.sample(1).iloc[0]
            st.info(f"💡 Try searching for **{random_track['track_name']}** by {random_track.get('artist_name', '?')}")

    if seed_name:
        with col_b:
            if st.button("✨ Find Similar Songs", type="primary", use_container_width=True):
                with st.spinner("Computing acoustic similarity…"):
                    recs_result = rec.recommend(
                        seed_name, seed_artist, n=n_recs,
                        language_filter=lang_filter  or None,
                        genre_filter=genre_filter    or None,
                        region_filter="All",
                        feature_filters=feature_filters or None,
                    )
                st.session_state["recs"]        = recs_result
                st.session_state["seed_name"]   = seed_name
                st.session_state["seed_artist"] = seed_artist
                st.session_state["seed_feat"]   = rec.get_seed_features(seed_name, seed_artist)

    if "recs" in st.session_state:
        recs_data = st.session_state["recs"]
        s_name    = st.session_state["seed_name"]
        seed_feat = st.session_state.get("seed_feat", {})

        if recs_data.empty:
            st.warning("No songs matched your filters — try relaxing the language/genre/audio feature selections.")
        else:
            st.html(f"<h3 style='font-family:Inter,sans-serif;color:#f1f5f9;font-weight:800;margin:1.5rem 0 1rem;'>Acoustic Matches for: {s_name}</h3>")

            radar_feats = [f for f in RADAR_FEATS if f in seed_feat]
            if radar_feats:
                norm_seed = normalise_for_radar(seed_feat, feat_rngs)
                vals = [norm_seed.get(f, 0) for f in radar_feats] + [norm_seed.get(radar_feats[0], 0)]
                cats = radar_feats + [radar_feats[0]]
                fig_r = go.Figure(go.Scatterpolar(
                    r=vals, theta=cats, fill='toself',
                    line_color='#c084fc', fillcolor='rgba(192,132,252,0.12)',
                ))
                fig_r.update_layout(
                    polar=dict(
                        bgcolor='rgba(0,0,0,0)',
                        radialaxis=dict(
                            visible=True, range=[0, 1],
                            gridcolor='rgba(255,255,255,0.08)',
                            tickfont=dict(color='#4b5563', size=8),
                        ),
                        angularaxis=dict(
                            gridcolor='rgba(255,255,255,0.06)',
                            tickfont=dict(color='#9ca3af', size=10),
                        ),
                    ),
                    **dark_chart_layout(margin=dict(t=30, b=20, l=60, r=60), height=380, showlegend=False),
                )
                with st.expander("🔬 Audio Fingerprint of Seed Song"):
                    st.plotly_chart(fig_r, use_container_width=True)

            for i, row in recs_data.iterrows():
                track  = row["track_name"]
                artist = str(row.get("artist_name", ""))
                genre  = str(row.get("genre", "—"))[:20]
                if genre.lower() in _BAD: genre = "—"

                st.html(f"""
<div class="song-row">
    <span class="song-rank">#{i+1}</span>
    <div style="flex:1;min-width:0">
        <div class="song-title">{track}</div>
        <div class="song-artist">{artist}</div>
    </div>
    <span class="song-badge badge-genre">{genre}</span>
    <a class="yt-link" href="{yt_url(track, artist)}" target="_blank">▶ YouTube</a>
</div>""")

            comp_feats = [f for f in RADAR_FEATS if f in seed_feat and f in recs_data.columns]
            if comp_feats and len(recs_data) >= 2:
                with st.expander("🆚 Compare Seed vs Top Recommendations"):
                    fig_comp = go.Figure()
                    norm_seed = normalise_for_radar(seed_feat, feat_rngs)
                    sv = [norm_seed.get(f, 0) for f in comp_feats] + [norm_seed.get(comp_feats[0], 0)]
                    fig_comp.add_trace(go.Scatterpolar(
                        r=sv, theta=comp_feats + [comp_feats[0]],
                        fill='toself', name=s_name[:20],
                        line_color='#c084fc', fillcolor='rgba(192,132,252,0.1)',
                    ))
                    for j, (_, row) in enumerate(recs_data.head(3).iterrows()):
                        raw_vals  = {f: float(row[f]) if pd.notna(row.get(f)) else 0.0 for f in comp_feats}
                        norm_vals = normalise_for_radar(raw_vals, feat_rngs)
                        rv  = [norm_vals.get(f, 0) for f in comp_feats] + [norm_vals.get(comp_feats[0], 0)]
                        clr = ['#38bdf8', '#34d399', '#fbbf24'][j]
                        fig_comp.add_trace(go.Scatterpolar(
                            r=rv, theta=comp_feats + [comp_feats[0]], fill='toself',
                            name=row['track_name'][:18], line_color=clr,
                            fillcolor=f'rgba({int(clr[1:3],16)},{int(clr[3:5],16)},{int(clr[5:7],16)},0.06)',
                        ))
                    fig_comp.update_layout(
                        polar=dict(
                            bgcolor='rgba(0,0,0,0)',
                            radialaxis=dict(
                                range=[0, 1],
                                gridcolor='rgba(255,255,255,0.08)',
                                tickfont=dict(color='#4b5563', size=8),
                            ),
                            angularaxis=dict(
                                gridcolor='rgba(255,255,255,0.06)',
                                tickfont=dict(color='#9ca3af', size=10),
                            ),
                        ),
                        **dark_chart_layout(
                            legend=dict(font=dict(color='#9ca3af', size=11), bgcolor='rgba(0,0,0,0)'),
                            margin=dict(t=30, b=20, l=60, r=60), height=420,
                        ),
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)

            safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', s_name[:20])
            st.download_button(
                "⬇ Download Recommendations CSV",
                data=recs_data.to_csv(index=False),
                file_name=f"recs_{safe_name}.csv",
                mime="text/csv",
                key="download_recs_csv",
            )


with tab_explore:
    df = rec.df

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Songs",    f"{stats['total_songs']:,}")
    c2.metric("Unique Artists", f"{stats['artists']:,}")
    c3.metric("Genres",         f"{stats['genres']:,}")
    c4.metric("Languages",      stats['languages'])

    col_a, col_b = st.columns(2)

    with col_a:
        if "language" in df.columns:
            lc = df["language"].value_counts().head(15)
            lc = lc[~lc.index.str.lower().isin(["unknown", "nan", ""])]
            fig_lang = px.bar(x=lc.values, y=lc.index, orientation="h",
                              color=lc.values, color_continuous_scale=["#34d399", "#c084fc"],
                              labels={"x": "Songs", "y": ""}, title="Top Languages")
            fig_lang.update_layout(**dark_chart_layout(
                height=360, coloraxis_showscale=False,
                title_font_color='#e2e8f0', title_font_family='Inter',
                yaxis=dict(autorange="reversed"), margin=dict(l=0, r=0, t=40, b=10),
            ))
            st.plotly_chart(fig_lang, use_container_width=True)

    with col_b:
        if "genre" in df.columns:
            gc = df["genre"].value_counts().head(15)
            gc = gc[~gc.index.str.lower().isin(["unknown", "nan", ""])]
            fig_genre_top = px.bar(x=gc.values, y=gc.index, orientation="h",
                                   color=gc.values, color_continuous_scale=["#818cf8", "#38bdf8"],
                                   labels={"x": "Songs", "y": ""}, title="Top Genres")
            fig_genre_top.update_layout(**dark_chart_layout(
                height=360, coloraxis_showscale=False,
                title_font_color='#e2e8f0', title_font_family='Inter',
                yaxis=dict(autorange="reversed"), margin=dict(l=0, r=0, t=40, b=10),
            ))
            st.plotly_chart(fig_genre_top, use_container_width=True)

    st.markdown("#### Audio Feature Distributions")

    unit_feats = [f for f in ["danceability", "energy", "speechiness", "acousticness",
                               "instrumentalness", "liveness", "valence"] if f in df.columns]

    if unit_feats:
        plot_df  = df[unit_feats].copy().apply(pd.to_numeric, errors="coerce").dropna()
        palette  = ["#c084fc", "#818cf8", "#38bdf8", "#34d399", "#fbbf24", "#fb923c", "#f472b6"]
        labels   = [f.capitalize() for f in unit_feats]
        means    = [plot_df[f].mean() for f in unit_feats]
        stds     = [plot_df[f].std()  for f in unit_feats]

        fig_bar = go.Figure()
        for feat, mean, std, clr in zip(labels, means, stds, palette):
            fig_bar.add_trace(go.Bar(
                x=[mean], y=[feat], orientation='h', name=feat,
                marker=dict(color=clr, opacity=0.88, line=dict(width=0)),
                error_x=dict(type='data', array=[std], color='rgba(255,255,255,0.35)', thickness=2, width=6),
                hovertemplate=f"<b>{feat}</b><br>Mean: {mean:.3f}<br>Std Dev: {std:.3f}<extra></extra>",
                width=0.55,
            ))
        annotations = [
            dict(x=mean + 0.02, y=feat, text=f"{mean:.2f}", xanchor='left', yanchor='middle',
                 font=dict(color='#9ca3af', size=11, family='Inter'), showarrow=False)
            for feat, mean in zip(labels, means)
        ]
        fig_bar.update_layout(
            **dark_chart_layout(height=380, showlegend=False, barmode='group', annotations=annotations),
            xaxis=dict(
                title=dict(text="Average Value (0 – 1)", font=dict(color='#6b7280', size=11)),
                range=[0, 1.15], gridcolor='rgba(255,255,255,0.06)',
                tickfont=dict(color='#6b7280', size=10), zeroline=False,
            ),
            yaxis=dict(
                tickfont=dict(color='#c0c0c8', size=12, family='Inter'),
                gridcolor='rgba(0,0,0,0)', categoryorder='array',
                categoryarray=list(reversed(labels)),
            ),
            margin=dict(l=10, r=60, t=10, b=40),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption("Mean audio feature values (0–1 scale). Error bars = ±1 standard deviation.")

    if "genre" in df.columns:
        st.markdown("#### All Genres")
        gc2 = df["genre"].value_counts().head(25)
        gc2 = gc2[~gc2.index.str.lower().isin(["unknown", "nan", ""])]
        fig_genre2 = px.bar(x=gc2.index, y=gc2.values, color=gc2.values,
                            color_continuous_scale=["#34d399", "#818cf8", "#c084fc"],
                            labels={"x": "Genre", "y": "Songs"})
        fig_genre2.update_layout(**dark_chart_layout(
            height=340, coloraxis_showscale=False,
            xaxis=dict(gridcolor='rgba(255,255,255,0.04)', tickangle=-35,
                       tickfont=dict(color='#6b7280')),
            yaxis=dict(gridcolor='rgba(255,255,255,0.04)', tickfont=dict(color='#6b7280')),
            margin=dict(l=0, r=0, t=20, b=80),
        ))
        st.plotly_chart(fig_genre2, use_container_width=True)


with tab_about:
    st.markdown("""
## 🎵 About This Project

### Architecture
A content-based music recommendation engine using Spotify audio features and cosine similarity. The system is trained on a global multi-million-song dataset and computes similarity on-demand — no NxN matrix, making it scalable to 1M+ songs across any genre or language.

### How Recommendations Work
1. **Feature Extraction** — Each song becomes a 9-dimensional vector of Spotify audio features
2. **Normalisation** — Min-Max scaling to [0, 1] for fair comparison
3. **Cosine Similarity** — One seed vector vs all N songs via fast matrix multiplication
4. **Filtering** — Language, Genre, and Audio Feature bucket filters
5. **Ranking** — Top-N most similar songs returned with confidence scores

### Audio Features

| Feature | Range | What it measures |
|---|---|---|
| Danceability | 0–1 | Rhythmic suitability for dancing |
| Energy | 0–1 | Intensity & perceptual activity |
| Loudness | dB | Average loudness |
| Speechiness | 0–1 | Spoken word presence |
| Acousticness | 0–1 | Acoustic confidence |
| Instrumentalness | 0–1 | No-vocals prediction |
| Liveness | 0–1 | Live audience presence |
| Valence | 0–1 | Musical positiveness (happiness) |
| Tempo | BPM | Beats per minute |

### Dataset
The recommender is trained on a combined global dataset drawn from multiple Spotify data sources, covering millions of tracks spanning Pop, Rock, Hip-Hop, Jazz, Electronic, Classical, Folk, Anime, Country, R&B, Indie, and many more genres. Data preparation merges, deduplicates, and standardises all sources into a single optimised CSV.

### Tech Stack
`Python` · `Streamlit` · `scikit-learn` · `pandas` · `NumPy` · `Plotly`
""")