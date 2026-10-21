"""OpinionScope - Opinion Mining & Tag-Based Categorization dashboard.

Run:  streamlit run app.py
Data: data/processed_reviews.csv with columns
      review_text, cleaned_text, sentiment, sentiment_score  (+ optional like_count, published_at,
      tokens, lemmas, stems)
"""
from __future__ import annotations

import html
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_FILE = BASE_DIR / "data" / "processed_reviews.csv"

st.set_page_config(page_title="Opinion Mining ", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

# ======================================================================
# Theme: one set of design tokens drives CSS *and* charts, light or dark
# ======================================================================
LIGHT = dict(
    mode="light", bg="#F6F7FB", panel="#FFFFFF", ink="#101828", muted="#667085", line="#E4E7EC",
    grid="#EEF0F4", accent="#4F46E5", accent_lo="#D9DBFF", accent_soft="#EEF0FF",
    pos="#12805C", neu="#C98A0B", neg="#D14343", hero_a="#3B36C9", hero_b="#6D5BF5",
    shadow="0 1px 2px rgba(16,24,40,.05), 0 6px 18px rgba(16,24,40,.05)",
    colorway=["#4F46E5", "#0EA5E9", "#12805C", "#C98A0B", "#D14343", "#8B5CF6", "#EC4899", "#64748B"],
)
DARK = dict(
    mode="dark", bg="#0B1020", panel="#121A2E", ink="#E6EAF4", muted="#9AA6BD", line="#232C45",
    grid="#1E2740", accent="#8B8CFF", accent_lo="#2A3170", accent_soft="#1B2245",
    pos="#34D399", neu="#FBBF24", neg="#F87171", hero_a="#2B2A7A", hero_b="#4B3FB8",
    shadow="0 1px 2px rgba(0,0,0,.4), 0 6px 18px rgba(0,0,0,.35)",
    colorway=["#8B8CFF", "#38BDF8", "#34D399", "#FBBF24", "#F87171", "#C4B5FD", "#F472B6", "#94A3B8"],
)


def detect_theme() -> str:
    """The theme in .streamlit/config.toml wins, so native widgets and our CSS always agree."""
    try:
        base = st.get_option("theme.base")
        if base in ("light", "dark"):
            return base
    except Exception:
        pass
    try:  # no config: follow the active browser theme (Streamlit >= 1.46)
        t = st.context.theme.type
        if t in ("light", "dark"):
            return t
    except Exception:
        pass
    return "light"


T = DARK if detect_theme() == "dark" else LIGHT
FONT = "Inter, 'Segoe UI', system-ui, -apple-system, Arial, sans-serif"
SENT_ORDER = ["Positive", "Neutral", "Negative"]
SENT_COLORS = {"Positive": T["pos"], "Neutral": T["neu"], "Negative": T["neg"]}

CSS = Template("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, [data-testid="stAppViewContainer"] { font-family: ${font}; }
.stApp { background: ${bg}; color: ${ink}; }
[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
.block-container { max-width: 1480px; padding: 1.6rem 2.2rem 3rem; }
[data-testid="stSidebar"] { background: ${panel}; border-right: 1px solid ${line}; }
[data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }

.brand { display:flex; align-items:center; gap:.65rem; font-weight:800; font-size:1.12rem; color:${ink}; letter-spacing:-.02em; }
.brand .logo { width:32px; height:32px; border-radius:10px; display:grid; place-items:center; color:#fff; font-size:.95rem;
  background: linear-gradient(135deg, ${accent}, #8B5CF6); }
.side-label { color:${muted}; font-size:.68rem; font-weight:700; letter-spacing:.13em; text-transform:uppercase; margin:1.2rem 0 .45rem; }

[data-testid="stSidebar"] [role="radiogroup"] { gap:.15rem; }
[data-testid="stSidebar"] [role="radiogroup"] label { padding:.5rem .7rem; border-radius:10px; width:100%; cursor:pointer; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background:${accent_soft}; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background:${accent_soft}; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p { color:${accent}; font-weight:650; }

.hero { display:flex; justify-content:space-between; align-items:flex-end; gap:1.5rem; flex-wrap:wrap;
  padding:1.5rem 1.8rem; border-radius:18px; color:#fff; margin-bottom:1.1rem;
  background: linear-gradient(120deg, ${hero_a}, ${hero_b}); box-shadow:${shadow}; }
.hero .eyebrow { font-size:.7rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; opacity:.8; }
.hero h1 { font-size:2rem; font-weight:800; letter-spacing:-.04em; margin:.15rem 0 .25rem; color:#fff !important; padding:0; }
.hero p { margin:0; opacity:.88; font-size:.93rem; }
.hero .count { text-align:right; }
.hero .count b { font-size:1.9rem; letter-spacing:-.03em; display:block; line-height:1.1; }
.hero .count span { font-size:.78rem; opacity:.8; }
.chips { margin:.8rem 0 0; }
.chip { display:inline-block; background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.22);
  padding:.18rem .65rem; border-radius:999px; font-size:.74rem; margin:0 .35rem .3rem 0; }

.kpi { background:${panel}; border:1px solid ${line}; border-radius:14px; padding:1rem 1.15rem; box-shadow:${shadow}; height:100%; }
.kpi .label { color:${muted}; font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.07em; }
.kpi .value { color:${ink}; font-size:1.85rem; font-weight:760; letter-spacing:-.035em; margin:.3rem 0 .05rem; line-height:1.15; }
.kpi .sub { color:${muted}; font-size:.78rem; }
.kpi .bar { height:5px; background:${grid}; border-radius:99px; margin-top:.65rem; overflow:hidden; }
.kpi .bar span { display:block; height:100%; border-radius:99px; }

.insight { background:${panel}; border:1px solid ${line}; border-left:4px solid ${accent}; border-radius:12px;
  padding:.9rem 1.1rem; box-shadow:${shadow}; height:100%; }
.insight .t { color:${muted}; font-size:.7rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase; }
.insight .v { color:${ink}; font-size:1.05rem; font-weight:700; margin:.25rem 0 .1rem; }
.insight .d { color:${muted}; font-size:.82rem; }

/* explicit text colours: never depend on the native theme for readability */
[data-testid="stSidebar"] label, [data-testid="stSidebar"] label p, [data-testid="stSidebar"] label span,
[data-testid="stWidgetLabel"] p, [data-testid="stSidebar"] [role="radiogroup"] label p { color:${ink} !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] { color:${muted} !important; }
[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4, [data-testid="stMarkdownContainer"] h5, [data-testid="stMarkdownContainer"] h6 { color:${ink}; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color:${ink}; }
.hero p, .hero span, .hero b { color:#fff !important; }
.kpi .label, .kpi .sub, .insight .t, .insight .d, .page-sub { color:${muted} !important; }
.kpi .value, .insight .v, .page-title { color:${ink} !important; }
[data-baseweb="popover"] li, [data-baseweb="menu"] li { color:${ink}; }
.page-title { color:${ink}; font-size:1.5rem; font-weight:780; letter-spacing:-.035em; margin:1.3rem 0 .15rem; }
.page-sub { color:${muted}; font-size:.9rem; margin-bottom:1rem; }
.pill { display:inline-block; padding:.1rem .55rem; border-radius:999px; font-size:.72rem; font-weight:650; }
[data-testid="stDownloadButton"] button, [data-testid="stSidebar"] button { border-radius:10px; }
</style>
""")
st.markdown(CSS.safe_substitute(**{k: v for k, v in T.items() if isinstance(v, str)}, font=FONT),
            unsafe_allow_html=True)

# ======================================================================
# Streamlit-version-safe helpers
# ======================================================================
try:
    _VER = tuple(int(x) for x in re.findall(r"\d+", st.__version__)[:2])
except Exception:
    _VER = (1, 0)
NEW_API = _VER >= (1, 50)


def chart(fig, key=None):
    if NEW_API:
        st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False})
    else:
        st.plotly_chart(fig, use_container_width=True, key=key, config={"displaylogo": False})


def table(frame, height=None, column_config=None):
    kw = dict(hide_index=True, column_config=column_config)
    if height:
        kw["height"] = height
    try:
        st.dataframe(frame, **kw, **({"width": "stretch"} if NEW_API else {"use_container_width": True}))
    except Exception:
        st.dataframe(frame, hide_index=True)


def image(arr, caption=None):
    try:
        st.image(arr, caption=caption, **({"width": "stretch"} if NEW_API else {"use_container_width": True}))
    except Exception:
        st.image(arr, caption=caption)


def style_fig(fig, height=360):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=50, b=8), colorway=T["colorway"],
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=T["muted"]),
        title=dict(font=dict(size=14, color=T["ink"]), x=0, xanchor="left"),
        legend=dict(title_text="", orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(color=T["ink"], size=12)),
        hoverlabel=dict(bgcolor=T["panel"], font_color=T["ink"], bordercolor=T["line"]),
    )
    fig.update_xaxes(gridcolor=T["grid"], zeroline=False, linecolor=T["line"], tickfont=dict(color=T["muted"]))
    fig.update_yaxes(gridcolor=T["grid"], zeroline=False, linecolor=T["line"], tickfont=dict(color=T["muted"]))
    return fig


def hbar(frame, x, y, title, color=None, height=None):
    d = frame.sort_values(x)
    fig = px.bar(d, x=x, y=y, orientation="h", title=title, color=color,
                 **({} if color else {}),
                 **({"color_continuous_scale": [[0, T["accent_lo"]], [1, T["accent"]]]}
                    if color and pd.api.types.is_numeric_dtype(d[color]) else {}))
    fig.update_coloraxes(showscale=False)
    fig.update_yaxes(title=None, showgrid=False)
    fig.update_xaxes(title=None)
    if not color:
        fig.update_traces(marker_color=T["accent"])
    return style_fig(fig, height or max(340, 26 * len(d) + 80))


def kpi(label, value, sub="", color=None, bar=None):
    bar_html = (f'<div class="bar"><span style="width:{max(0, min(bar, 100)):.1f}%;background:{color}"></span></div>'
                if bar is not None and color else "")
    st.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
                f'<div class="sub">{sub}</div>{bar_html}</div>', unsafe_allow_html=True)


def page_header(title, sub):
    st.markdown(f'<div class="page-title">{title}</div><div class="page-sub">{sub}</div>', unsafe_allow_html=True)


def download_csv(frame, filename, label):
    st.download_button(label, frame.to_csv(index=False).encode("utf-8-sig"), file_name=filename, mime="text/csv")


# ======================================================================
# Data
# ======================================================================
ALIASES = {"text": "review_text", "comment": "review_text", "clean": "cleaned_text",
           "compound": "sentiment_score", "likes": "like_count", "votes": "like_count", "time": "published_at"}


@st.cache_data(show_spinner="Loading dataset...")
def load_data(path: str, mtime: float) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d.rename(columns={k: v for k, v in ALIASES.items() if k in d.columns and v not in d.columns})
    required = {"review_text", "cleaned_text", "sentiment", "sentiment_score"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    for c in ["review_text", "cleaned_text", "sentiment"]:
        d[c] = d[c].fillna("").astype(str)
    d["sentiment"] = d["sentiment"].str.strip().str.title()
    d["sentiment_score"] = pd.to_numeric(d["sentiment_score"], errors="coerce").fillna(0.0)
    if "like_count" not in d.columns:
        d["like_count"] = 0
    d["like_count"] = pd.to_numeric(d["like_count"], errors="coerce").fillna(0).astype(int)
    if "published_at" in d.columns:
        d["published_at"] = pd.to_datetime(d["published_at"], errors="coerce", utc=True)
    return d.reset_index(drop=True)


if not PROCESSED_FILE.exists():
    st.error("Processed dataset not found.")
    st.info("Place the collected feedback at `data/reviews.csv`, run `python analysis.py`, "
            "and make sure it writes `data/processed_reviews.csv`.")
    st.stop()
try:
    df = load_data(str(PROCESSED_FILE), PROCESSED_FILE.stat().st_mtime)
except Exception as e:
    st.error(f"{e}. Re-run `python analysis.py`.")
    st.stop()

# ======================================================================
# NLP tables (cached; recomputed only when the filtered subset changes)
# ======================================================================
STOP_TAGS = {"thing", "things", "someone", "something", "today", "video", "youtube", "channel", "like", "get", "one"}
CATEGORY_WORDS = {
    "Performance": {"slow", "lag", "laggy", "crash", "crashes", "crashing", "freeze", "bug", "bugs", "error", "errors", "fast", "speed", "battery"},
    "Usability": {"easy", "simple", "confusing", "difficult", "interface", "design", "navigation", "experience", "screen", "display"},
    "Quality": {"quality", "excellent", "poor", "great", "bad", "good", "amazing", "worst", "best", "camera", "build"},
    "Cost & Value": {"price", "cost", "expensive", "cheap", "value", "payment", "paid", "free", "money"},
    "Support": {"support", "service", "help", "customer", "response", "team"},
    "Features": {"feature", "features", "option", "options", "update", "updates", "function"},
    "Reliability": {"reliable", "reliability", "stable", "unstable", "working", "works", "issue", "issues"},
}


def categorize(term: str) -> str:
    words = set(str(term).lower().split())
    for name, vocab in CATEGORY_WORDS.items():
        if words & vocab:
            return name
    return "General"


@lru_cache(maxsize=1)
def _stemmer():
    try:
        from nltk.stem import PorterStemmer
        return PorterStemmer().stem
    except Exception:
        return lambda w: re.sub(r"(ing|ed|es|s)$", "", w) if len(w) > 4 else w


@lru_cache(maxsize=4096)
def _synonyms(word: str) -> str:
    try:
        from nltk.corpus import wordnet
        out = []
        for syn in wordnet.synsets(word)[:2]:
            for lemma in syn.lemma_names():
                lemma = lemma.replace("_", " ").lower()
                if lemma != word and lemma not in out:
                    out.append(lemma)
        return ", ".join(out[:3])
    except Exception:
        return ""


def _min_df(n_docs: int) -> int:
    return 2 if n_docs >= 200 else 1


def _valid(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["cleaned_text"].str.strip().ne("")]


@st.cache_data(show_spinner=False)
def vectorize(texts: tuple, kind: str, ngram: tuple):
    """Return (terms, matrix) for 'bow' or 'tfidf'."""
    cls = CountVectorizer if kind == "bow" else TfidfVectorizer
    kw = dict(ngram_range=ngram, max_features=3000, min_df=_min_df(len(texts)), token_pattern=r"(?u)\b\w+\b")
    if kind == "tfidf":
        kw["sublinear_tf"] = True
    vec = cls(**kw)
    mat = vec.fit_transform(texts)
    return vec.get_feature_names_out(), mat.tocsr()


@st.cache_data(show_spinner=False)
def svd_projection(texts: tuple, max_comp: int = 50):
    terms, mat = vectorize(texts, "tfidf", (1, 2))
    k = min(max_comp, mat.shape[1] - 1, mat.shape[0] - 1)
    if k < 2:
        return None
    svd = TruncatedSVD(n_components=k, random_state=42)
    Z = svd.fit_transform(mat)
    top = {}
    for i in range(2):
        idx = np.argsort(svd.components_[i])
        top[i] = {"positive pole": [terms[j] for j in idx[::-1][:8]], "negative pole": [terms[j] for j in idx[:8]]}
    return Z[:, :2], np.cumsum(svd.explained_variance_ratio_), svd.explained_variance_ratio_[:2], mat.shape, top


@st.cache_data(show_spinner="Building tags...")
def build_tag_stages(frame: pd.DataFrame, top_refined: int = 40, top_enriched: int = 25):
    """Stage 1 ranked -> stage 2 refined -> stage 3 enriched."""
    valid = _valid(frame)
    if len(valid) < 2:
        return (pd.DataFrame(),) * 3
    texts = tuple(valid["cleaned_text"])
    terms, bow = vectorize(texts, "bow", (1, 2))
    _, tf = vectorize(texts, "tfidf", (1, 2))
    # --- 1. rank: blend of normalised frequency and mean TF-IDF
    r = pd.DataFrame({
        "term": terms,
        "frequency": np.asarray(bow.sum(axis=0)).ravel(),
        "reviews": np.asarray((bow > 0).sum(axis=0)).ravel(),
        "mean_tfidf": np.asarray(tf.mean(axis=0)).ravel(),
    })
    r = r[~r["term"].isin(STOP_TAGS)].copy()
    r["rank_score"] = .55 * r["frequency"] / max(r["frequency"].max(), 1) + .45 * r["mean_tfidf"] / max(r["mean_tfidf"].max(), 1e-12)
    ranked = r.sort_values(["rank_score", "frequency"], ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    ranked["tag"] = ranked["term"].str.title()

    # --- 2. refine: junk filter, stem de-duplication, drop unigrams absorbed by a bigram
    stem, seen, keep = _stemmer(), set(), []
    for row in ranked.head(200).itertuples():
        words = row.term.split()
        if any(len(w) < 3 or w.isdigit() for w in words) or len(set(words)) < len(words):
            continue
        key = " ".join(sorted(stem(w) for w in words))
        if key in seen:
            continue
        seen.add(key)
        keep.append(row._asdict())
    refined = pd.DataFrame(keep)
    if refined.empty:
        return ranked, refined, refined
    bigrams = [(b["term"].split(), b["frequency"]) for _, b in refined[refined["term"].str.contains(" ")].iterrows()]
    absorbed = refined.apply(lambda x: (" " not in x["term"]) and any(
        x["term"] in w and f >= .8 * x["frequency"] for w, f in bigrams), axis=1)
    refined = refined[~absorbed].head(top_refined).reset_index(drop=True)
    refined["rank"] = range(1, len(refined) + 1)

    # --- 3. enrich: category, related terms, synonyms, sentiment profile
    rows = []
    for row in refined.head(top_enriched).itertuples():
        mask = valid["cleaned_text"].str.contains(rf"(?<!\w){re.escape(row.term)}(?!\w)", regex=True, na=False)
        sub = valid[mask]
        if sub.empty:
            continue
        co = Counter(w for t in sub["cleaned_text"] for w in set(t.split()) if w not in row.term.split() and len(w) > 2)
        c = sub["sentiment"].value_counts()
        n = len(sub)
        pos, neu, neg = (int(c.get(s, 0)) for s in SENT_ORDER)
        rows.append({
            "Tag": row.tag, "Category": categorize(row.term), "Reviews": n,
            "Related terms": ", ".join(w for w, _ in co.most_common(3)),
            "Synonyms": _synonyms(row.term) if " " not in row.term else "",
            "Positive": pos, "Neutral": neu, "Negative": neg,
            "Net sentiment": round(100 * (pos - neg) / n, 1),
            "Dominant": max(zip((pos, neu, neg), SENT_ORDER))[1],
            "Mean score": round(float(sub["sentiment_score"].mean()), 3),
            "rank_score": row.rank_score,
        })
    return ranked, refined, pd.DataFrame(rows)


# ======================================================================
# Sidebar: navigation + filters (reset handled via callback = no crash)
# ======================================================================
PAGES = ["Overview", "Preprocessing", "Bag of Words", "TF-IDF", "Dimensionality Reduction",
         "Sentiment", "Tag Intelligence", "Review Explorer"]
like_min, like_max = int(df["like_count"].min()), int(df["like_count"].max())
has_dates = "published_at" in df.columns and df["published_at"].notna().any()
if has_dates:
    d_min, d_max = df["published_at"].min().date(), df["published_at"].max().date()


def reset_filters():
    st.session_state["flt_sent"] = SENT_ORDER.copy()
    st.session_state["flt_kw"] = ""
    if like_max > like_min:
        st.session_state["flt_likes"] = (like_min, like_max)
    if has_dates and d_min != d_max:
        st.session_state["flt_dates"] = (d_min, d_max)


st.session_state.setdefault("flt_sent", SENT_ORDER.copy())
st.session_state.setdefault("flt_kw", "")
if like_max > like_min:
    st.session_state.setdefault("flt_likes", (like_min, like_max))
if has_dates and d_min != d_max:
    st.session_state.setdefault("flt_dates", (d_min, d_max))

with st.sidebar:
    st.markdown('<div class="brand"><div class="logo">◎</div>Opinion Mining</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-label">Workspace</div>', unsafe_allow_html=True)
    page = st.radio("Navigate", PAGES, label_visibility="collapsed", key="page")
    st.markdown('<div class="side-label">Filters</div>', unsafe_allow_html=True)
    sent_sel = st.multiselect("Sentiment", SENT_ORDER, key="flt_sent")
    if like_max > like_min:
        likes_rng = st.slider("Likes", like_min, like_max, key="flt_likes")
    else:
        likes_rng = (like_min, like_max)
    if has_dates and d_min != d_max:
        dates_rng = st.date_input("Date range", min_value=d_min, max_value=d_max, key="flt_dates")
    else:
        dates_rng = None
    keyword = st.text_input("Contains keyword", placeholder="e.g. camera, battery, slow", key="flt_kw")
    st.button("Reset filters", on_click=reset_filters, use_container_width=True)
    st.caption(f"{len(df):,} reviews in dataset · every page uses the filtered set")

filtered = df[df["sentiment"].isin(sent_sel)] if sent_sel else df.iloc[0:0]
filtered = filtered[filtered["like_count"].between(*likes_rng)]
if dates_rng and len(dates_rng) == 2 and has_dates:
    s, e = pd.Timestamp(dates_rng[0], tz="UTC"), pd.Timestamp(dates_rng[1], tz="UTC") + pd.Timedelta(days=1)
    filtered = filtered[filtered["published_at"].isna() | filtered["published_at"].between(s, e)]
kw = keyword.strip()
if kw:
    filtered = filtered[filtered["review_text"].str.contains(kw, case=False, regex=False, na=False)]
n = len(filtered)

# ======================================================================
# Header + KPIs
# ======================================================================
chips = [f"Sentiment: {', '.join(sent_sel) if sent_sel and len(sent_sel) < 3 else 'All'}",
         f"Likes: {likes_rng[0]}–{likes_rng[1]}"]
if kw:
    chips.append(f"Keyword: {html.escape(kw)}")
st.markdown(
    f'<div class="hero"><div><div class="eyebrow">Opinion mining and Tag-Based Categorization of User Feedback</div>'
    
    f'<div class="chips">{"".join(f"<span class=chip>{c}</span>" for c in chips)}</div></div>'
    f'<div class="count"><b>{n:,}</b><span>of {len(df):,} reviews in view</span></div></div>',
    unsafe_allow_html=True)

if not n:
    st.warning("No reviews match these filters. Use **Reset filters** in the sidebar.")
    st.stop()

pos_p = 100 * filtered["sentiment"].eq("Positive").mean()
neu_p = 100 * filtered["sentiment"].eq("Neutral").mean()
neg_p = 100 * filtered["sentiment"].eq("Negative").mean()
mean_score = filtered["sentiment_score"].mean()
k = st.columns(5, gap="medium")
with k[0]: kpi("Reviews in view", f"{n:,}", f"{100 * n / len(df):.0f}% of dataset", T["accent"], 100 * n / len(df))
with k[1]: kpi("Positive", f"{pos_p:.1f}%", f"{int(round(pos_p * n / 100)):,} reviews", T["pos"], pos_p)
with k[2]: kpi("Neutral", f"{neu_p:.1f}%", f"{int(round(neu_p * n / 100)):,} reviews", T["neu"], neu_p)
with k[3]: kpi("Negative", f"{neg_p:.1f}%", f"{int(round(neg_p * n / 100)):,} reviews", T["neg"], neg_p)
with k[4]: kpi("Mean VADER score", f"{mean_score:+.3f}", "range −1 to +1", T["accent"], (mean_score + 1) * 50)

valid = _valid(filtered)
texts = tuple(valid["cleaned_text"])

# ======================================================================
# Pages
# ======================================================================
if page == "Overview":
    page_header("Overview", "A snapshot of the feedback currently in view.")
    ranked, refined, enriched = build_tag_stages(filtered[["cleaned_text", "sentiment", "sentiment_score"]])
    if not enriched.empty:
        top = enriched.sort_values("rank_score", ascending=False).iloc[0]
        solid = enriched[enriched["Reviews"] >= 3]
        c = st.columns(3, gap="medium")
        with c[0]:
            st.markdown(f'<div class="insight"><div class="t">Top theme</div><div class="v">{html.escape(top["Tag"])}</div>'
                        f'<div class="d">{top["Category"]} · in {top["Reviews"]} reviews · net sentiment {top["Net sentiment"]:+.0f}</div></div>',
                        unsafe_allow_html=True)
        if not solid.empty:
            best, worst = solid.sort_values("Net sentiment").iloc[-1], solid.sort_values("Net sentiment").iloc[0]
            with c[1]:
                st.markdown(f'<div class="insight" style="border-left-color:{T["pos"]}"><div class="t">Most praised tag</div>'
                            f'<div class="v">{html.escape(best["Tag"])}</div><div class="d">Net sentiment {best["Net sentiment"]:+.0f} across {best["Reviews"]} reviews</div></div>',
                            unsafe_allow_html=True)
            with c[2]:
                st.markdown(f'<div class="insight" style="border-left-color:{T["neg"]}"><div class="t">Most criticised tag</div>'
                            f'<div class="v">{html.escape(worst["Tag"])}</div><div class="d">Net sentiment {worst["Net sentiment"]:+.0f} across {worst["Reviews"]} reviews</div></div>',
                            unsafe_allow_html=True)
    st.markdown("")
    l, r = st.columns(2, gap="large")
    counts = filtered["sentiment"].value_counts().reindex(SENT_ORDER, fill_value=0).rename_axis("Sentiment").reset_index(name="Reviews")
    with l, st.container(border=True):
        fig = px.pie(counts, names="Sentiment", values="Reviews", hole=.66, color="Sentiment",
                     color_discrete_map=SENT_COLORS, title="Sentiment mix")
        fig.update_traces(textinfo="percent+label", marker=dict(line=dict(color=T["panel"], width=3)))
        fig.update_layout(showlegend=False)
        chart(style_fig(fig, 340))
    with r, st.container(border=True):
        fig = px.histogram(filtered, x="sentiment_score", nbins=30, color="sentiment", color_discrete_map=SENT_COLORS,
                           category_orders={"sentiment": SENT_ORDER}, title="VADER score distribution",
                           labels={"sentiment_score": "Compound score"})
        fig.update_layout(barmode="stack", bargap=.04)
        fig.update_yaxes(title=None)
        chart(style_fig(fig, 340))
    if has_dates:
        dts = filtered.dropna(subset=["published_at"])
        if dts["published_at"].dt.normalize().nunique() >= 3:
            span = (dts["published_at"].max() - dts["published_at"].min()).days
            freq = "D" if span <= 45 else "W" if span <= 400 else "M"
            tl = dts.assign(period=dts["published_at"].dt.tz_convert(None).dt.to_period(freq).dt.start_time)
            tl = tl.groupby(["period", "sentiment"]).size().reset_index(name="Reviews")
            with st.container(border=True):
                fig = px.bar(tl, x="period", y="Reviews", color="sentiment", color_discrete_map=SENT_COLORS,
                             category_orders={"sentiment": SENT_ORDER}, title="Sentiment over time")
                fig.update_xaxes(title=None)
                fig.update_yaxes(title=None)
                chart(style_fig(fig, 320))
    st.markdown("##### Highest-engagement reviews")
    top_rev = filtered.sort_values("like_count", ascending=False).head(8)[["like_count", "sentiment", "sentiment_score", "review_text"]]
    table(top_rev, column_config={"like_count": "Likes", "sentiment": "Sentiment", "sentiment_score": st.column_config.NumberColumn("Score", format="%.3f"),
                                  "review_text": st.column_config.TextColumn("Review", width="large")})

elif page == "Preprocessing":
    page_header("Text preprocessing", "Cleaning pipeline: lowercase → strip URLs/mentions/emoji → tokenize → remove stop-words → lemmatize / stem.")
    raw_len = filtered["review_text"].str.split().str.len()
    clean_len = filtered["cleaned_text"].str.split().str.len()
    vocab_raw = len({w for t in filtered["review_text"].str.lower() for w in re.findall(r"[a-z']+", t)})
    vocab_clean = len({w for t in filtered["cleaned_text"] for w in t.split()})
    c = st.columns(4, gap="medium")
    with c[0]: kpi("Avg words / review (raw)", f"{raw_len.mean():.1f}", "before cleaning")
    with c[1]: kpi("Avg words / review (clean)", f"{clean_len.mean():.1f}", f"{100 * (1 - clean_len.sum() / max(raw_len.sum(), 1)):.0f}% tokens removed", T["accent"])
    with c[2]: kpi("Vocabulary (raw)", f"{vocab_raw:,}", "unique words")
    with c[3]: kpi("Vocabulary (clean)", f"{vocab_clean:,}", f"{100 * (1 - vocab_clean / max(vocab_raw, 1)):.0f}% smaller", T["accent"])
    st.markdown("")
    with st.container(border=True):
        lens = pd.concat([pd.DataFrame({"words": raw_len, "stage": "Raw"}), pd.DataFrame({"words": clean_len, "stage": "Cleaned"})])
        fig = px.histogram(lens, x="words", color="stage", barmode="overlay", opacity=.75, nbins=30,
                           color_discrete_sequence=[T["muted"], T["accent"]], title="Review length before vs after cleaning")
        fig.update_yaxes(title=None)
        chart(style_fig(fig, 320))
    st.markdown("##### Before → after")
    cols = [c for c in ["review_text", "tokens", "lemmas", "stems", "cleaned_text"] if c in filtered.columns]
    table(filtered[cols].head(25), height=420)

elif page == "Bag of Words":
    page_header("Bag of Words", "Raw term counts from the cleaned text of the filtered reviews.")
    if not texts:
        st.info("No usable text in this subset.")
    else:
        mode = st.segmented_control("Terms", ["Unigrams", "Bigrams", "Both"], default="Unigrams", key="bow_mode") \
            if hasattr(st, "segmented_control") else st.radio("Terms", ["Unigrams", "Bigrams", "Both"], horizontal=True, key="bow_mode_r")
        mode = mode or "Unigrams"
        ng = {"Unigrams": (1, 1), "Bigrams": (2, 2), "Both": (1, 2)}[mode]
        terms, mat = vectorize(texts, "bow", ng)
        freq = pd.DataFrame({"term": terms, "frequency": np.asarray(mat.sum(axis=0)).ravel(),
                             "reviews": np.asarray((mat > 0).sum(axis=0)).ravel()}).sort_values(["frequency", "term"], ascending=[False, True])
        c = st.columns(3, gap="medium")
        with c[0]: kpi("Matrix shape", f"{mat.shape[0]:,} × {mat.shape[1]:,}", "reviews × terms")
        with c[1]: kpi("Sparsity", f"{100 * (1 - mat.nnz / (mat.shape[0] * mat.shape[1])):.1f}%", "share of zero cells")
        with c[2]: kpi("Total tokens", f"{int(mat.sum()):,}", "after cleaning")
        st.markdown("")
        top_n = st.select_slider("Terms to show", [10, 15, 20, 30, 40], value=20, key="bow_n")
        with st.container(border=True):
            chart(hbar(freq.head(top_n), "frequency", "term", "Most frequent terms", color="frequency"))
        table(freq.head(top_n))
        download_csv(freq, "bow_frequencies.csv", "Download BoW CSV")

elif page == "TF-IDF":
    page_header("TF-IDF", "Term distinctiveness (sublinear TF × IDF, unigrams + bigrams). High scores mean frequent in few reviews, not 'positive'.")
    if not texts:
        st.info("No usable text in this subset.")
    else:
        terms, mat = vectorize(texts, "tfidf", (1, 2))
        sc = pd.DataFrame({"term": terms, "mean_tfidf": np.asarray(mat.mean(axis=0)).ravel(),
                           "max_tfidf": np.asarray(mat.max(axis=0).todense()).ravel(),
                           "reviews": np.asarray((mat > 0).sum(axis=0)).ravel()}).sort_values("mean_tfidf", ascending=False)
        c = st.columns(3, gap="medium")
        with c[0]: kpi("Feature space", f"{mat.shape[1]:,}", "terms kept (max 3,000)")
        with c[1]: kpi("Matrix shape", f"{mat.shape[0]:,} × {mat.shape[1]:,}", "reviews × terms")
        with c[2]: kpi("Top term", html.escape(str(sc.iloc[0]["term"])), f"mean TF-IDF {sc.iloc[0]['mean_tfidf']:.4f}", T["accent"])
        st.markdown("")
        top_n = st.select_slider("Terms to show", [10, 15, 20, 30, 40], value=20, key="tf_n")
        l, r = st.columns([1, 1], gap="large")
        with l, st.container(border=True):
            chart(hbar(sc.head(top_n), "mean_tfidf", "term", "Mean TF-IDF by term", color="mean_tfidf"))
        with r, st.container(border=True):
            top_terms = list(sc.head(12)["term"])
            idx = [list(terms).index(t) for t in top_terms]
            sub = mat[:, idx]
            rows = np.argsort(-np.asarray(sub.sum(axis=1)).ravel())[:15]
            heat = sub[rows].toarray()
            fig = go.Figure(go.Heatmap(z=heat, x=top_terms, y=[f"Review {i + 1}" for i in range(len(rows))],
                                       colorscale=[[0, T["panel"]], [1, T["accent"]]], showscale=False,
                                       hovertemplate="%{y}<br>%{x}: %{z:.3f}<extra></extra>"))
            fig.update_layout(title="TF-IDF weights · top terms × top reviews")
            fig.update_yaxes(autorange="reversed", showgrid=False)
            fig.update_xaxes(tickangle=-40, showgrid=False)
            chart(style_fig(fig, max(340, 26 * top_n + 80)))
        table(sc.head(top_n), column_config={"mean_tfidf": st.column_config.NumberColumn(format="%.4f"), "max_tfidf": st.column_config.NumberColumn(format="%.4f")})
        download_csv(sc, "tfidf_scores.csv", "Download TF-IDF CSV")

elif page == "Dimensionality Reduction":
    page_header("Dimensionality reduction", "Truncated SVD (Latent Semantic Analysis) on the TF-IDF matrix.")
    proj = svd_projection(texts) if len(texts) >= 4 else None
    if proj is None:
        st.info("At least four usable reviews with enough distinct terms are needed.")
    else:
        coords, cum, evr2, shape, top = proj
        n90 = int(np.searchsorted(cum, .90) + 1) if cum[-1] >= .90 else len(cum)
        c = st.columns(3, gap="medium")
        with c[0]: kpi("Original dimensions", f"{shape[1]:,}", "TF-IDF features")
        with c[1]: kpi("Components for 90% variance", f"{n90}" + ("" if cum[-1] >= .90 else "+"), f"{100 * cum[min(n90, len(cum)) - 1]:.1f}% retained", T["accent"])
        with c[2]: kpi("2-D explained variance", f"{100 * evr2.sum():.1f}%", "components 1 + 2")
        st.markdown("")
        l, r = st.columns(2, gap="large")
        with l, st.container(border=True):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(1, len(cum) + 1), y=cum, mode="lines+markers", line=dict(color=T["accent"], width=3),
                                     marker=dict(size=5), name="Cumulative variance"))
            fig.add_hline(y=.9, line_dash="dash", line_color=T["muted"], annotation_text="90%", annotation_font_color=T["muted"])
            fig.update_layout(title="Cumulative explained variance", showlegend=False)
            fig.update_xaxes(title="Components")
            fig.update_yaxes(tickformat=".0%")
            chart(style_fig(fig, 420))
        with r, st.container(border=True):
            plot = valid.assign(**{"Component 1": coords[:, 0], "Component 2": coords[:, 1]})
            plot["preview"] = plot["review_text"].str.slice(0, 110)
            color_by = st.radio("Colour by", ["Sentiment", "Likes"], horizontal=True, key="svd_color", label_visibility="collapsed")
            kwargs = dict(x="Component 1", y="Component 2", opacity=.8, hover_data={"preview": True, "sentiment_score": ":.2f"},
                          title="2-D LSA projection", render_mode="webgl" if len(plot) > 3000 else "auto")
            if color_by == "Sentiment":
                fig = px.scatter(plot, color="sentiment", color_discrete_map=SENT_COLORS, category_orders={"sentiment": SENT_ORDER}, **kwargs)
            else:
                fig = px.scatter(plot, color="like_count", color_continuous_scale=[[0, T["accent_lo"]], [1, T["accent"]]], **kwargs)
            fig.update_traces(marker=dict(size=7))
            chart(style_fig(fig, 380))
        st.markdown("##### What the first two components represent")
        comp = pd.DataFrame({f"Component {i + 1} · {pole}": [", ".join(v)] for i in range(2) for pole, v in top[i].items()}).T.reset_index()
        comp.columns = ["Component / pole", "Highest-loading terms"]
        table(comp)

elif page == "Sentiment":
    page_header("Sentiment analysis", "Review-level polarity from VADER: ≥ +0.05 Positive, ≤ −0.05 Negative, otherwise Neutral.")
    counts = filtered["sentiment"].value_counts().reindex(SENT_ORDER, fill_value=0).rename_axis("Sentiment").reset_index(name="Reviews")
    l, r = st.columns(2, gap="large")
    with l, st.container(border=True):
        fig = px.bar(counts, x="Sentiment", y="Reviews", color="Sentiment", color_discrete_map=SENT_COLORS, text="Reviews", title="Reviews by sentiment")
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title=None, showgrid=False)
        fig.update_yaxes(title=None)
        chart(style_fig(fig, 360))
    with r, st.container(border=True):
        bins = pd.cut(filtered["sentiment_score"], [-1.01, -.6, -.05, .05, .6, 1.01],
                      labels=["Strong negative", "Mild negative", "Neutral", "Mild positive", "Strong positive"])
        inten = bins.value_counts().reindex(bins.cat.categories).rename_axis("Intensity").reset_index(name="Reviews")
        fig = px.bar(inten, x="Intensity", y="Reviews", title="Sentiment intensity", color="Intensity",
                     color_discrete_sequence=[T["neg"], T["neg"], T["neu"], T["pos"], T["pos"]])
        fig.update_traces(marker_opacity=.9)
        for i, op in enumerate([1, .55, 1, .55, 1]):
            fig.data[i].marker.opacity = op
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title=None, showgrid=False)
        fig.update_yaxes(title=None)
        chart(style_fig(fig, 360))
    if filtered["like_count"].max() > 0:
        with st.container(border=True):
            fig = px.box(filtered, x="sentiment", y="like_count", color="sentiment", color_discrete_map=SENT_COLORS,
                         category_orders={"sentiment": SENT_ORDER}, points="outliers", title="Likes by sentiment")
            fig.update_layout(showlegend=False)
            fig.update_xaxes(title=None)
            fig.update_yaxes(title=None)
            chart(style_fig(fig, 320))
    counts["Share (%)"] = (100 * counts["Reviews"] / n).round(1)
    table(counts)
    st.caption("VADER scores whole reviews, not individual words, and can misread sarcasm, slang and mixed opinions. "
               "Validate a hand-labelled sample before quoting accuracy.")
    with st.expander("Borderline predictions worth manual review"):
        bl = filtered.assign(distance=(filtered["sentiment_score"].abs() - .05).abs()).sort_values("distance").head(20)
        table(bl[["review_text", "sentiment_score", "sentiment"]],
              column_config={"review_text": st.column_config.TextColumn("Review", width="large")})

elif page == "Tag Intelligence":
    page_header("Tag intelligence", "Tags are generated in three stages: rank → refine → enrich.")
    ranked, refined, enriched = build_tag_stages(filtered[["cleaned_text", "sentiment", "sentiment_score"]])
    if ranked.empty or refined.empty:
        st.info("Not enough usable text in this subset to generate tags.")
    else:
        stage = st.radio("Stage", ["1 · Ranked", "2 · Refined", "3 · Enriched"], horizontal=True, key="tag_stage")
        desc = {"1": "All candidate terms scored by 55% normalised frequency + 45% mean TF-IDF.",
                "2": "Junk removed, stem-duplicates merged, unigrams absorbed by stronger bigrams dropped.",
                "3": "Adds category, related terms, synonyms and the sentiment profile of reviews using each tag."}
        st.caption(desc[stage[0]])
        c = st.columns(3, gap="medium")
        with c[0]: kpi("Candidates", f"{len(ranked):,}", "stage 1")
        with c[1]: kpi("After refinement", f"{len(refined):,}", f"{100 * (1 - len(refined) / max(len(ranked), 1)):.0f}% pruned", T["accent"])
        with c[2]: kpi("Enriched tags", f"{len(enriched):,}", "with sentiment profile", T["accent"])
        st.markdown("")
        if stage[0] == "1":
            data = ranked.head(40).rename(columns={"tag": "Tag"})
        elif stage[0] == "2":
            data = refined.assign(Tag=refined["tag"]) if "Tag" not in refined else refined
        else:
            data = enriched
        weights = {str(t): float(w) for t, w in zip(data["Tag"], data["rank_score"]) if w > 0}
        dom = dict(zip(enriched["Tag"], enriched["Dominant"])) if stage[0] == "3" else {}
        pal = {"Positive": T["pos"], "Neutral": T["neu"], "Negative": T["neg"]}
        accents = T["colorway"]

        def color_func(word, **_):
            if dom:
                return pal.get(dom.get(word, ""), T["muted"])
            return accents[hash(word) % 3]

        with st.container(border=True):
            cloud = WordCloud(width=1400, height=460, background_color=None, mode="RGBA", color_func=color_func,
                              collocations=False, prefer_horizontal=.95, max_words=60).generate_from_frequencies(weights)
            image(cloud.to_array(), "Tag cloud · size = rank score" + (" · colour = dominant sentiment (green positive, amber neutral, red negative)" if dom else ""))
        l, r = st.columns([1, 1], gap="large")
        with l, st.container(border=True):
            chart(hbar(data.head(20), "rank_score", "Tag", "Top tags by rank score", color="rank_score"))
        with r, st.container(border=True):
            if stage[0] == "3":
                e = enriched.sort_values("Reviews", ascending=False).head(15)
                fig = go.Figure()
                for s in SENT_ORDER:
                    fig.add_trace(go.Bar(name=s, y=e["Tag"], x=e[s], orientation="h", marker_color=SENT_COLORS[s]))
                fig.update_layout(barmode="stack", title="Sentiment of reviews mentioning each tag")
                fig.update_yaxes(autorange="reversed", showgrid=False)
                chart(style_fig(fig, max(340, 26 * len(e) + 80)))
            else:
                cat = ranked.head(40).assign(Category=lambda x: x["term"].map(categorize))
                cat = cat.groupby("Category", as_index=False)["rank_score"].sum()
                chart(hbar(cat, "rank_score", "Category", "Theme weight by category (top 40 candidates)", color="rank_score"))
        show_cols = {"1": ["rank", "tag", "frequency", "reviews", "mean_tfidf", "rank_score"],
                     "2": ["rank", "tag", "frequency", "reviews", "mean_tfidf", "rank_score"],
                     "3": ["Tag", "Category", "Reviews", "Related terms", "Synonyms", "Positive", "Neutral", "Negative", "Net sentiment", "Mean score"]}[stage[0]]
        show = data[[c for c in show_cols if c in data.columns]]
        table(show)
        download_csv(data, f"tags_stage_{stage[0]}.csv", "Download this stage (CSV)")

elif page == "Review Explorer":
    page_header("Review explorer", "Every review matching the current filters.")
    sort_by = st.selectbox("Sort by", ["Most liked", "Most negative", "Most positive", "Newest first"], key="sort")
    view = filtered.copy()
    if sort_by == "Most liked":
        view = view.sort_values("like_count", ascending=False)
    elif sort_by == "Most negative":
        view = view.sort_values("sentiment_score")
    elif sort_by == "Most positive":
        view = view.sort_values("sentiment_score", ascending=False)
    elif has_dates:
        view = view.sort_values("published_at", ascending=False)
    cols = [c for c in ["like_count", "published_at", "sentiment", "sentiment_score", "review_text", "cleaned_text"] if c in view.columns]
    table(view[cols], height=520, column_config={
        "like_count": "Likes", "sentiment": "Sentiment",
        "sentiment_score": st.column_config.ProgressColumn("Score", min_value=-1, max_value=1, format="%.2f"),
        "review_text": st.column_config.TextColumn("Review", width="large"),
        "cleaned_text": st.column_config.TextColumn("Cleaned", width="medium")})
    download_csv(view, "filtered_reviews.csv", "Download matching reviews")
