from pathlib import Path
import re
import json
import warnings

import numpy as np
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

INPUT_FILE = DATA_DIR / "reviews.csv"
PROCESSED_FILE = DATA_DIR / "processed_reviews.csv"

# VADER lexicon is needed for sentiment scoring.
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

# Preserve negation words because they change sentiment.
STOP_WORDS = set(ENGLISH_STOP_WORDS) - {"not", "no", "nor", "never"}
TAG_GROUPS = {
    "Performance": {"slow", "lag", "laggy", "crash", "crashes", "crashing", "freeze", "freezes", "bug", "bugs", "error", "errors"},
    "Usability": {"easy", "simple", "confusing", "difficult", "interface", "design", "user", "navigation", "experience"},
    "Quality": {"quality", "excellent", "poor", "great", "bad", "good", "amazing", "worst", "best"},
    "Cost / Value": {"price", "cost", "expensive", "cheap", "value", "payment", "paid", "free"},
    "Support": {"support", "service", "help", "customer", "response", "team"},
    "Features": {"feature", "features", "option", "options", "update", "updates", "function", "functions"},
    "Reliability": {"reliable", "reliability", "stable", "unstable", "working", "works", "issue", "issues"},
    "General Feedback": set()
}

def clean_text(text):
    """Normalize text while preserving negation words and basic word tokens."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[@#]\w+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)

def safe_write_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")

def run_analysis():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}. Run collect_comments.py first, "
            "or place a permitted feedback CSV there with a review_text column."
        )

    df = pd.read_csv(INPUT_FILE)
    if "review_text" not in df.columns:
        raise ValueError("Input CSV must contain a 'review_text' column.")

    df = df.dropna(subset=["review_text"]).copy()
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[df["review_text"] != ""]
    dedupe_col = "comment_id" if "comment_id" in df.columns else "review_text"
    df = df.drop_duplicates(subset=[dedupe_col]).reset_index(drop=True)

    if "like_count" not in df.columns:
        df["like_count"] = 0
    df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0).astype(int)

    df["cleaned_text"] = df["review_text"].apply(clean_text)
    df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("At least two non-empty reviews are needed for analysis.")

    # Bag of Words
    bow_vectorizer = CountVectorizer(max_features=1500, ngram_range=(1, 2), min_df=1)
    bow = bow_vectorizer.fit_transform(df["cleaned_text"])
    bow_terms = bow_vectorizer.get_feature_names_out()
    bow_freq = np.asarray(bow.sum(axis=0)).ravel()
    bow_df = pd.DataFrame({"term": bow_terms, "frequency": bow_freq})
    bow_df = bow_df.sort_values("frequency", ascending=False).reset_index(drop=True)
    safe_write_csv(bow_df, OUTPUT_DIR / "bow_frequency.csv")

    # Store a small, readable BoW matrix for dashboard/report.
    bow_matrix = pd.DataFrame(
        bow[:min(30, len(df)), :min(30, len(bow_terms))].toarray(),
        columns=bow_terms[:min(30, len(bow_terms))]
    )
    safe_write_csv(bow_matrix, OUTPUT_DIR / "bow_matrix_preview.csv")

    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    tfidf = tfidf_vectorizer.fit_transform(df["cleaned_text"])
    tfidf_terms = tfidf_vectorizer.get_feature_names_out()
    tfidf_mean = np.asarray(tfidf.mean(axis=0)).ravel()
    tfidf_df = pd.DataFrame({"term": tfidf_terms, "mean_tfidf": tfidf_mean})
    tfidf_df = tfidf_df.sort_values("mean_tfidf", ascending=False).reset_index(drop=True)
    safe_write_csv(tfidf_df, OUTPUT_DIR / "tfidf_ranked_terms.csv")

    # Sentiment scores: classify original text so negation/punctuation/emojis aren't removed first.
    analyzer = SentimentIntensityAnalyzer()
    sentiment_results = df["review_text"].apply(analyzer.polarity_scores)
    df["sentiment_score"] = sentiment_results.apply(lambda x: x["compound"])
    df["positive_score"] = sentiment_results.apply(lambda x: x["pos"])
    df["negative_score"] = sentiment_results.apply(lambda x: x["neg"])
    df["neutral_score"] = sentiment_results.apply(lambda x: x["neu"])

    def label(score):
        if score >= 0.05:
            return "Positive"
        if score <= -0.05:
            return "Negative"
        return "Neutral"

    df["sentiment"] = df["sentiment_score"].apply(label)

    # SVD/LSA coordinates. With only two or more rows/features, 2 components are possible.
    if tfidf.shape[0] >= 2 and tfidf.shape[1] >= 2:
        svd = TruncatedSVD(n_components=2, random_state=42)
        coords = svd.fit_transform(tfidf)
        df["component_1"] = coords[:, 0]
        df["component_2"] = coords[:, 1]
        explained = float(svd.explained_variance_ratio_.sum())
    else:
        df["component_1"] = 0.0
        df["component_2"] = 0.0
        explained = 0.0

    # Tags: rank candidate terms using corpus frequency and mean TF-IDF.
    # Refinement uses normalization and phrase terms from the vectorizers.
    tags = bow_df.merge(tfidf_df, on="term", how="left").fillna({"mean_tfidf": 0})
    max_freq = max(float(tags["frequency"].max()), 1.0)
    max_tfidf = max(float(tags["mean_tfidf"].max()), 1e-9)
    tags["frequency_norm"] = tags["frequency"] / max_freq
    tags["tfidf_norm"] = tags["mean_tfidf"] / max_tfidf
    tags["rank_score"] = 0.55 * tags["frequency_norm"] + 0.45 * tags["tfidf_norm"]

    # Remove generic/noisy one-word tags where possible; retain phrases and meaningful terms.
    generic = {"thing", "things", "someone", "something", "today", "video", "youtube", "channel"}
    tags = tags[~tags["term"].isin(generic)].copy()
    tags["tag"] = tags["term"].str.replace("_", " ", regex=False).str.title()

    def categorize(term):
        words = set(term.split())
        for category, vocabulary in TAG_GROUPS.items():
            if category != "General Feedback" and words.intersection(vocabulary):
                return category
        return "General Feedback"

    tags["category"] = tags["term"].apply(categorize)
    tags = tags.sort_values(["rank_score", "frequency"], ascending=False).reset_index(drop=True)
    tags["rank"] = np.arange(1, len(tags) + 1)
    safe_write_csv(tags, OUTPUT_DIR / "ranked_refined_tags.csv")

    # Associate each tag with sentiment of reviews containing it.
    tag_sentiments = []
    for _, row in tags.head(100).iterrows():
        term = str(row["term"])
        contains = df["cleaned_text"].str.contains(
            rf"(?<!\w){re.escape(term)}(?!\w)", regex=True, na=False
        )
        matched = df.loc[contains]
        if len(matched) == 0:
            continue
        counts = matched["sentiment"].value_counts()
        tag_sentiments.append({
            "tag": row["tag"],
            "category": row["category"],
            "review_count": len(matched),
            "positive": int(counts.get("Positive", 0)),
            "negative": int(counts.get("Negative", 0)),
            "neutral": int(counts.get("Neutral", 0)),
            "mean_sentiment": float(matched["sentiment_score"].mean()),
            "rank_score": float(row["rank_score"])
        })
    tag_sentiment_df = pd.DataFrame(tag_sentiments)
    safe_write_csv(tag_sentiment_df, OUTPUT_DIR / "tag_sentiment.csv")

    # Likes rank is kept separate from sentiment.
    df = df.sort_values(["like_count", "published_at"] if "published_at" in df.columns else ["like_count"],
                        ascending=False).reset_index(drop=True)
    df["likes_rank"] = np.arange(1, len(df) + 1)
    safe_write_csv(df, PROCESSED_FILE)

    sentiment_counts = df["sentiment"].value_counts().to_dict()
    summary = {
        "total_reviews": int(len(df)),
        "positive": int(sentiment_counts.get("Positive", 0)),
        "negative": int(sentiment_counts.get("Negative", 0)),
        "neutral": int(sentiment_counts.get("Neutral", 0)),
        "mean_sentiment_score": float(df["sentiment_score"].mean()),
        "median_likes": float(df["like_count"].median()),
        "max_likes": int(df["like_count"].max()),
        "svd_explained_variance_ratio_sum": explained
    }
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Analysis complete.")
    print(f"Reviews analyzed: {len(df)}")
    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())
    print("\nTop terms by frequency:")
    print(bow_df.head(10).to_string(index=False))
    print("\nTop terms by mean TF-IDF:")
    print(tfidf_df.head(10).to_string(index=False))
    print("\nTop refined tags:")
    print(tags[["rank", "tag", "category", "frequency", "rank_score"]].head(15).to_string(index=False))
    print(f"\nSaved processed data to {PROCESSED_FILE}")
    print(f"Saved charts' data to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_analysis()
