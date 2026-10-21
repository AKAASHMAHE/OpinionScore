# Opinion Mining and Tag-Based Categorization

A Python case-study project with data collection, text preprocessing, BoW, TF-IDF, SVD/LSA, VADER sentiment, ranked tags, and a Streamlit dashboard.

## Important data-source note
Before analyzing comments from any platform, check the platform's current terms, API policies, retention rules, and permission requirements. Publicly viewable content is not automatically permitted for every kind of collection, derived analysis, or redistribution. Do not publicly share raw comments or API keys unless permitted. If platform terms do not allow the intended analysis, use consent-based feedback or a dataset with a license that permits text mining and derived sentiment/tag outputs.

## Setup
1. Install Python 3.10+.
2. Open a terminal in this project folder.
3. Install dependencies:
   `pip install -r requirements.txt`
4. For API collection, copy `.env.example` to `.env` and enter your API key.
5. Configure the video ID in `collect_comments.py` only if your intended use is permitted.
6. Collect permitted data:
   `python collect_comments.py`
7. Run analysis:
   `python analysis.py`
8. Launch dashboard:
   `streamlit run app.py`

## Input CSV format
Place a CSV at `data/reviews.csv` with at least a `review_text` column. Optional columns:
- `comment_id`
- `published_at`
- `like_count`
- `video_id`
- `source_url`

The collector writes top-level comment text and metadata. It does not collect all replies.

## Outputs
- `data/processed_reviews.csv`
- `outputs/bow_frequency.csv`
- `outputs/bow_matrix_preview.csv`
- `outputs/tfidf_ranked_terms.csv`
- `outputs/ranked_refined_tags.csv`
- `outputs/tag_sentiment.csv`
- `outputs/summary.json`

## Method notes
- BoW and TF-IDF use unigrams and bigrams.
- VADER is lexicon/rule-based; it is not a trained classifier and should be evaluated against manually labeled examples if you report accuracy.
- SVD/LSA gives a two-dimensional projection for visualization. Clusters are exploratory, not proof of sentiment separation.
- Tag enrichment uses a transparent keyword-to-category mapping that should be refined for the chosen topic.
- Like counts are engagement signals, not measures of truth or sentiment accuracy.
