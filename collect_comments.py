import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

load_dotenv(BASE_DIR / ".env")
API_KEY = os.getenv("YOUTUBE_API_KEY")

VIDEO_ID = "JwAfHEHQKto"  # Replace if you choose a different video
MAX_COMMENTS = 500

if not API_KEY:
    raise ValueError("Missing YOUTUBE_API_KEY. Add it to a .env file.")
if not VIDEO_ID or VIDEO_ID == "PASTE_VIDEO_ID_HERE":
    raise ValueError("Set VIDEO_ID to the chosen video's ID.")

youtube = build("youtube", "v3", developerKey=API_KEY)
comments = []
next_page_token = None

try:
    while len(comments) < MAX_COMMENTS:
        remaining = MAX_COMMENTS - len(comments)
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=VIDEO_ID,
            maxResults=min(100, remaining),
            order="time",
            textFormat="plainText",
            pageToken=next_page_token
        ).execute()

        items = response.get("items", [])
        for item in items:
            top_comment = item["snippet"]["topLevelComment"]
            snippet = top_comment["snippet"]
            text = snippet.get("textDisplay", "").strip()
            if not text:
                continue

            comments.append({
                "comment_id": top_comment["id"],
                "review_text": text,
                "published_at": snippet.get("publishedAt", ""),
                "like_count": int(snippet.get("likeCount", 0) or 0),
                "video_id": VIDEO_ID,
                "source_url": f"https://www.youtube.com/watch?v={VIDEO_ID}"
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token or not items:
            break

    df = pd.DataFrame(comments)
    if df.empty:
        raise RuntimeError(
            "No comments returned. Check that comments are enabled and the video ID is correct."
        )

    df = df.drop_duplicates(subset=["comment_id"])
    df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["like_count", "published_at"], ascending=[False, False])

    output = DATA_DIR / "reviews.csv"
    df.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"Collection complete: {len(df)} unique top-level comments")
    print(f"Saved: {output}")
    print("\nTop 10 comments by likes:")
    print(df[["like_count", "review_text"]].head(10).to_string(index=False))

except HttpError as exc:
    print("YouTube API request failed.")
    print(exc)
    print("Check API enablement, key restrictions, quota, video ID, and comment availability.")
