#!/usr/bin/env python3
"""
Public Safety Pulse — Safety Sentiment NLP Pipeline
====================================================
Analyzes Yelp reviews, Reddit posts, and other text for safety-related sentiment.
Creates a "Review Safety Sentiment Score" by block/neighborhood.

Methods:
  1. Keyword extraction (safety-related terms)
  2. Sentiment scoring (TextBlob / spaCy)
  3. Geocoded aggregation to block or neighborhood level

Usage:
  python sentiment_analysis.py --source yelp
  python sentiment_analysis.py --source reddit
  python sentiment_analysis.py --all
"""

import re
import json
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd

# Try importing NLP libraries
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False
    print("⚠ textblob not installed. Run: pip install textblob")

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# ─── Safety Keyword Lexicon ─────────────────────────────────────────────────

# Weighted keywords: positive score = feels safe, negative = feels unsafe
SAFETY_LEXICON = {
    # Strongly negative (-3)
    "mugged": -3, "robbed": -3, "assaulted": -3, "attacked": -3,
    "stabbed": -3, "shot": -3, "dangerous": -3, "terrifying": -3,
    
    # Moderately negative (-2)
    "unsafe": -2, "sketchy": -2, "scary": -2, "threatening": -2,
    "crime": -2, "theft": -2, "robbery": -2, "break-in": -2,
    "breakin": -2, "carjack": -2, "needles": -2, "feces": -2,
    "aggressive": -2, "harassed": -2, "vandalized": -2,
    
    # Mildly negative (-1)
    "dirty": -1, "filthy": -1, "gross": -1, "smelly": -1,
    "homeless": -1, "tent": -1, "encampment": -1, "panhandling": -1,
    "sketchy": -1, "shady": -1, "rundown": -1, "abandoned": -1,
    "graffiti": -1, "trash": -1, "littered": -1, "dark": -1,
    "uncomfortable": -1, "uneasy": -1, "nervous": -1, "wary": -1,
    "avoid": -1, "wouldn't walk": -1, "don't go": -1,
    
    # Mildly positive (+1)
    "okay": 1, "fine": 1, "decent": 1, "improving": 1,
    "better": 1, "cleaned up": 1, "not bad": 1,
    
    # Moderately positive (+2)
    "safe": 2, "clean": 2, "comfortable": 2, "pleasant": 2,
    "friendly": 2, "welcoming": 2, "well-lit": 2, "vibrant": 2,
    "lively": 2, "family-friendly": 2, "walkable": 2,
    
    # Strongly positive (+3)
    "very safe": 3, "perfectly safe": 3, "love this area": 3,
    "feel comfortable": 3, "great neighborhood": 3,
}

# Broader safety topic detection (boolean: is this text about safety?)
SAFETY_TOPIC_PATTERNS = [
    r'\b(safe|safety|unsafe|danger|crime|theft|rob|assault|attack)\b',
    r'\b(homeless|tent|encampment|needle|feces|human waste)\b',
    r'\b(sketch|scar|creep|threat|harass|mug|vandal)\b',
    r'\b(clean|dirty|filth|trash|graffiti|litter)\b',
    r'\b(police|cop|sfpd|patrol|security|guard)\b',
    r'\b(walk|walking|stroll|pedestrian)\s.*(safe|night|dark|afraid|comfortable)',
    r'\b(avoid|wouldn\'t go|don\'t go|stay away)\b',
    r'\b(car break|smash|window broken|catalytic)\b',
    r'\b(well.?lit|dark|lighting|streetlight)\b',
    r'\b(comfortable|uncomfortable|uneasy|nervous)\b',
]


def is_safety_related(text):
    """Check if text contains safety-related content."""
    if not text:
        return False
    text_lower = text.lower()
    for pattern in SAFETY_TOPIC_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def compute_safety_score(text):
    """
    Compute a safety sentiment score for a text.
    Returns dict with:
      - safety_score: weighted keyword score (-10 to +10 range)
      - general_sentiment: TextBlob polarity (-1 to +1)
      - safety_keywords_found: list of matched keywords
      - is_safety_related: boolean
    """
    if not text:
        return {
            "safety_score": 0,
            "general_sentiment": 0,
            "safety_keywords_found": [],
            "is_safety_related": False,
        }
    
    text_lower = text.lower()
    
    # Keyword matching
    found_keywords = []
    total_score = 0
    
    for keyword, weight in SAFETY_LEXICON.items():
        # Use word boundary matching for single words, substring for phrases
        if " " in keyword:
            if keyword in text_lower:
                found_keywords.append(keyword)
                total_score += weight
        else:
            if re.search(rf'\b{re.escape(keyword)}\b', text_lower):
                found_keywords.append(keyword)
                total_score += weight
    
    # General sentiment via TextBlob
    general_sentiment = 0
    if HAS_TEXTBLOB:
        blob = TextBlob(text)
        general_sentiment = blob.sentiment.polarity
    
    # Is safety-related?
    related = is_safety_related(text) or len(found_keywords) > 0
    
    return {
        "safety_score": total_score,
        "general_sentiment": round(general_sentiment, 3),
        "safety_keywords_found": found_keywords,
        "is_safety_related": related,
        "keyword_count": len(found_keywords),
    }


def analyze_yelp_reviews():
    """Analyze Yelp reviews for safety sentiment."""
    print("\n" + "="*60)
    print("ANALYZING: Yelp Review Safety Sentiment")
    print("="*60)
    
    path = RAW_DIR / "layer3" / "yelp_reviews.parquet"
    if not path.exists():
        print("  ⚠ Yelp reviews not found. Run pull_all_data.py --source yelp first.")
        return None
    
    df = pd.read_parquet(path)
    print(f"  Loaded {len(df)} reviews")
    
    # Score each review
    scores = df["text"].apply(compute_safety_score)
    score_df = pd.DataFrame(scores.tolist())
    df = pd.concat([df, score_df], axis=1)
    
    # Filter to safety-related reviews
    safety_reviews = df[df["is_safety_related"]].copy()
    print(f"  Safety-related reviews: {len(safety_reviews)} ({len(safety_reviews)/len(df)*100:.1f}%)")
    
    # Aggregate by CBD area
    if "cbd" in df.columns:
        cbd_scores = (
            safety_reviews.groupby("cbd")
            .agg(
                avg_safety_score=("safety_score", "mean"),
                median_safety_score=("safety_score", "median"),
                review_count=("safety_score", "count"),
                avg_general_sentiment=("general_sentiment", "mean"),
            )
            .reset_index()
        )
        print("\n  Safety sentiment by CBD area:")
        print(cbd_scores.to_string(index=False))
    
    # Save
    df.to_parquet(PROCESSED_DIR / "yelp_safety_sentiment.parquet", index=False)
    if len(safety_reviews) > 0:
        safety_reviews.to_parquet(PROCESSED_DIR / "yelp_safety_reviews_only.parquet", index=False)
    
    print(f"\n  ✓ Saved scored reviews → processed/yelp_safety_sentiment.parquet")
    
    # Top negative keywords found
    all_keywords = []
    for kw_list in safety_reviews["safety_keywords_found"]:
        all_keywords.extend(kw_list)
    
    if all_keywords:
        print("\n  Top safety keywords in reviews:")
        for kw, count in Counter(all_keywords).most_common(15):
            print(f"    {kw}: {count}")
    
    return df


def analyze_reddit_posts():
    """Analyze Reddit posts for safety sentiment."""
    print("\n" + "="*60)
    print("ANALYZING: Reddit Safety Sentiment")
    print("="*60)
    
    path = RAW_DIR / "layer3" / "reddit_safety_posts.parquet"
    if not path.exists():
        print("  ⚠ Reddit data not found. Run pull_all_data.py --source reddit first.")
        return None
    
    df = pd.read_parquet(path)
    print(f"  Loaded {len(df)} posts")
    
    # Combine title + body for analysis
    df["full_text"] = df["title"].fillna("") + " " + df["selftext"].fillna("")
    
    # Score each post
    scores = df["full_text"].apply(compute_safety_score)
    score_df = pd.DataFrame(scores.tolist())
    df = pd.concat([df, score_df], axis=1)
    
    # Filter to genuinely safety-related
    safety_posts = df[df["is_safety_related"]].copy()
    print(f"  Safety-related posts: {len(safety_posts)} ({len(safety_posts)/len(df)*100:.1f}%)")
    
    # Time series of safety sentiment
    if "created_datetime" in df.columns:
        df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce")
        df["year_month"] = df["created_datetime"].dt.to_period("M").astype(str)
        
        monthly = (
            safety_posts.groupby("year_month")
            .agg(
                avg_safety_score=("safety_score", "mean"),
                post_count=("safety_score", "count"),
                avg_engagement=("score", "mean"),
            )
            .reset_index()
        )
        print("\n  Monthly safety sentiment trend:")
        print(monthly.tail(6).to_string(index=False))
    
    # Save
    df.to_parquet(PROCESSED_DIR / "reddit_safety_sentiment.parquet", index=False)
    print(f"\n  ✓ Saved scored posts → processed/reddit_safety_sentiment.parquet")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Safety Sentiment NLP Pipeline")
    parser.add_argument("--source", choices=["yelp", "reddit", "all"], default="all")
    args = parser.parse_args()
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.source in ["yelp", "all"]:
        analyze_yelp_reviews()
    
    if args.source in ["reddit", "all"]:
        analyze_reddit_posts()
    
    print("\n✓ Sentiment analysis complete.")


if __name__ == "__main__":
    main()
