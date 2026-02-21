#!/usr/bin/env python3
"""
Public Safety Pulse — Master Data Pipeline
============================================
Pulls all freely available safety and experience data for San Francisco
to build the MVP "Safety Perception Index" dashboard.

Data Sources (by layer):
  Layer 1 — Hard Incident Data: SFPD incidents, Fire calls, Traffic crashes
  Layer 2 — Condition/Environment: 311 cases, tent counts, ambassador data
  Layer 3 — Behavioral/Sentiment Proxies: BART ridership, Yelp reviews, Reddit

Usage:
  python pull_all_data.py --all              # Pull everything
  python pull_all_data.py --layer 1          # Pull only Layer 1
  python pull_all_data.py --source 311       # Pull only 311 data
  python pull_all_data.py --source sfpd      # Pull only SFPD incidents
  python pull_all_data.py --months 12        # Last 12 months (default)

Requirements:
  pip install pandas requests sodapy geopandas h3 tqdm

Author: City Science Lab San Francisco
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ─── Configuration ──────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# DataSF Socrata endpoints (no auth required for read-only)
SOCRATA_BASE = "https://data.sfgov.org/resource"
SOCRATA_DATASETS = {
    "sfpd_incidents": "wg3w-h783",       # Police Incident Reports (2018–Present)
    "sfpd_incidents_historical": "tmnf-yvry",  # Police Incident Reports (2003–2018)
    "311_cases": "vw6y-z8j6",            # 311 Cases
    "fire_calls": "nuek-vuh3",           # Fire Department Calls for Service
    "traffic_crashes": "ubvf-ztfx",      # Traffic Crashes Resulting in Injury
    "fire_violations": "cf7p-gxqh",      # Fire Violations (bonus)
}

# Safety-relevant 311 categories for disorder index
SAFETY_311_CATEGORIES = [
    "Encampments",
    "Homeless Concerns",
    "Street and Sidewalk Cleaning",
    "Graffiti",
    "Streetlights",
    "Abandoned Vehicle",
    "Damaged Property",
    "SFPD Requests",
    "General Requests",
    "Noise Report",
    "Sewer Issues",
]

# BART ridership endpoint
BART_RIDERSHIP_URL = "https://data.bart.gov/api/views/6bv4-3hif/rows.csv?accessType=DOWNLOAD"

# Downtown SF bounding box (approx) for focused queries
SF_DOWNTOWN_BBOX = {
    "lat_min": 37.770,
    "lat_max": 37.800,
    "lon_min": -122.420,
    "lon_max": -122.390,
}

# Target CBD areas for MVP focus
TARGET_CBDS = {
    "union_square": {"lat": 37.7879, "lon": -122.4074, "radius_m": 500},
    "financial_district": {"lat": 37.7946, "lon": -122.3999, "radius_m": 600},
    "soma_west": {"lat": 37.7785, "lon": -122.4056, "radius_m": 500},
    "mid_market_tenderloin": {"lat": 37.7820, "lon": -122.4120, "radius_m": 500},
    "yerba_buena": {"lat": 37.7850, "lon": -122.4010, "radius_m": 400},
}


# ─── Utility Functions ──────────────────────────────────────────────────────

def ensure_dirs():
    """Create data directories if they don't exist."""
    for subdir in ["raw/layer1", "raw/layer2", "raw/layer3", "processed"]:
        (DATA_DIR / subdir).mkdir(parents=True, exist_ok=True)


def socrata_fetch(dataset_id, params=None, limit=50000, offset=0):
    """
    Fetch data from DataSF Socrata API with pagination.
    No authentication required for read-only access.
    Rate limits are generous (~1000 req/hour unauthenticated).
    """
    url = f"{SOCRATA_BASE}/{dataset_id}.json"
    all_records = []
    
    default_params = {
        "$limit": limit,
        "$offset": offset,
        "$order": ":id",
    }
    if params:
        default_params.update(params)
    
    while True:
        try:
            resp = requests.get(url, params=default_params, timeout=120)
            resp.raise_for_status()
            records = resp.json()
            
            if not records:
                break
                
            all_records.extend(records)
            print(f"  Fetched {len(all_records)} records so far...")
            
            if len(records) < limit:
                break
                
            default_params["$offset"] += limit
            time.sleep(0.5)  # Be respectful
            
        except requests.exceptions.RequestException as e:
            print(f"  Error fetching {dataset_id}: {e}")
            if all_records:
                print(f"  Returning {len(all_records)} records fetched before error")
                break
            raise
    
    return all_records


def get_date_filter(months_back=12):
    """Generate a Socrata $where clause for date filtering."""
    cutoff = (datetime.now() - timedelta(days=months_back * 30)).strftime("%Y-%m-%dT00:00:00")
    return cutoff


# ─── Layer 1: Hard Incident Data ────────────────────────────────────────────

def pull_sfpd_incidents(months=12):
    """
    Source #1: SFPD Police Incident Reports (2018–Present)
    - Individual incidents, geocoded, timestamped
    - Near-daily updates
    - MVP use: Crime heatmaps, time-of-day patterns, category breakdowns
    """
    print("\n" + "="*60)
    print("PULLING: SFPD Police Incident Reports")
    print("="*60)
    
    cutoff = get_date_filter(months)
    params = {
        "$where": f"incident_datetime > '{cutoff}'",
        "$select": (
            "incident_datetime, incident_date, incident_time, incident_year, "
            "incident_day_of_week, report_datetime, row_id, incident_id, "
            "incident_number, cad_number, report_type_code, report_type_description, "
            "incident_code, incident_category, incident_subcategory, "
            "incident_description, resolution, intersection, cnn, "
            "police_district, analysis_neighborhood, supervisor_district, "
            "latitude, longitude, point"
        ),
    }
    
    records = socrata_fetch(SOCRATA_DATASETS["sfpd_incidents"], params)
    df = pd.DataFrame(records)
    
    if len(df) > 0:
        # Type conversions
        df["incident_datetime"] = pd.to_datetime(df["incident_datetime"], errors="coerce")
        for col in ["latitude", "longitude"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        outpath = RAW_DIR / "layer1" / "sfpd_incidents.parquet"
        df.to_parquet(outpath, index=False)
        print(f"  ✓ Saved {len(df)} SFPD incidents → {outpath}")
        
        # Quick stats
        print(f"  Date range: {df['incident_datetime'].min()} to {df['incident_datetime'].max()}")
        print(f"  Top categories: {df['incident_category'].value_counts().head(5).to_dict()}")
    else:
        print("  ⚠ No SFPD records returned")
    
    return df


def pull_fire_calls(months=12):
    """
    Source #4: Fire Department Calls for Service
    - Individual calls, geocoded, timestamped
    - MVP use: Medical emergency hotspots, overdose clustering
    """
    print("\n" + "="*60)
    print("PULLING: Fire Department Calls for Service")
    print("="*60)
    
    cutoff = get_date_filter(months)
    params = {
        "$where": f"call_date > '{cutoff}'",
        "$select": (
            "call_number, unit_id, incident_number, call_type, "
            "call_date, received_dttm, entry_dttm, dispatch_dttm, "
            "response_dttm, on_scene_dttm, transport_dttm, hospital_dttm, "
            "call_final_disposition, available_dttm, address, city, "
            "zipcode_of_incident, battalion, station_area, box, "
            "original_priority, priority, final_priority, "
            "als_unit, call_type_group, number_of_alarms, "
            "unit_type, unit_sequence_in_call_dispatch, "
            "fire_prevention_district, supervisor_district, "
            "neighborhood_district, location, row_id"
        ),
    }
    
    records = socrata_fetch(SOCRATA_DATASETS["fire_calls"], params)
    df = pd.DataFrame(records)
    
    if len(df) > 0:
        df["call_date"] = pd.to_datetime(df["call_date"], errors="coerce")
        
        outpath = RAW_DIR / "layer1" / "fire_calls.parquet"
        df.to_parquet(outpath, index=False)
        print(f"  ✓ Saved {len(df)} fire calls → {outpath}")
        print(f"  Top call types: {df['call_type'].value_counts().head(5).to_dict()}")
    
    return df


def pull_traffic_crashes(months=12):
    """
    Source #5: Vision Zero Traffic Crash Injury Data
    - Individual crashes, geocoded
    - MVP use: Pedestrian/cyclist injury hotspots
    """
    print("\n" + "="*60)
    print("PULLING: Traffic Crash Injury Data")
    print("="*60)
    
    cutoff = get_date_filter(months)
    params = {
        "$where": f"collision_datetime > '{cutoff}'",
    }
    
    records = socrata_fetch(SOCRATA_DATASETS["traffic_crashes"], params)
    df = pd.DataFrame(records)
    
    if len(df) > 0:
        outpath = RAW_DIR / "layer1" / "traffic_crashes.parquet"
        df.to_parquet(outpath, index=False)
        print(f"  ✓ Saved {len(df)} traffic crashes → {outpath}")
    
    return df


# ─── Layer 2: Condition & Environment Data ──────────────────────────────────

def pull_311_cases(months=12):
    """
    Source #7: 311 Cases — THE MVP WORKHORSE
    - Individual cases, geocoded, timestamped
    - Nightly updates
    - MVP use: "Disorder density" index by block and time window
    - Key categories: Encampments, Homeless Concerns, Cleaning, Graffiti,
      Streetlights, Abandoned Vehicles, etc.
    """
    print("\n" + "="*60)
    print("PULLING: 311 Cases (MVP Workhorse)")
    print("="*60)
    
    cutoff = get_date_filter(months)
    
    # Pull ALL 311 cases (filter locally — Socrata rejects long category filter URLs)
    params = {
        "$where": f"opened > '{cutoff}'",
        "$select": (
            "service_request_id, requested_datetime, opened, closed, updated_datetime, "
            "status_description, status_notes, agency_responsible, "
            "service_name, service_subtype, service_details, "
            "address, street, supervisor_district, neighborhood, "
            "point, source, media_url, lat, long"
        ),
    }
    
    records = socrata_fetch(SOCRATA_DATASETS["311_cases"], params)
    df_all = pd.DataFrame(records)
    
    if len(df_all) > 0:
        df_all["opened"] = pd.to_datetime(df_all["opened"], errors="coerce")
        df_all["closed"] = pd.to_datetime(df_all["closed"], errors="coerce")
        for col in ["lat", "long"]:
            if col in df_all.columns:
                df_all[col] = pd.to_numeric(df_all[col], errors="coerce")
        
        # Save ALL 311 cases
        outpath_all = RAW_DIR / "layer2" / "311_cases_all.parquet"
        df_all.to_parquet(outpath_all, index=False)
        print(f"  ✓ Saved {len(df_all)} total 311 cases → {outpath_all}")
        
        # Now filter to safety-relevant categories locally
        df = df_all[df_all["service_name"].isin(SAFETY_311_CATEGORIES)].copy()
        
        outpath = RAW_DIR / "layer2" / "311_cases_safety.parquet"
        df.to_parquet(outpath, index=False)
        print(f"  ✓ Saved {len(df)} safety-relevant 311 cases → {outpath}")
        print(f"  Category breakdown:")
        for cat, count in df["service_name"].value_counts().head(10).items():
            print(f"    {cat}: {count:,}")
    
    return df


# ─── Layer 3: Behavioral & Sentiment Proxies ────────────────────────────────

def pull_bart_ridership():
    """
    Source #14: BART Ridership by Station (Daily)
    - Daily station exits
    - MVP use: Foot traffic / confidence proxy at downtown stations
    """
    print("\n" + "="*60)
    print("PULLING: BART Ridership Data")
    print("="*60)
    
    # BART publishes ridership data — try the open data portal
    try:
        # BART ridership via their open data
        url = "https://data.bart.gov/api/views/6bv4-3hif/rows.csv?accessType=DOWNLOAD"
        resp = requests.get(url, timeout=60)
        
        if resp.status_code == 200:
            outpath = RAW_DIR / "layer3" / "bart_ridership.csv"
            with open(outpath, "wb") as f:
                f.write(resp.content)
            df = pd.read_csv(outpath)
            print(f"  ✓ Saved BART ridership → {outpath}")
            print(f"  Shape: {df.shape}")
            return df
        else:
            print(f"  ⚠ BART data returned status {resp.status_code}")
            print("  Manual download: https://www.bart.gov/about/reports/ridership")
            
    except Exception as e:
        print(f"  ⚠ Could not auto-download BART data: {e}")
        print("  Manual download instructions:")
        print("  1. Go to https://www.bart.gov/about/reports/ridership")
        print("  2. Download monthly ridership Excel files")
        print("  3. Save to data/raw/layer3/bart_ridership/")
    
    return None


def pull_yelp_reviews(api_key=None):
    """
    Source #16: Yelp/Google Reviews (Safety-Related Mentions)
    - Requires Yelp Fusion API key (free: 5,000 calls/day)
    - MVP use: NLP sentiment analysis on safety-related review text
    
    To get an API key:
    1. Go to https://www.yelp.com/developers
    2. Create an app
    3. Copy the API key
    4. Set env var: export YELP_API_KEY=your_key_here
    """
    print("\n" + "="*60)
    print("PULLING: Yelp Reviews (Safety-Related)")
    print("="*60)
    
    api_key = api_key or os.environ.get("YELP_API_KEY")
    
    if not api_key:
        print("  ⚠ No Yelp API key found.")
        print("  Set YELP_API_KEY environment variable or pass --yelp-key")
        print("  Register at: https://www.yelp.com/developers")
        print("  Free tier: 5,000 API calls/day")
        _generate_yelp_placeholder()
        return None
    
    headers = {"Authorization": f"Bearer {api_key}"}
    all_businesses = []
    
    for cbd_name, cbd_info in TARGET_CBDS.items():
        print(f"\n  Searching Yelp for businesses in {cbd_name}...")
        
        params = {
            "latitude": cbd_info["lat"],
            "longitude": cbd_info["lon"],
            "radius": cbd_info["radius_m"],
            "limit": 50,
            "sort_by": "review_count",
        }
        
        try:
            resp = requests.get(
                "https://api.yelp.com/v3/businesses/search",
                headers=headers,
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            
            for biz in data.get("businesses", []):
                all_businesses.append({
                    "cbd": cbd_name,
                    "id": biz["id"],
                    "name": biz["name"],
                    "rating": biz.get("rating"),
                    "review_count": biz.get("review_count"),
                    "latitude": biz["coordinates"]["latitude"],
                    "longitude": biz["coordinates"]["longitude"],
                    "address": " ".join(biz["location"].get("display_address", [])),
                    "categories": ", ".join([c["title"] for c in biz.get("categories", [])]),
                })
            
            print(f"    Found {len(data.get('businesses', []))} businesses")
            time.sleep(0.3)
            
        except Exception as e:
            print(f"    Error searching {cbd_name}: {e}")
    
    if all_businesses:
        df = pd.DataFrame(all_businesses)
        outpath = RAW_DIR / "layer3" / "yelp_businesses.parquet"
        df.to_parquet(outpath, index=False)
        print(f"\n  ✓ Saved {len(df)} Yelp businesses → {outpath}")
        
        # Now pull reviews for top businesses
        print("\n  Pulling reviews for top businesses...")
        all_reviews = []
        
        for _, biz in df.head(100).iterrows():  # Top 100 by review count
            try:
                resp = requests.get(
                    f"https://api.yelp.com/v3/businesses/{biz['id']}/reviews",
                    headers=headers,
                    params={"limit": 50, "sort_by": "newest"},
                    timeout=30,
                )
                resp.raise_for_status()
                reviews = resp.json().get("reviews", [])
                
                for rev in reviews:
                    all_reviews.append({
                        "business_id": biz["id"],
                        "business_name": biz["name"],
                        "cbd": biz["cbd"],
                        "latitude": biz["latitude"],
                        "longitude": biz["longitude"],
                        "rating": rev.get("rating"),
                        "text": rev.get("text", ""),
                        "time_created": rev.get("time_created"),
                    })
                
                time.sleep(0.3)  # Rate limiting
                
            except Exception as e:
                continue
        
        if all_reviews:
            df_reviews = pd.DataFrame(all_reviews)
            outpath = RAW_DIR / "layer3" / "yelp_reviews.parquet"
            df_reviews.to_parquet(outpath, index=False)
            print(f"  ✓ Saved {len(df_reviews)} Yelp reviews → {outpath}")
    
    return df if all_businesses else None


def _generate_yelp_placeholder():
    """Generate a placeholder script for Yelp extraction."""
    script = '''#!/usr/bin/env python3
"""
Yelp Review Extraction Script
Run this after setting your YELP_API_KEY environment variable.

export YELP_API_KEY=your_key_here
python yelp_extract.py
"""
# This is auto-generated — run pull_all_data.py --source yelp --yelp-key YOUR_KEY
print("Set YELP_API_KEY and re-run the main pipeline.")
'''
    outpath = RAW_DIR / "layer3" / "yelp_extract_standalone.py"
    with open(outpath, "w") as f:
        f.write(script)
    print(f"  Created standalone script: {outpath}")


def pull_reddit_sentiment():
    """
    Source #17: Reddit / Social Media Sentiment
    - Subreddits: r/sanfrancisco, r/bayarea, r/AskSF
    - MVP use: Macro sentiment trends, qualitative context
    
    Note: Reddit API now requires OAuth. This uses the public JSON endpoints
    which don't require auth but have stricter rate limits.
    """
    print("\n" + "="*60)
    print("PULLING: Reddit Safety Sentiment")
    print("="*60)
    
    subreddits = ["sanfrancisco", "bayarea", "AskSF"]
    safety_keywords = [
        "safety", "safe", "unsafe", "crime", "theft", "robbery",
        "homeless", "tent", "encampment", "sketchy", "dangerous",
        "clean", "dirty", "needles", "break-in", "car break",
        "mugging", "assault", "police", "sfpd",
    ]
    
    all_posts = []
    
    for sub in subreddits:
        print(f"\n  Searching r/{sub}...")
        
        for keyword in safety_keywords[:5]:  # Start with top keywords
            try:
                url = f"https://www.reddit.com/r/{sub}/search.json"
                params = {
                    "q": keyword,
                    "sort": "new",
                    "t": "year",
                    "limit": 25,
                    "restrict_sr": "true",
                }
                headers = {"User-Agent": "PSP-MVP-Research/1.0"}
                
                resp = requests.get(url, params=params, headers=headers, timeout=30)
                
                if resp.status_code == 200:
                    data = resp.json()
                    posts = data.get("data", {}).get("children", [])
                    
                    for post in posts:
                        p = post["data"]
                        all_posts.append({
                            "subreddit": sub,
                            "keyword": keyword,
                            "title": p.get("title", ""),
                            "selftext": p.get("selftext", "")[:500],  # Truncate
                            "score": p.get("score", 0),
                            "num_comments": p.get("num_comments", 0),
                            "created_utc": p.get("created_utc"),
                            "permalink": p.get("permalink", ""),
                        })
                    
                    print(f"    '{keyword}': {len(posts)} posts")
                else:
                    print(f"    '{keyword}': HTTP {resp.status_code}")
                
                time.sleep(2)  # Reddit rate limiting (be very respectful)
                
            except Exception as e:
                print(f"    Error: {e}")
                time.sleep(5)
    
    if all_posts:
        df = pd.DataFrame(all_posts)
        df["created_datetime"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
        df = df.drop_duplicates(subset=["permalink"])
        
        outpath = RAW_DIR / "layer3" / "reddit_safety_posts.parquet"
        df.to_parquet(outpath, index=False)
        print(f"\n  ✓ Saved {len(df)} unique Reddit posts → {outpath}")
    
    return pd.DataFrame(all_posts) if all_posts else None


# ─── Data Processing / Composite Index ──────────────────────────────────────

def process_disorder_index():
    """
    Build the 311-based "Disorder Density Index" per block per time window.
    This is the core of the MVP.
    
    Uses H3 hexagonal grid (resolution 9 ≈ 0.1 km²) for consistent block-level areas.
    Time windows: 4-hour blocks (morning/midday/afternoon/evening/night/late night).
    """
    print("\n" + "="*60)
    print("PROCESSING: Building Disorder Density Index")
    print("="*60)
    
    try:
        import h3
    except ImportError:
        print("  ⚠ h3 library not installed. Run: pip install h3")
        print("  Skipping H3 gridding, using neighborhood-level aggregation instead.")
        return _process_disorder_index_neighborhood()
    
    # Load 311 data
    path_311 = RAW_DIR / "layer2" / "311_cases_safety.parquet"
    if not path_311.exists():
        print("  ⚠ 311 data not found. Run --source 311 first.")
        return None
    
    df = pd.read_parquet(path_311)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["long"] = pd.to_numeric(df["long"], errors="coerce")
    df = df.dropna(subset=["lat", "long"])
    
    # Assign H3 hex cells (resolution 9)
    H3_RES = 9
    df["h3_index"] = df.apply(
        lambda row: h3.latlng_to_cell(row["lat"], row["long"], H3_RES), axis=1
    )
    
    # Assign time windows
    df["opened"] = pd.to_datetime(df["opened"], errors="coerce")
    df["hour"] = df["opened"].dt.hour
    df["time_window"] = pd.cut(
        df["hour"],
        bins=[0, 4, 8, 12, 16, 20, 24],
        labels=["late_night", "morning", "midday", "afternoon", "evening", "night"],
        right=False,
    )
    df["day_of_week"] = df["opened"].dt.day_name()
    df["year_month"] = df["opened"].dt.to_period("M").astype(str)
    
    # Aggregate: count per hex per time window per month
    disorder_index = (
        df.groupby(["h3_index", "year_month", "time_window", "service_name"])
        .size()
        .reset_index(name="case_count")
    )
    
    # Also compute total disorder score per hex per month
    hex_monthly = (
        df.groupby(["h3_index", "year_month"])
        .agg(
            total_cases=("service_request_id", "count"),
            encampments=("service_name", lambda x: (x == "Encampments").sum()),
            cleaning=("service_name", lambda x: (x == "Street and Sidewalk Cleaning").sum()),
            homeless=("service_name", lambda x: (x == "Homeless Concerns").sum()),
            graffiti=("service_name", lambda x: (x == "Graffiti").sum()),
            streetlights=("service_name", lambda x: (x == "Streetlights").sum()),
        )
        .reset_index()
    )
    
    # Get hex center coordinates for mapping
    hex_centers = df.groupby("h3_index").agg(
        center_lat=("lat", "mean"),
        center_lon=("long", "mean"),
        neighborhood=("neighborhood", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown"),
    ).reset_index()
    
    hex_monthly = hex_monthly.merge(hex_centers, on="h3_index", how="left")
    
    # Save
    disorder_index.to_parquet(PROCESSED_DIR / "disorder_index_detailed.parquet", index=False)
    hex_monthly.to_parquet(PROCESSED_DIR / "disorder_index_monthly.parquet", index=False)
    
    print(f"  ✓ Disorder index: {len(disorder_index)} records (detailed)")
    print(f"  ✓ Disorder index: {len(hex_monthly)} records (monthly by hex)")
    print(f"  Unique H3 cells: {df['h3_index'].nunique()}")
    
    return hex_monthly


def _process_disorder_index_neighborhood():
    """Fallback: neighborhood-level aggregation if H3 not available."""
    path_311 = RAW_DIR / "layer2" / "311_cases_safety.parquet"
    if not path_311.exists():
        return None
    
    df = pd.read_parquet(path_311)
    df["opened"] = pd.to_datetime(df["opened"], errors="coerce")
    df["year_month"] = df["opened"].dt.to_period("M").astype(str)
    
    result = (
        df.groupby(["neighborhood", "year_month", "service_name"])
        .size()
        .reset_index(name="case_count")
    )
    
    result.to_parquet(PROCESSED_DIR / "disorder_index_neighborhood.parquet", index=False)
    print(f"  ✓ Saved neighborhood-level disorder index: {len(result)} records")
    return result


def process_crime_overlay():
    """
    Build crime density overlay from SFPD incidents.
    Key insight: Where crime and 311 diverge reveals either underreporting
    (high crime, low 311) or perception problems (high 311, low crime).
    """
    print("\n" + "="*60)
    print("PROCESSING: Building Crime Density Overlay")
    print("="*60)
    
    path_sfpd = RAW_DIR / "layer1" / "sfpd_incidents.parquet"
    if not path_sfpd.exists():
        print("  ⚠ SFPD data not found. Run --source sfpd first.")
        return None
    
    df = pd.read_parquet(path_sfpd)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    df["incident_datetime"] = pd.to_datetime(df["incident_datetime"], errors="coerce")
    df["year_month"] = df["incident_datetime"].dt.to_period("M").astype(str)
    
    # Aggregate by neighborhood and month
    crime_monthly = (
        df.groupby(["analysis_neighborhood", "year_month", "incident_category"])
        .size()
        .reset_index(name="incident_count")
    )
    
    crime_monthly.to_parquet(PROCESSED_DIR / "crime_density_monthly.parquet", index=False)
    print(f"  ✓ Crime density: {len(crime_monthly)} records")
    
    return crime_monthly


def build_composite_index():
    """
    Build the composite Safety Perception Score.
    
    Formula (illustrative):
    Block Safety Perception Score =
      w1 × (inverse 311 disorder density) +
      w2 × (inverse crime density) +
      w3 × (foot traffic volume relative to baseline) +
      w4 × (Yelp safety sentiment score) +
      w5 × (City Survey neighborhood score)
    
    For MVP, we calibrate weights against City Survey ground truth.
    """
    print("\n" + "="*60)
    print("PROCESSING: Building Composite Safety Perception Index")
    print("="*60)
    
    # Check what data we have
    has_disorder = (PROCESSED_DIR / "disorder_index_monthly.parquet").exists()
    has_crime = (PROCESSED_DIR / "crime_density_monthly.parquet").exists()
    
    if not has_disorder:
        print("  ⚠ Need at least disorder index. Run processing first.")
        return None
    
    # Load available data
    disorder = pd.read_parquet(PROCESSED_DIR / "disorder_index_monthly.parquet")
    
    # For MVP, create a simple normalized score per neighborhood per month
    # Higher disorder = lower safety perception
    
    if "neighborhood" in disorder.columns:
        group_col = "neighborhood"
    elif "h3_index" in disorder.columns:
        group_col = "h3_index"
    else:
        print("  ⚠ No grouping column found in disorder index")
        return None
    
    # Normalize disorder score (0-100, where 100 = safest)
    max_cases = disorder["total_cases"].max()
    disorder["disorder_score_normalized"] = 100 * (1 - disorder["total_cases"] / max_cases)
    
    # If crime data available, merge
    if has_crime:
        crime = pd.read_parquet(PROCESSED_DIR / "crime_density_monthly.parquet")
        crime_totals = crime.groupby(["analysis_neighborhood", "year_month"])["incident_count"].sum().reset_index()
        crime_totals.columns = ["neighborhood", "year_month", "total_crimes"]
        
        max_crimes = crime_totals["total_crimes"].max()
        crime_totals["crime_score_normalized"] = 100 * (1 - crime_totals["total_crimes"] / max_crimes)
        
        # Merge
        if group_col == "neighborhood":
            composite = disorder.merge(
                crime_totals[["neighborhood", "year_month", "crime_score_normalized"]],
                on=["neighborhood", "year_month"],
                how="left",
            )
        else:
            composite = disorder.copy()
            composite["crime_score_normalized"] = 50  # Default if can't join
    else:
        composite = disorder.copy()
        composite["crime_score_normalized"] = 50
    
    # Compute composite (equal weights for MVP, calibrate later)
    composite["composite_safety_score"] = (
        0.5 * composite["disorder_score_normalized"] +
        0.5 * composite.get("crime_score_normalized", 50)
    )
    
    composite.to_parquet(PROCESSED_DIR / "composite_safety_index.parquet", index=False)
    print(f"  ✓ Composite safety index: {len(composite)} records")
    print(f"  Score range: {composite['composite_safety_score'].min():.1f} - {composite['composite_safety_score'].max():.1f}")
    
    return composite


# ─── Manual Download Instructions ───────────────────────────────────────────

def print_manual_downloads():
    """Print instructions for data that requires manual download."""
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*60)
    
    instructions = """
The following data sources require manual download or special access:

━━━ LAYER 1: Hard Incident Data ━━━

📊 SFPD CompStat Reports (Monthly PDFs)
   URL: https://www.sanfranciscopolice.org/stay-safe/crime-data/crime-reports
   Save to: data/raw/layer1/compstat/
   Format: PDF → will need parsing

📊 SFPD Crime Dashboard
   URL: https://www.sanfranciscopolice.org/stay-safe/crime-data/crime-dashboard
   Download the underlying data CSV
   Save to: data/raw/layer1/crime_dashboard.csv

📊 SFPD Community Survey
   URL: https://www.sanfranciscopolice.org/your-sfpd/community-surveys
   Save PDF reports to: data/raw/layer1/sfpd_survey/

━━━ LAYER 2: Condition & Environment Data ━━━

📊 Tent/Structure Counts (Quarterly)
   URL: https://sf.gov/data--healthy-streets-data-and-information
   Download dashboard data
   Save to: data/raw/layer2/tent_counts/

📊 Point in Time Homeless Count
   URL: https://sf.gov/data--healthy-streets-data-and-information
   Save PDF + data to: data/raw/layer2/pit_count/

📊 Community Ambassadors Program Data
   URL: https://sf.gov/data--community-ambassadors-program-data-2023-2024
   Download dashboard data
   Save to: data/raw/layer2/ambassador_data/

📊 Pit Stop Locations
   URL: https://sf.gov (Public Works data)
   Save to: data/raw/layer2/pit_stops/

━━━ LAYER 3: Behavioral & Sentiment Proxies ━━━

📊 SF City Survey — Safety & Policing Module (CRITICAL for validation)
   URL: https://sf.gov/data--city-survey-safety-and-policing
   Download ALL available data + dashboards
   Save to: data/raw/layer3/city_survey/
   NOTE: This is your calibration ground truth!

📊 BART Ridership (if auto-download failed)
   URL: https://www.bart.gov/about/reports/ridership
   Download monthly Excel files
   Save to: data/raw/layer3/bart_ridership/

📊 Muni Ridership by Route
   URL: https://www.sf.gov/data--muni-ridership
   Also check: https://www.sfmta.com
   Save to: data/raw/layer3/muni_ridership/

📊 Replica / SafeGraph Mobility Data (PAID/PARTNER access)
   Replica: https://replica.co — mobility data
   SafeGraph (now Dewey): foot traffic + spending
   Access path: MIT affiliation may open academic/government access
   Contact: Reach out via MIT City Science consortium

📊 SFCTA Downtown Travel Study
   URL: https://www.sfcta.org
   Save reports to: data/raw/layer3/sfcta/
"""
    print(instructions)


# ─── Export for Dashboard ───────────────────────────────────────────────────

def export_for_dashboard():
    """
    Export processed data as JSON files ready for the interactive dashboard.
    """
    print("\n" + "="*60)
    print("EXPORTING: Dashboard-ready JSON files")
    print("="*60)
    
    dashboard_dir = DATA_DIR / "dashboard_export"
    dashboard_dir.mkdir(exist_ok=True)
    
    # Export whatever processed data we have
    for parquet_file in PROCESSED_DIR.glob("*.parquet"):
        try:
            df = pd.read_parquet(parquet_file)
            json_path = dashboard_dir / f"{parquet_file.stem}.json"
            df.to_json(json_path, orient="records", date_format="iso")
            print(f"  ✓ {parquet_file.stem} → {json_path} ({len(df)} records)")
        except Exception as e:
            print(f"  ⚠ Error exporting {parquet_file.stem}: {e}")
    
    # Also export raw 311 as GeoJSON for map layers
    path_311 = RAW_DIR / "layer2" / "311_cases_safety.parquet"
    if path_311.exists():
        try:
            df = pd.read_parquet(path_311)
            df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
            df["long"] = pd.to_numeric(df["long"], errors="coerce")
            df = df.dropna(subset=["lat", "long"])
            
            # Convert to GeoJSON
            features = []
            for _, row in df.head(50000).iterrows():  # Cap for file size
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(row["long"]), float(row["lat"])],
                    },
                    "properties": {
                        "category": row.get("service_name", ""),
                        "date": str(row.get("opened", "")),
                        "neighborhood": row.get("neighborhood", ""),
                        "status": row.get("status_description", ""),
                    },
                })
            
            geojson = {"type": "FeatureCollection", "features": features}
            geojson_path = dashboard_dir / "311_safety_cases.geojson"
            with open(geojson_path, "w") as f:
                json.dump(geojson, f)
            print(f"  ✓ 311 GeoJSON → {geojson_path} ({len(features)} features)")
            
        except Exception as e:
            print(f"  ⚠ Error creating GeoJSON: {e}")
    
    # Export SFPD as GeoJSON
    path_sfpd = RAW_DIR / "layer1" / "sfpd_incidents.parquet"
    if path_sfpd.exists():
        try:
            df = pd.read_parquet(path_sfpd)
            df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
            df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
            df = df.dropna(subset=["latitude", "longitude"])
            
            features = []
            for _, row in df.head(50000).iterrows():
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(row["longitude"]), float(row["latitude"])],
                    },
                    "properties": {
                        "category": row.get("incident_category", ""),
                        "datetime": str(row.get("incident_datetime", "")),
                        "neighborhood": row.get("analysis_neighborhood", ""),
                        "district": row.get("police_district", ""),
                        "resolution": row.get("resolution", ""),
                    },
                })
            
            geojson = {"type": "FeatureCollection", "features": features}
            geojson_path = dashboard_dir / "sfpd_incidents.geojson"
            with open(geojson_path, "w") as f:
                json.dump(geojson, f)
            print(f"  ✓ SFPD GeoJSON → {geojson_path} ({len(features)} features)")
            
        except Exception as e:
            print(f"  ⚠ Error creating SFPD GeoJSON: {e}")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Public Safety Pulse — Master Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pull_all_data.py --all              Pull all data sources
  python pull_all_data.py --layer 1          Pull only Layer 1 (incidents)
  python pull_all_data.py --layer 2          Pull only Layer 2 (conditions)
  python pull_all_data.py --layer 3          Pull only Layer 3 (sentiment)
  python pull_all_data.py --source 311       Pull only 311 cases
  python pull_all_data.py --source sfpd      Pull only SFPD incidents
  python pull_all_data.py --process          Run processing only
  python pull_all_data.py --export           Export for dashboard
  python pull_all_data.py --manual           Show manual download instructions
        """,
    )
    
    parser.add_argument("--all", action="store_true", help="Pull all data sources")
    parser.add_argument("--layer", type=int, choices=[1, 2, 3], help="Pull specific layer")
    parser.add_argument("--source", type=str, help="Pull specific source (sfpd, fire, traffic, 311, bart, yelp, reddit)")
    parser.add_argument("--months", type=int, default=12, help="Months of history (default: 12)")
    parser.add_argument("--process", action="store_true", help="Run data processing")
    parser.add_argument("--export", action="store_true", help="Export for dashboard")
    parser.add_argument("--manual", action="store_true", help="Show manual download instructions")
    parser.add_argument("--yelp-key", type=str, help="Yelp Fusion API key")
    
    args = parser.parse_args()
    
    ensure_dirs()
    
    if args.manual:
        print_manual_downloads()
        return
    
    # Default to --all if no specific action
    if not any([args.all, args.layer, args.source, args.process, args.export]):
        args.all = True
    
    # ── Pull Data ──
    
    if args.all or args.layer == 1 or args.source in ["sfpd"]:
        pull_sfpd_incidents(args.months)
    
    if args.all or args.layer == 1 or args.source in ["fire"]:
        pull_fire_calls(args.months)
    
    if args.all or args.layer == 1 or args.source in ["traffic"]:
        pull_traffic_crashes(args.months)
    
    if args.all or args.layer == 2 or args.source in ["311"]:
        pull_311_cases(args.months)
    
    if args.all or args.layer == 3 or args.source in ["bart"]:
        pull_bart_ridership()
    
    if args.all or args.layer == 3 or args.source in ["yelp"]:
        pull_yelp_reviews(api_key=args.yelp_key)
    
    if args.all or args.layer == 3 or args.source in ["reddit"]:
        pull_reddit_sentiment()
    
    # ── Process ──
    
    if args.all or args.process:
        process_disorder_index()
        process_crime_overlay()
        build_composite_index()
    
    # ── Export ──
    
    if args.all or args.export:
        export_for_dashboard()
    
    # Always show manual downloads
    if args.all:
        print_manual_downloads()
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"\nData directory: {DATA_DIR}")
    print(f"Raw data: {RAW_DIR}")
    print(f"Processed data: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
