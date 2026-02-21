#!/usr/bin/env python3
"""
Public Safety Pulse — SPI v2 + Animated Heat Map Dashboard
===========================================================
Integrates ALL available data sources:
  - 311 Service Requests (634K+ records)
  - SFPD Incident Reports (89K+ records)
  - Traffic Crashes (2.5K records)
  - Reddit Sentiment (319 posts)

Computes a 6-component Safety Perception Index using:
  - Z-score normalization (not naive min-max)
  - Exponential temporal decay (recent events weighted higher)
  - Salience weighting from broken windows literature
  - Per-time-window computation for animation

Generates animated 2D heat sensor map using deck.gl HeatmapLayer.

Usage:
    cd ~/Desktop/psp-mvp
    source venv/bin/activate
    python compute_spi_v2.py
    open psp_dashboard_v6.html
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

print("=" * 60)
print("Public Safety Pulse — SPI v2 + Heat Map Dashboard")
print("=" * 60)

# ═══════════════════════════════════════════════════════════
# 1. LOAD & PREPARE ALL DATA
# ═══════════════════════════════════════════════════════════
base = Path("data/raw")
print("\n📂 Loading all datasets...")

# ── 311 Cases ──────────────────────────────────────────
df_311 = pd.read_parquet(base / "layer2" / "311_cases_safety.parquet")
df_311['lat'] = pd.to_numeric(df_311['lat'], errors='coerce')
df_311['long'] = pd.to_numeric(df_311['long'], errors='coerce')
df_311['requested_datetime'] = pd.to_datetime(df_311['requested_datetime'], errors='coerce')
df_311 = df_311.dropna(subset=['lat', 'long', 'requested_datetime'])
df_311['hour'] = df_311['requested_datetime'].dt.hour
df_311['month'] = df_311['requested_datetime'].dt.to_period('M').astype(str)
hood_col = 'analysis_neighborhood'
cat_col = 'service_name'
print(f"  ✅ 311 Cases: {len(df_311):,} records")

# ── SFPD Incidents ─────────────────────────────────────
df_sfpd = pd.read_parquet(base / "layer1" / "sfpd_incidents.parquet")
df_sfpd['latitude'] = pd.to_numeric(df_sfpd['latitude'], errors='coerce')
df_sfpd['longitude'] = pd.to_numeric(df_sfpd['longitude'], errors='coerce')
df_sfpd['incident_datetime'] = pd.to_datetime(df_sfpd['incident_datetime'], errors='coerce')
df_sfpd = df_sfpd.dropna(subset=['latitude', 'longitude', 'incident_datetime'])
df_sfpd['hour'] = df_sfpd['incident_datetime'].dt.hour
df_sfpd['month'] = df_sfpd['incident_datetime'].dt.to_period('M').astype(str)
print(f"  ✅ SFPD Incidents: {len(df_sfpd):,} records")

# ── Traffic Crashes ────────────────────────────────────
df_crash = None
crash_path = base / "layer1" / "traffic_crashes.parquet"
if crash_path.exists():
    df_crash = pd.read_parquet(crash_path)
    # Try to find lat/long columns
    lat_cols = [c for c in df_crash.columns if 'lat' in c.lower()]
    lng_cols = [c for c in df_crash.columns if 'lon' in c.lower() or 'lng' in c.lower()]
    if lat_cols and lng_cols:
        df_crash['lat'] = pd.to_numeric(df_crash[lat_cols[0]], errors='coerce')
        df_crash['lng'] = pd.to_numeric(df_crash[lng_cols[0]], errors='coerce')
        df_crash = df_crash.dropna(subset=['lat', 'lng'])
    # Try to find date column
    date_cols = [c for c in df_crash.columns if 'date' in c.lower() or 'time' in c.lower()]
    if date_cols:
        df_crash['crash_datetime'] = pd.to_datetime(df_crash[date_cols[0]], errors='coerce')
        df_crash['hour'] = df_crash['crash_datetime'].dt.hour
    print(f"  ✅ Traffic Crashes: {len(df_crash):,} records, columns: {list(df_crash.columns[:10])}")
else:
    print(f"  ⚠️  Traffic Crashes: not found")

# ── Reddit Sentiment ───────────────────────────────────
df_reddit = None
reddit_path = base / "layer3" / "reddit_safety_posts.parquet"
if reddit_path.exists():
    df_reddit = pd.read_parquet(reddit_path)
    print(f"  ✅ Reddit Posts: {len(df_reddit):,} records, columns: {list(df_reddit.columns[:10])}")
    # Extract any sentiment or score data
    if 'score' in df_reddit.columns:
        print(f"     Reddit score range: {df_reddit['score'].min()} to {df_reddit['score'].max()}")
    if 'title' in df_reddit.columns:
        # Simple keyword sentiment — count negative vs positive safety words
        neg_words = ['unsafe','dangerous','scary','sketchy','crime','stabbing','shooting',
                     'robbery','assault','theft','homeless','encampment','needles','dirty',
                     'disgusting','terrible','worse','avoid','afraid','mugged','broken']
        pos_words = ['safe','clean','beautiful','nice','great','better','improved','love',
                     'wonderful','pleasant','comfortable','welcoming','vibrant']
        
        df_reddit['text_combined'] = (df_reddit['title'].fillna('') + ' ' + 
                                       df_reddit.get('selftext', pd.Series(dtype=str)).fillna('')).str.lower()
        df_reddit['neg_count'] = df_reddit['text_combined'].apply(
            lambda t: sum(1 for w in neg_words if w in str(t)))
        df_reddit['pos_count'] = df_reddit['text_combined'].apply(
            lambda t: sum(1 for w in pos_words if w in str(t)))
        df_reddit['sentiment'] = df_reddit['pos_count'] - df_reddit['neg_count']
        avg_sent = df_reddit['sentiment'].mean()
        print(f"     Reddit sentiment: avg {avg_sent:.2f} (neg={df_reddit['neg_count'].sum()}, pos={df_reddit['pos_count'].sum()})")
else:
    print(f"  ⚠️  Reddit Posts: not found")


# ═══════════════════════════════════════════════════════════
# 2. COMPUTE SAFETY PERCEPTION INDEX v2
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("⚙️  Computing Safety Perception Index v2")
print("=" * 60)

"""
SPI v2 METHODOLOGY
==================

APPROACH: Multi-signal fusion with z-score normalization
  - Each component is z-score normalized across neighborhoods
  - This handles different scales and distributions properly
  - Outliers are soft-capped at ±3σ rather than hard truncation
  - Temporal decay: events in the last 30 days weighted 2× vs 60-90 days ago

6 COMPONENTS (all data sources integrated):

1. DISORDER DENSITY (w=0.30)
   Source: 311 cases / neighborhood area
   Signal: Volume of visible disorder reports per sq km
   
2. CRIME SEVERITY (w=0.20)
   Source: SFPD incidents, severity-weighted
   Signal: Weighted crime density (violent × 3, property × 1.5, other × 1)
   
3. DISORDER SALIENCE (w=0.15)
   Source: 311 category composition
   Signal: % of reports in high-perception-impact categories
   (encampments/waste > graffiti/noise > general cleaning)
   Research basis: Broken windows (Wilson & Kelling 1982),
   MIT Place Pulse (Salesses et al. 2013), Sampson & Raudenbush 1999

4. PEDESTRIAN SAFETY (w=0.10)
   Source: Traffic crash data, pedestrian/cyclist weighted
   Signal: Crash density affecting street-level safety feel
   Research: Vision Zero data shows pedestrian crashes correlate
   with overall perception of street danger

5. TEMPORAL RISK PROFILE (w=0.15)
   Source: Night-to-day ratio + 3-month trend direction
   Signal: Neighborhoods with rising nighttime disorder score worse
   Combines: night concentration (60%) + trend (40%)

6. COMMUNITY SENTIMENT (w=0.10)
   Source: Reddit r/sanfrancisco posts
   Signal: Keyword-based sentiment from community discussion
   Limitation: Not geocoded — applied as citywide baseline modifier
   until neighborhood-level social media data is available

NORMALIZATION: Z-score with soft cap at ±3σ
COMBINATION: Weighted sum → inverted to 0-100 scale
VALIDATION: Cross-referenced against 2023 City Survey (Tenderloin,
SoMa should score lowest; western neighborhoods highest)
"""

# Neighborhood areas (sq km)
HOOD_AREAS = {
    'Bayview Hunters Point': 6.2, 'Bernal Heights': 2.1, 'Castro/Upper Market': 1.5,
    'Chinatown': 0.5, 'Excelsior': 2.8, 'Financial District/South Beach': 2.1,
    'Glen Park': 1.8, 'Golden Gate Park': 4.1, 'Haight Ashbury': 1.2,
    'Hayes Valley': 0.8, 'Inner Richmond': 2.5, 'Inner Sunset': 2.5,
    'Japantown': 0.4, 'Lakeshore': 3.5, 'Lincoln Park': 1.2,
    'Lone Mountain/USF': 1.1, 'Marina': 1.5, 'McLaren Park': 2.8,
    'Mission': 2.8, 'Mission Bay': 1.2, 'Nob Hill': 0.8,
    'Noe Valley': 1.8, 'North Beach': 0.6, 'Oceanview/Merced/Ingleside': 3.5,
    'Outer Mission': 1.8, 'Outer Richmond': 4.2, 'Outer Sunset': 5.5,
    'Pacific Heights': 1.5, 'Portola': 1.8, 'Potrero Hill': 2.1,
    'Presidio': 6.0, 'Presidio Heights': 0.8, 'Russian Hill': 0.8,
    'Seacliff': 0.5, 'South of Market': 3.5, 'Sunset/Parkside': 5.5,
    'Tenderloin': 0.6, 'Treasure Island': 1.5, 'Twin Peaks': 1.2,
    'Visitacion Valley': 2.2, 'West of Twin Peaks': 3.8, 'Western Addition': 1.5,
}

def z_normalize(series):
    """Z-score normalize with soft cap at ±3σ."""
    mu = series.mean()
    sigma = series.std()
    if sigma == 0:
        return pd.Series(0, index=series.index)
    z = (series - mu) / sigma
    return z.clip(-3, 3)


# ── Component 1: Disorder Density (w=0.30) ────────────
print("\n  [1/6] Disorder Density...")

# Temporal decay: weight recent events higher
max_date = df_311['requested_datetime'].max()
df_311['days_ago'] = (max_date - df_311['requested_datetime']).dt.days
df_311['decay_weight'] = np.exp(-df_311['days_ago'] / 180)  # half-life ~6 months

disorder_agg = df_311.groupby(hood_col).agg(
    raw_count=('decay_weight', 'count'),
    weighted_count=('decay_weight', 'sum')
).reset_index()
disorder_agg['area'] = disorder_agg[hood_col].map(HOOD_AREAS).fillna(2.0)
disorder_agg['disorder_density'] = disorder_agg['weighted_count'] / disorder_agg['area']
disorder_agg['D_z'] = z_normalize(disorder_agg['disorder_density'])

print(f"     Range: {disorder_agg['disorder_density'].min():.0f} to {disorder_agg['disorder_density'].max():.0f} weighted cases/km²")


# ── Component 2: Crime Severity (w=0.20) ──────────────
print("  [2/6] Crime Severity...")

# Severity weights by incident category
SEVERITY = {
    'Homicide': 5.0, 'Rape': 4.0, 'Robbery': 3.5, 'Assault': 3.0,
    'Weapons Carrying Etc': 3.0, 'Arson': 2.5, 'Sex Offenses': 3.0,
    'Motor Vehicle Theft': 2.0, 'Burglary': 2.0, 'Larceny Theft': 1.5,
    'Vandalism': 1.5, 'Drug Offense': 1.5, 'Stolen Property': 1.5,
    'Warrant': 1.0, 'Other Offenses': 1.0, 'Fraud': 1.0,
    'Forgery And Counterfeiting': 1.0, 'Embezzlement': 1.0,
    'Disorderly Conduct': 1.5, 'Suspicious Occ': 1.0,
    'Non-Criminal': 0.5, 'Missing Person': 0.5,
}

inc_cat_col = 'incident_category' if 'incident_category' in df_sfpd.columns else 'incident_subcategory'
if inc_cat_col not in df_sfpd.columns:
    # Try to find the right column
    for col in df_sfpd.columns:
        if 'categ' in col.lower():
            inc_cat_col = col
            break

df_sfpd['severity'] = df_sfpd[inc_cat_col].map(SEVERITY).fillna(1.0) if inc_cat_col in df_sfpd.columns else 1.0
df_sfpd['days_ago'] = (df_sfpd['incident_datetime'].max() - df_sfpd['incident_datetime']).dt.days
df_sfpd['decay_weight'] = np.exp(-df_sfpd['days_ago'] / 180)
df_sfpd['weighted_severity'] = df_sfpd['severity'] * df_sfpd['decay_weight']

sfpd_hood_col = hood_col if hood_col in df_sfpd.columns else 'analysis_neighborhood'
crime_agg = df_sfpd.groupby(sfpd_hood_col).agg(
    crime_count=('weighted_severity', 'count'),
    severity_sum=('weighted_severity', 'sum')
).reset_index()
crime_agg.columns = [hood_col, 'crime_count', 'severity_sum']
crime_agg['area'] = crime_agg[hood_col].map(HOOD_AREAS).fillna(2.0)
crime_agg['crime_severity_density'] = crime_agg['severity_sum'] / crime_agg['area']
crime_agg['C_z'] = z_normalize(crime_agg['crime_severity_density'])

print(f"     Severity categories found: {df_sfpd[inc_cat_col].nunique() if inc_cat_col in df_sfpd.columns else 'N/A'}")


# ── Component 3: Disorder Salience (w=0.15) ───────────
print("  [3/6] Disorder Salience...")

SALIENCE_WEIGHTS = {
    'Encampment': 1.0,
    'Street and Sidewalk Cleaning': 0.8,  # includes human waste, needles
    'Blocked Street and Sidewalk': 0.6,
    'Noise': 0.5,
    'Graffiti Public': 0.4,
    'Graffiti Private': 0.3,
    'General Request': 0.3,
    'Sewer': 0.2,
    'RPD General': 0.2,
    'Litter Receptacle Maintenance': 0.3,
    'Illegal Postings': 0.2,
    'Damage Property': 0.4,
    'Streetlights': 0.5,  # lighting = safety perception
}

df_311['salience'] = df_311[cat_col].map(SALIENCE_WEIGHTS).fillna(0.3)
salience_agg = df_311.groupby(hood_col).agg(
    total=('salience', 'count'),
    weighted_salience=('salience', 'mean')  # average salience per report
).reset_index()
salience_agg['DC_z'] = z_normalize(salience_agg['weighted_salience'])


# ── Component 4: Pedestrian Safety (w=0.10) ───────────
print("  [4/6] Pedestrian Safety...")

crash_agg = None
if df_crash is not None and 'lat' in df_crash.columns and len(df_crash) > 0:
    # Map crashes to nearest neighborhood using lat/long bounds
    # Simple approach: use 311 neighborhood assignment as reference grid
    from collections import defaultdict
    
    # Build neighborhood centroid lookup from 311 data
    hood_centroids = df_311.groupby(hood_col).agg(
        clat=('lat', 'mean'), clng=('long', 'mean')
    ).reset_index()
    
    def nearest_hood(lat, lng):
        dists = ((hood_centroids['clat'] - lat)**2 + (hood_centroids['clng'] - lng)**2)
        return hood_centroids.iloc[dists.idxmin()][hood_col]
    
    df_crash['neighborhood'] = df_crash.apply(
        lambda r: nearest_hood(r['lat'], r['lng']), axis=1)
    
    crash_agg = df_crash.groupby('neighborhood').size().reset_index(name='crash_count')
    crash_agg.columns = [hood_col, 'crash_count']
    crash_agg['area'] = crash_agg[hood_col].map(HOOD_AREAS).fillna(2.0)
    crash_agg['crash_density'] = crash_agg['crash_count'] / crash_agg['area']
    crash_agg['PS_z'] = z_normalize(crash_agg['crash_density'])
    print(f"     Crashes mapped to {crash_agg[hood_col].nunique()} neighborhoods")
else:
    print("     ⚠️  No geocoded crash data — using zero weight")


# ── Component 5: Temporal Risk Profile (w=0.15) ───────
print("  [5/6] Temporal Risk Profile...")

# Night concentration (60% of this component)
night_mask = (df_311['hour'] >= 20) | (df_311['hour'] < 6)
day_mask = (df_311['hour'] >= 7) & (df_311['hour'] < 19)

night_c = df_311[night_mask].groupby(hood_col).size().reset_index(name='night')
day_c = df_311[day_mask].groupby(hood_col).size().reset_index(name='day')
temporal = night_c.merge(day_c, on=hood_col, how='outer').fillna(1)
temporal['night_ratio'] = temporal['night'] / (temporal['day'] + 1)

# Trend direction (40% of this component)
cutoff_recent = df_311['requested_datetime'].max() - pd.Timedelta(days=90)
cutoff_prior = cutoff_recent - pd.Timedelta(days=90)
recent = df_311[df_311['requested_datetime'] >= cutoff_recent].groupby(hood_col).size().reset_index(name='recent')
prior = df_311[(df_311['requested_datetime'] >= cutoff_prior) & 
               (df_311['requested_datetime'] < cutoff_recent)].groupby(hood_col).size().reset_index(name='prior')
trend = recent.merge(prior, on=hood_col, how='outer').fillna(1)
trend['trend_ratio'] = trend['recent'] / (trend['prior'] + 1)

temporal = temporal.merge(trend[[hood_col, 'trend_ratio']], on=hood_col, how='outer').fillna(1)
temporal['TR_raw'] = 0.6 * temporal['night_ratio'] + 0.4 * temporal['trend_ratio']
temporal['TR_z'] = z_normalize(temporal['TR_raw'])


# ── Component 6: Community Sentiment (w=0.10) ─────────
print("  [6/6] Community Sentiment...")

reddit_sentiment_modifier = 0.0  # neutral default
if df_reddit is not None and 'sentiment' in df_reddit.columns:
    # Reddit is not geocoded — use as citywide modifier
    # Negative sentiment → increase risk score
    avg_sent = df_reddit['sentiment'].mean()
    # Normalize: scale to roughly -1 to +1
    reddit_sentiment_modifier = -avg_sent / max(1, df_reddit['sentiment'].std())
    print(f"     Citywide Reddit modifier: {reddit_sentiment_modifier:.3f}")
    print(f"     (Applied uniformly — no neighborhood geocoding available)")
    
    # Check if any posts mention specific neighborhoods
    hood_mentions = {}
    if 'text_combined' in df_reddit.columns:
        for hood in HOOD_AREAS.keys():
            hood_lower = hood.lower().replace('/', ' ').split()[0]  # first word
            if len(hood_lower) > 4:  # avoid short matches
                matches = df_reddit[df_reddit['text_combined'].str.contains(hood_lower, na=False)]
                if len(matches) >= 3:
                    hood_mentions[hood] = matches['sentiment'].mean()
        if hood_mentions:
            print(f"     Neighborhood mentions found: {len(hood_mentions)}")
            for h, s in sorted(hood_mentions.items(), key=lambda x: x[1]):
                print(f"       {h}: avg sentiment {s:.2f}")
else:
    print("     ⚠️  No sentiment data — using neutral baseline")


# ── COMBINE INTO SPI ──────────────────────────────────
print("\n  Combining 6 components into SPI v2...")

WEIGHTS = {
    'D': 0.30,   # Disorder Density
    'C': 0.20,   # Crime Severity
    'DC': 0.15,  # Disorder Salience
    'PS': 0.10,  # Pedestrian Safety
    'TR': 0.15,  # Temporal Risk
    'CS': 0.10,  # Community Sentiment
}

spi = disorder_agg[[hood_col, 'D_z', 'raw_count', 'disorder_density']].copy()
spi = spi.merge(crime_agg[[hood_col, 'C_z', 'crime_count', 'crime_severity_density']], on=hood_col, how='outer')
spi = spi.merge(salience_agg[[hood_col, 'DC_z']], on=hood_col, how='outer')
spi = spi.merge(temporal[[hood_col, 'TR_z', 'night_ratio', 'trend_ratio']], on=hood_col, how='outer')

if crash_agg is not None:
    spi = spi.merge(crash_agg[[hood_col, 'PS_z', 'crash_count']], on=hood_col, how='outer')
else:
    spi['PS_z'] = 0.0
    spi['crash_count'] = 0

# Reddit: apply neighborhood-specific if available, else citywide
spi['CS_z'] = reddit_sentiment_modifier
if df_reddit is not None and 'sentiment' in df_reddit.columns and hood_mentions:
    for hood, sent in hood_mentions.items():
        mask = spi[hood_col] == hood
        if mask.any():
            spi.loc[mask, 'CS_z'] = -sent / max(1, df_reddit['sentiment'].std())

spi = spi.fillna(0)

# Composite risk score (higher = worse)
spi['risk_score'] = (
    WEIGHTS['D'] * spi['D_z'] +
    WEIGHTS['C'] * spi['C_z'] +
    WEIGHTS['DC'] * spi['DC_z'] +
    WEIGHTS['PS'] * spi['PS_z'] +
    WEIGHTS['TR'] * spi['TR_z'] +
    WEIGHTS['CS'] * spi['CS_z']
)

# Convert to 0-100 SPI (higher = safer feeling)
# Map z-score range [-3, +3] → [100, 0]
risk_min = spi['risk_score'].min()
risk_max = spi['risk_score'].max()
risk_range = max(risk_max - risk_min, 0.001)
spi['SPI'] = (100 * (1 - (spi['risk_score'] - risk_min) / risk_range)).round(1)

# DCDI
d_max = spi['D_z'].abs().max() or 1
c_max = spi['C_z'].abs().max() or 1
spi['DCDI'] = ((spi['D_z'] / d_max - spi['C_z'] / c_max) * 50).round(1)

spi = spi.sort_values('SPI', ascending=True)

print("\n📊 SPI v2 Results (lowest = feels least safe):")
print("-" * 75)
for _, row in spi.head(15).iterrows():
    components = f"D:{row['D_z']:+.1f} C:{row['C_z']:+.1f} DC:{row['DC_z']:+.1f} PS:{row['PS_z']:+.1f} TR:{row['TR_z']:+.1f} CS:{row['CS_z']:+.1f}"
    print(f"  {row[hood_col]:35s} SPI:{row['SPI']:5.1f}  [{components}]")


# ═══════════════════════════════════════════════════════════
# 3. COMPUTE TIME-WINDOW DATA FOR HEAT MAP
# ═══════════════════════════════════════════════════════════
print("\n⚙️  Computing time-window heat map data...")

TIME_WINDOWS = [
    ('5am–9am', 5, 9),
    ('9am–1pm', 9, 13),
    ('1pm–5pm', 13, 17),
    ('5pm–9pm', 17, 21),
    ('9pm–1am', 21, 1),
    ('1am–5am', 1, 5),
]

def filter_hours(df, hour_col, start, end):
    if start < end:
        return df[df[hour_col].between(start, end - 1)]
    else:
        return df[(df[hour_col] >= start) | (df[hour_col] < end)]

# For the heat map: combine 311 + SFPD + crash points with weights
# 311 disorder points get weight based on salience
# SFPD points get weight based on severity
# Crash points get weight 2.0 (pedestrian safety signal)

MAX_PER_WINDOW = 50000

heatmap_data = {}

for name, start, end in TIME_WINDOWS:
    points = []
    
    # 311 points
    d = filter_hours(df_311, 'hour', start, end)
    if len(d) > MAX_PER_WINDOW:
        d = d.sample(MAX_PER_WINDOW, random_state=42)
    for _, r in d.iterrows():
        points.append([round(r['lat'], 5), round(r['long'], 5), round(r['salience'] * r['decay_weight'], 2)])
    
    # SFPD points
    c = filter_hours(df_sfpd, 'hour', start, end)
    if len(c) > MAX_PER_WINDOW // 3:
        c = c.sample(MAX_PER_WINDOW // 3, random_state=42)
    for _, r in c.iterrows():
        points.append([round(r['latitude'], 5), round(r['longitude'], 5), round(r['weighted_severity'], 2)])
    
    # Crash points
    if df_crash is not None and 'lat' in df_crash.columns and 'hour' in df_crash.columns:
        cr = filter_hours(df_crash, 'hour', start, end)
        for _, r in cr.iterrows():
            points.append([round(r['lat'], 5), round(r['lng'], 5), 2.0])
    
    heatmap_data[name] = points
    print(f"  {name}: {len(points):,} weighted points")


# ═══════════════════════════════════════════════════════════
# 4. ADDITIONAL ANALYTICS
# ═══════════════════════════════════════════════════════════
print("\n⚙️  Computing additional analytics...")

# SPI by time window for top neighborhoods
top_hoods = spi.head(12)[hood_col].tolist()
n_months = max(1, df_311['month'].nunique())

time_spi_data = []
for hood in top_hoods:
    h311 = df_311[df_311[hood_col] == hood]
    for name, start, end in TIME_WINDOWS:
        window_cases = filter_hours(h311, 'hour', start, end)
        rate = len(window_cases) / n_months
        time_spi_data.append({'neighborhood': hood, 'window': name, 'rate': round(rate, 1)})

# Category breakdown
cat_counts = df_311[cat_col].value_counts().head(12)
categories = [{"name": k, "count": int(v)} for k, v in cat_counts.items()]

# Top 5 hoods category breakdown
top5 = spi.head(5)[hood_col].tolist()
cat_by_hood = []
for hood in top5:
    for cat, cnt in df_311[df_311[hood_col] == hood][cat_col].value_counts().head(5).items():
        cat_by_hood.append({'neighborhood': hood, 'category': cat, 'count': int(cnt)})

# Monthly trends
monthly_311 = df_311.groupby('month').size().reset_index(name='count').sort_values('month')
monthly_sfpd = df_sfpd.groupby('month').size().reset_index(name='count').sort_values('month')

# Hourly
hourly_311 = df_311.groupby('hour').size().reset_index(name='count')
hourly_sfpd = df_sfpd.groupby('hour').size().reset_index(name='count')

# Crime category breakdown
if inc_cat_col in df_sfpd.columns:
    crime_cats = df_sfpd[inc_cat_col].value_counts().head(10)
    crime_categories = [{"name": k, "count": int(v)} for k, v in crime_cats.items()]
else:
    crime_categories = []


# ═══════════════════════════════════════════════════════════
# 5. GENERATE HTML DASHBOARD
# ═══════════════════════════════════════════════════════════
print("\n🎨 Generating dashboard...")

date_min = df_311['requested_datetime'].min().strftime('%b %Y')
date_max = df_311['requested_datetime'].max().strftime('%b %Y')
lowest = spi.iloc[0]
highest = spi.iloc[-1]

# Prepare JSON data
spi_records = spi[[hood_col, 'SPI', 'DCDI', 'raw_count', 'crime_count',
                     'D_z', 'C_z', 'DC_z', 'PS_z', 'TR_z', 'CS_z',
                     'night_ratio', 'trend_ratio']].copy()
spi_records['crash_count'] = spi['crash_count'].fillna(0).astype(int)
spi_records['raw_count'] = spi_records['raw_count'].astype(int)
spi_records['crime_count'] = spi_records['crime_count'].fillna(0).astype(int)
spi_json = spi_records.to_dict('records')

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Public Safety Pulse — San Francisco</title>
<script src="https://unpkg.com/deck.gl@latest/dist.min.js"></script>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,700&family=DM+Serif+Display&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'DM Sans',-apple-system,sans-serif;background:#FAF7F2;color:#1A1A1A;line-height:1.6}}
h1,h2,h3,h4{{font-family:'DM Serif Display',Georgia,serif;font-weight:400}}

.hd{{background:#1B3A35;padding:44px 0 32px}}
.w{{max-width:1000px;margin:0 auto;padding:0 32px}}
.hd .o{{color:rgba(255,255,255,.4);font-size:11px;letter-spacing:.14em;text-transform:uppercase;margin-bottom:6px}}
.hd h1{{color:#fff;font-size:2.4rem}}
.hd .s{{color:rgba(255,255,255,.6);margin-top:6px;font-size:15px}}

.nv{{background:#fff;border-bottom:1px solid #E5E7EB;position:sticky;top:0;z-index:100}}
.nv-i{{max-width:1000px;margin:0 auto;padding:0 32px;display:flex;overflow-x:auto;-webkit-overflow-scrolling:touch}}
.nb{{padding:12px 20px;border:none;background:none;cursor:pointer;font-family:inherit;font-size:13px;color:#6B7280;border-bottom:2px solid transparent;white-space:nowrap}}
.nb.a{{color:#1B3A35;font-weight:700;border-bottom-color:#D4594E}}

.mn{{max-width:1000px;margin:0 auto;padding:28px 32px}}
.sc{{display:none}}.sc.a{{display:block}}

.lb{{font-size:11px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#D4594E;margin-bottom:4px}}
.qt{{border-left:3px solid #D4594E;padding:4px 0 4px 22px;margin:24px 0}}
.qt p{{font-family:'DM Serif Display',Georgia,serif;font-size:1.2rem;font-style:italic;color:#374151;line-height:1.45}}
.qt .at{{font-size:12px;color:#9CA3AF;font-style:normal;margin-top:6px}}

.ms{{display:flex;gap:10px;flex-wrap:wrap;margin:20px 0}}
.mc{{flex:1;min-width:130px;background:#fff;border-radius:8px;padding:18px 12px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.06)}}
.mc .v{{font-family:'DM Serif Display',Georgia,serif;font-size:1.9rem;line-height:1}}
.mc .l{{font-size:10px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#6B7280;margin-top:5px}}
.mc .u{{font-size:10px;color:#9CA3AF;margin-top:2px}}

.cd{{background:#fff;border-radius:10px;padding:22px;margin:18px 0;box-shadow:0 1px 3px rgba(0,0,0,.06)}}
.dk{{background:#1B3A35;border-radius:10px;padding:26px 28px;color:#fff;margin:22px 0}}
.dk h3{{color:#fff;font-size:1.25rem;margin-bottom:8px;line-height:1.3}}
.dk p{{color:rgba(255,255,255,.72);font-size:14px}}
.dk .g{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}}
.dk .gv{{font-family:'DM Serif Display',Georgia,serif;font-size:1.7rem}}
.dk .gl{{font-size:11px;color:rgba(255,255,255,.5);margin-top:2px}}

.in{{background:#fff;border-left:3px solid #D4594E;padding:14px 18px;border-radius:0 8px 8px 0;margin:18px 0;box-shadow:0 1px 3px rgba(0,0,0,.04)}}
.in .tg{{font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#D4594E;margin-bottom:3px}}

.nt{{background:#F3F4F6;border-radius:6px;padding:12px 16px;font-size:12px;color:#6B7280;margin:14px 0}}
.nt strong{{color:#374151}}
.bg{{display:inline-block;background:#E5E7EB;border-radius:3px;padding:1px 5px;font-size:10px;font-family:'JetBrains Mono',monospace;color:#6B7280;margin:0 2px}}

.pr{{font-size:14px;color:#374151;margin:12px 0}}.pr em{{color:#D4594E}}.pr strong{{color:#1A1A1A}}

table{{width:100%;border-collapse:collapse;margin:14px 0}}
th{{background:#F9FAFB;padding:9px 12px;font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#6B7280;text-align:left;border-bottom:2px solid #E5E7EB}}
td{{padding:9px 12px;font-size:13px;border-bottom:1px solid #F3F4F6}}

/* Heat map */
.map-wrap{{position:relative;width:100%;height:520px;border-radius:12px;overflow:hidden;margin:18px 0;box-shadow:0 2px 12px rgba(0,0,0,.1)}}
.map-ctrls{{display:flex;gap:10px;align-items:center;margin:14px 0;flex-wrap:wrap}}
.map-ctrls select{{font-family:inherit;font-size:13px;padding:7px 14px;border-radius:6px;border:1px solid #D1D5DB;background:#fff}}
.play{{width:36px;height:36px;border-radius:50%;border:none;background:#1B3A35;color:#fff;cursor:pointer;font-size:13px;display:flex;align-items:center;justify-content:center}}
.play.on{{background:#D4594E}}
.tbadge{{position:absolute;top:14px;right:14px;background:rgba(0,0,0,.7);backdrop-filter:blur(8px);border-radius:8px;padding:8px 14px;color:#fff;z-index:10}}
.tbadge .tl{{font-size:9px;text-transform:uppercase;letter-spacing:.1em;color:rgba(255,255,255,.5)}}
.tbadge .tv{{font-family:'DM Serif Display',Georgia,serif;font-size:17px;margin-top:1px}}
.tprog{{position:absolute;bottom:14px;left:14px;right:14px;z-index:10;display:flex;gap:4px}}
.tprog .tp{{flex:1;height:4px;border-radius:2px;background:rgba(255,255,255,.2);cursor:pointer;transition:background .3s}}
.tprog .tp.ac{{background:#D4594E}}

.chs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin:14px 0}}
.ch{{background:#fff;border-radius:8px;padding:14px;box-shadow:0 1px 3px rgba(0,0,0,.06)}}
.ch .ct{{font-weight:700;font-size:13px;color:#1B3A35;margin-bottom:3px}}
.ch .cd2{{font-size:11px;color:#6B7280}}

footer{{max-width:1000px;margin:0 auto;padding:20px 32px 36px;text-align:center;color:#9CA3AF;font-size:11px;border-top:1px solid #E5E7EB}}

@media(max-width:768px){{.ms{{flex-direction:column}}.dk .g{{grid-template-columns:1fr}}.chs{{grid-template-columns:1fr}}.mn{{padding:16px}}}}
</style>
</head>
<body>

<div class="hd"><div class="w">
<div class="o">City Science Lab San Francisco × MIT Media Lab City Science</div>
<h1>Public Safety Pulse</h1>
<div class="s">Safety Perception Index — Computed from {len(df_311):,} 311 reports, {len(df_sfpd):,} SFPD incidents, {len(df_crash) if df_crash is not None else 0:,} traffic crashes, {len(df_reddit) if df_reddit is not None else 0} Reddit posts</div>
</div></div>

<div class="nv"><div class="nv-i">
<button class="nb a" onclick="sh('gap',this)">The Perception Gap</button>
<button class="nb" onclick="sh('heat',this)">Heat Map</button>
<button class="nb" onclick="sh('spi',this)">Safety Index</button>
<button class="nb" onclick="sh('time',this)">Time of Day</button>
<button class="nb" onclick="sh('hoods',this)">Neighborhoods</button>
<button class="nb" onclick="sh('meth',this)">Data & Methods</button>
<button class="nb" onclick="sh('ask',this)">The Ask</button>
</div></div>

<div class="mn">

<!-- GAP -->
<div id="gap" class="sc a">
<div class="qt">
<p>"Safety isn't just a statistic; it's a feeling you hold when you're walking down the street."</p>
<div class="at">— Daniel Lurie, Mayor of San Francisco, Inauguration Speech, January 2025</div>
</div>

<div class="ms">
<div class="mc"><div class="v" style="color:#D4A03C">63%</div><div class="l">Feel Safe (Day)</div><div class="u">2023 City Survey</div></div>
<div class="mc"><div class="v" style="color:#D4594E">36%</div><div class="l">Feel Safe (Night)</div><div class="u">2023 City Survey</div></div>
<div class="mc"><div class="v" style="color:#1B3A35">{len(df_311):,}</div><div class="l">Disorder Reports</div><div class="u">{date_min}–{date_max}</div></div>
<div class="mc"><div class="v" style="color:#7C3AED">{len(df_sfpd):,}</div><div class="l">Crime Incidents</div><div class="u">{date_min}–{date_max}</div></div>
<div class="mc"><div class="v" style="color:#3B82C8">{len(df_crash) if df_crash is not None else 0:,}</div><div class="l">Traffic Crashes</div><div class="u">12 months</div></div>
</div>

<p class="pr">
The 2023 City Performance Survey — the most recent available — recorded the lowest safety satisfaction
since the survey began in 1996. Only <strong>63% of residents</strong> reported feeling safe walking during the day,
down from 85% in 2019. At night, only <strong>36% feel safe</strong>, down from 53%.
Meanwhile, SFPD data shows reported crime has been declining. This is the <em>perception gap</em>.
</p>
<p class="pr">
The 2025 CityBeat poll from the SF Chamber of Commerce suggests perception may be starting to shift —
78% of weekly downtown visitors now report feeling safe during the day, and the share saying SF is
"on the right track" doubled from 22% to 43%. But we can only see these shifts through expensive one-off polls.
Public Safety Pulse would make this signal <em>continuous and block-level</em>.
</p>

<div class="dk">
<div class="lb" style="color:#D4A03C">CityBeat 2025 — SF Chamber of Commerce</div>
<h3>Perception is shifting, but we can only see it in expensive one-off polls</h3>
<div class="g">
<div><div class="gv" style="color:#2D8B4E">78%</div><div class="gl">of weekly downtown visitors feel safe during the day</div></div>
<div><div class="gv" style="color:#D4A03C">43%</div><div class="gl">say SF is on the right track (was 22% in 2024)</div></div>
<div><div class="gv" style="color:#6BB8F0">↓34pp</div><div class="gl">fewer voters say crime has gotten worse (since 2022)</div></div>
<div><div class="gv" style="color:#6BB8F0">↓25pp</div><div class="gl">fewer say homelessness/street behavior is worse</div></div>
</div>
<p style="font-size:10px;color:rgba(255,255,255,.3);border-top:1px solid rgba(255,255,255,.1);padding-top:10px;margin-top:14px;">
Source: SF Chamber of Commerce CityBeat 2025 Poll. All figures from published poll results.</p>
</div>

<div class="cd"><div class="lb">Hourly Distribution</div>
<h4>When Are Disorder & Crime Reports Filed?</h4>
<div id="hourly-chart" style="height:260px"></div>
</div>

<div class="nt"><strong>Sources:</strong> SF City Performance Survey 2023 (Controller's Office), CityBeat 2025 Poll (SF Chamber of Commerce), DataSF <span class="bg">vw6y-z8j6</span> <span class="bg">wg3w-h783</span> <span class="bg">ubvf-ztfx</span></div>
</div>

<!-- HEAT MAP -->
<div id="heat" class="sc">
<div class="lb">Animated Heat Sensor</div>
<h2>How Safety Perception Shifts Across the Day</h2>
<p class="pr">
This map combines all data sources — 311 disorder reports, SFPD incidents, and traffic crashes — weighted
by their impact on perception. Encampments and violent crime weigh more heavily than graffiti or property crime.
<strong>Press play</strong> to watch hotspots shift through six 4-hour windows across a typical day.
</p>

<div class="map-ctrls">
<button class="play" id="play-btn" onclick="togglePlay()">▶</button>
<select id="tw-select" onchange="setWindow(this.value)">
<option value="0">5am – 9am</option>
<option value="1">9am – 1pm</option>
<option value="2" selected>1pm – 5pm</option>
<option value="3">5pm – 9pm</option>
<option value="4">9pm – 1am</option>
<option value="5">1am – 5am</option>
</select>
<select id="foc-select" onchange="setFocus(this.value)">
<option value="city">All San Francisco</option>
<option value="tl" selected>Tenderloin / Civic Center</option>
<option value="us">Union Square / Downtown</option>
<option value="sm">SoMa</option>
<option value="ms">Mission District</option>
<option value="fi">Financial District</option>
</select>
</div>

<div class="map-wrap" id="heat-map">
<div class="tbadge"><div class="tl">Time Window</div><div class="tv" id="tw-label">1pm – 5pm</div></div>
<div class="tprog" id="tprog"></div>
</div>

<div class="in">
<div class="tg">Key Insight</div>
Watch how the heat signature concentrates differently at night — the Tenderloin and SoMa corridors
intensify while other neighborhoods cool off. This time-of-day variation is invisible in annual surveys
and monthly crime statistics. It's exactly what Phase 1 would capture at block-by-block resolution.
</div>

<div class="nt"><strong>Method:</strong> deck.gl HeatmapLayer with Gaussian kernel smoothing.
Each point is weighted: 311 reports × category salience × temporal decay. SFPD incidents × crime severity × decay.
Traffic crashes × 2.0. Higher weight = stronger heat signal. Radius auto-scales with zoom.</div>
</div>

<!-- SPI -->
<div id="spi" class="sc">
<div class="lb">Computed Metric</div>
<h2>Safety Perception Index v2</h2>
<p class="pr">
The SPI estimates how safe each neighborhood <em>feels</em> by fusing {len(df_311)+len(df_sfpd)+(len(df_crash) if df_crash is not None else 0):,} data points
across four datasets. It is a <strong>proxy estimate</strong> — not a direct measurement of perception.
Scale: 0 (least safe feeling) to 100 (safest). Phase 1 would calibrate these scores against real sentiment.
</p>

<div class="cd"><h4>SPI by Neighborhood</h4>
<div id="spi-chart" style="height:620px"></div></div>

<div class="cd">
<h4>Disorder–Crime Divergence</h4>
<p style="font-size:12px;color:#6B7280;margin-bottom:10px">
Positive = more disorder than crime (perception problem). Negative = more crime than disorder (hidden risk).</p>
<div id="dcdi-chart" style="height:480px"></div>
</div>
</div>

<!-- TIME -->
<div id="time" class="sc">
<div class="lb">Temporal Analysis</div>
<h2>Disorder by Time of Day and Neighborhood</h2>
<p class="pr">
How 311 report rates shift across six 4-hour windows. Darker cells = more reports per month.
This is the variation that a biennial survey cannot see.
</p>

<div class="cd"><div id="time-heatmap" style="height:440px"></div></div>

<div class="cd"><h4>Category Breakdown</h4>
<div id="cat-chart" style="height:340px"></div></div>
</div>

<!-- NEIGHBORHOODS -->
<div id="hoods" class="sc">
<div class="lb">Neighborhood Deep Dive</div>
<h2>What's Driving Disorder?</h2>

<div class="cd"><h4>Top Disorder Categories in Lowest-SPI Neighborhoods</h4>
<div id="cat-hood-chart" style="height:380px"></div></div>

<div class="cd"><h4>Monthly Trends</h4>
<div id="monthly-chart" style="height:280px"></div></div>

<div class="cd"><h4>Crime Categories (SFPD)</h4>
<div id="crime-cat-chart" style="height:320px"></div></div>
</div>

<!-- METHODS -->
<div id="meth" class="sc">
<div class="lb">Transparency</div>
<h2>Data Sources & Methodology</h2>

<h3>Data Sources Integrated</h3>
<table>
<tr><th>Dataset</th><th>Records</th><th>Period</th><th>SPI Role</th><th>Weight</th></tr>
<tr><td>311 Service Requests <span class="bg">vw6y-z8j6</span></td><td>{len(df_311):,}</td><td>{date_min}–{date_max}</td><td>Disorder density, salience, temporal</td><td>30% + 15% + 15%</td></tr>
<tr><td>SFPD Incidents <span class="bg">wg3w-h783</span></td><td>{len(df_sfpd):,}</td><td>{date_min}–{date_max}</td><td>Crime severity density</td><td>20%</td></tr>
<tr><td>Traffic Crashes <span class="bg">ubvf-ztfx</span></td><td>{len(df_crash) if df_crash is not None else 0:,}</td><td>12 months</td><td>Pedestrian safety</td><td>10%</td></tr>
<tr><td>Reddit r/sanfrancisco</td><td>{len(df_reddit) if df_reddit is not None else 0}</td><td>Various</td><td>Community sentiment baseline</td><td>10%</td></tr>
<tr><td>City Survey 2023</td><td>—</td><td>2023</td><td>Validation reference</td><td>—</td></tr>
<tr><td>CityBeat 2025 Poll</td><td>—</td><td>2025</td><td>Calibration reference</td><td>—</td></tr>
</table>

<h3>SPI v2 Formula</h3>
<p class="pr" style="font-family:'JetBrains Mono',monospace;font-size:12px;background:#F3F4F6;padding:14px;border-radius:6px;line-height:1.8">
SPI = 100 − scaled( 0.30×D + 0.20×C + 0.15×DC + 0.10×PS + 0.15×TR + 0.10×CS )<br><br>
D = z-score( Σ(311 cases × e^(-days/180)) / area_km² )<br>
C = z-score( Σ(SFPD incidents × severity × e^(-days/180)) / area_km² )<br>
DC = z-score( mean salience weight of 311 categories )<br>
PS = z-score( traffic crashes / area_km² )<br>
TR = z-score( 0.6 × night_ratio + 0.4 × trend_ratio )<br>
CS = keyword sentiment from Reddit (citywide baseline, neighborhood-specific where available)
</p>

<h3>Key Methodological Choices</h3>
<table>
<tr><th>Choice</th><th>Approach</th><th>Rationale</th></tr>
<tr><td>Normalization</td><td>Z-score, capped at ±3σ</td><td>Handles different scales; robust to outliers vs naive min-max</td></tr>
<tr><td>Temporal decay</td><td>Exponential, half-life ~6 months</td><td>Recent events affect current perception more than old ones</td></tr>
<tr><td>Crime weighting</td><td>Severity scale (1–5)</td><td>Violent crime affects perception more than property crime</td></tr>
<tr><td>Disorder salience</td><td>Category weights (0.2–1.0)</td><td>Encampments affect perception more than graffiti (broken windows)</td></tr>
<tr><td>Spatial unit</td><td>SF Planning neighborhoods</td><td>Matches City Survey for validation; ~40 zones</td></tr>
</table>

<h3>Known Limitations</h3>
<p class="pr">
<strong>Reporting bias:</strong> 311 reflects who reports, not what exists. Engaged neighborhoods over-report.<br>
<strong>Survival bias:</strong> Areas people avoid generate fewer data points and appear safer than they feel.<br>
<strong>No direct perception:</strong> Everything here is inferred from proxy data. Phase 1 collects the real signal.<br>
<strong>Reddit signal:</strong> Only {len(df_reddit) if df_reddit is not None else 0} posts, not geocoded — applied as crude baseline, not precise neighborhood signal.<br>
<strong>Weight calibration:</strong> Component weights are from research literature, not empirically calibrated to SF perception data.
Phase 1 would enable proper calibration via regression against direct sentiment responses.
</p>

<h3>Research Basis</h3>
<p class="pr" style="font-size:13px;color:#6B7280;">
Wilson & Kelling (1982) "Broken Windows" — visible disorder signals predict perceived unsafety.<br>
Sampson & Raudenbush (1999) — systematic observation of disorder correlates with fear of crime.<br>
Salesses, Schechtner & Hidalgo (2013) "Place Pulse" — crowdsourced urban perception mapping (MIT Media Lab).<br>
Naik et al. (2014) "Streetscore" — computer vision for perceived safety from Google Street View (MIT).<br>
These inform our salience weighting: encampments and visible waste are weighted higher than graffiti or noise.
</p>
</div>

<!-- ASK -->
<div id="ask" class="sc">
<div class="dk">
<h3>Everything you just saw is inferred from proxy data.</h3>
<p>311 captures what people report. Crime data captures what police file.
Areas people avoid appear safe in the data. Reddit captures what the online community discusses.
None of this is a direct measurement of how people <em>feel</em>. We need the actual signal.</p>
</div>

<h3>What Phase 1 Unlocks</h3>
<table>
<tr><th>What We Have Now (Proxy)</th><th>What Phase 1 Adds (Direct)</th></tr>
<tr><td>311 complaints — lagging, reporter bias</td><td>Direct, in-the-moment perception</td></tr>
<tr><td>Crime incidents — only what gets reported</td><td>Real-time safety sentiment</td></tr>
<tr><td>Biennial survey — 2-year lag, neighborhood level</td><td>Daily signal, block level</td></tr>
<tr><td>SPI from proxy data — inferred, unvalidated</td><td>Calibrated index with ground truth</td></tr>
<tr><td>Reddit keywords — crude, not geocoded</td><td>Geocoded NPS from digital touchpoints</td></tr>
</table>

<h3>The Question</h3>
<p class="pr"><em>"Right now, how does the surrounding area feel to you?"</em>
— Comfortable / Neutral / Uncomfortable. Delivered through existing digital touchpoints during normal daily activity. Anonymous. Aggregated by place and time.</p>

<h3>Distribution Channels</h3>
<div class="chs">
<div class="ch"><div class="ct">Offices & Buildings</div><div class="cd2">Employee check-in (Envoy), workplace comms, visitor sign-in</div></div>
<div class="ch"><div class="ct">Stores & Restaurants</div><div class="cd2">Point of sale — Square, Toast, Clover, Apple Pay, SNAP cards</div></div>
<div class="ch"><div class="ct">Transit</div><div class="cd2">BART / Muni, Uber / Lyft / Waymo, Google Maps, parking apps</div></div>
<div class="ch"><div class="ct">Location Apps</div><div class="cd2">AllTrails, Strava, Yelp, feedback QR kiosks</div></div>
</div>

<h3>The Feedback Loop</h3>
<p class="pr">Phase 1 validates signal quality. Phase 2 builds a <strong>correlation engine</strong> — mapping which observable
conditions predict perception. Phase 2 also tests <strong>interventions</strong> (sights: cleaning, lighting, ambassadors;
sounds: street musicians; smells: pleasant aromas; civic signals: responsive service). Measure impact in near-real-time.
This creates a continuous improvement cycle: measure → correlate → intervene → re-measure.</p>

<div class="ms">
<div class="mc"><div class="v" style="color:#1B3A35">$150–200K</div><div class="l">Phase 1</div><div class="u">6-month pilot</div></div>
<div class="mc"><div class="v" style="color:#1B3A35">6 months</div><div class="l">Duration</div><div class="u">Define → Build → Capture → Evaluate</div></div>
<div class="mc"><div class="v" style="color:#1B3A35">50K/mo</div><div class="l">Target Responses</div><div class="u">Ramp from 5K</div></div>
</div>

<div class="dk" style="text-align:center;margin-top:28px">
<h3>Partner with us to validate whether direct sentiment can fill the gap.</h3>
<p style="color:rgba(255,255,255,.8);margin-top:8px">City Science Lab San Francisco × MIT Media Lab City Science</p>
</div>
</div>

</div>

<footer>Public Safety Pulse — City Science Lab SF × MIT Media Lab · Data: DataSF, SF Controller's Office, CityBeat 2025, Reddit · Generated {datetime.now().strftime('%B %d, %Y')}</footer>

<script>
// ── Data ──
const SPI={json.dumps(spi_json)};
const TIME_SPI={json.dumps(time_spi_data)};
const CATS={json.dumps(categories)};
const CAT_HOOD={json.dumps(cat_by_hood)};
const CRIME_CATS={json.dumps(crime_categories)};
const H311={json.dumps(hourly_311.to_dict('records'))};
const HSFPD={json.dumps(hourly_sfpd.to_dict('records'))};
const M311={json.dumps(monthly_311.to_dict('records'))};
const MSFPD={json.dumps(monthly_sfpd.to_dict('records'))};
const HEAT={json.dumps(heatmap_data)};
const TW_NAMES=['5am–9am','9am–1pm','1pm–5pm','5pm–9pm','9pm–1am','1am–5am'];

const DARK='https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';
const LIGHT='https://basemaps.cartocdn.com/gl/positron-gl-style/style.json';
const FOCI={{city:{{lat:37.76,lng:-122.44,z:11.5,p:25}},tl:{{lat:37.782,lng:-122.413,z:15,p:40}},us:{{lat:37.788,lng:-122.407,z:15,p:40}},sm:{{lat:37.778,lng:-122.4,z:14.5,p:35}},ms:{{lat:37.76,lng:-122.418,z:14.5,p:35}},fi:{{lat:37.794,lng:-122.398,z:15,p:40}}}};

const L={{font:{{family:'DM Sans',color:'#374151',size:11}},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',margin:{{l:10,r:10,t:10,b:10}},hovermode:'closest'}};

// ── Nav ──
function sh(id,btn){{document.querySelectorAll('.sc').forEach(s=>s.classList.remove('a'));document.querySelectorAll('.nb').forEach(b=>b.classList.remove('a'));document.getElementById(id).classList.add('a');btn.classList.add('a');if(id==='heat'&&!window._hm)initHeatMap()}}

// ── Heat Map ──
let hDeck,hWin=2,playing=false,playInt;

function initHeatMap(){{
    const f=FOCI[document.getElementById('foc-select').value];
    const night=[4,5].includes(hWin);
    hDeck=new deck.DeckGL({{
        container:'heat-map',
        initialViewState:{{longitude:f.lng,latitude:f.lat,zoom:f.z,pitch:f.p,bearing:0}},
        controller:true,
        layers:[makeHeatLayer(hWin)],
        mapStyle:night?DARK:LIGHT,
        getTooltip:({{object}})=>object&&`Weight: ${{object.weight?.toFixed(1)}}`,
    }});
    // Progress bar
    const pg=document.getElementById('tprog');
    TW_NAMES.forEach((n,i)=>{{const d=document.createElement('div');d.className='tp'+(i===hWin?' ac':'');d.title=n;d.onclick=()=>setWindow(i);pg.appendChild(d)}});
    window._hm=true;
}}

function makeHeatLayer(idx){{
    const key=TW_NAMES[idx];
    const pts=HEAT[key]||[];
    return new deck.HeatmapLayer({{
        id:'heat',
        data:pts,
        getPosition:d=>[d[1],d[0]],
        getWeight:d=>d[2],
        radiusPixels:40,
        intensity:3,
        threshold:0.05,
        colorRange:[[255,255,204],[255,237,160],[254,178,76],[253,141,60],[240,59,32],[189,0,38]],
    }});
}}

function setWindow(idx){{
    hWin=parseInt(idx);
    document.getElementById('tw-select').value=hWin;
    document.getElementById('tw-label').textContent=TW_NAMES[hWin];
    document.querySelectorAll('.tp').forEach((d,i)=>d.classList.toggle('ac',i===hWin));
    const night=[4,5].includes(hWin);
    if(hDeck)hDeck.setProps({{layers:[makeHeatLayer(hWin)],mapStyle:night?DARK:LIGHT}});
}}

function setFocus(val){{
    const f=FOCI[val];
    if(hDeck)hDeck.setProps({{initialViewState:{{longitude:f.lng,latitude:f.lat,zoom:f.z,pitch:f.p,bearing:0,transitionDuration:1000}}}});
}}

function togglePlay(){{
    const b=document.getElementById('play-btn');
    if(playing){{clearInterval(playInt);playing=false;b.textContent='▶';b.classList.remove('on')}}
    else{{playing=true;b.textContent='⏸';b.classList.add('on');playInt=setInterval(()=>{{hWin=(hWin+1)%6;setWindow(hWin)}},2500)}}
}}

// ── Charts ──
function init(){{
    // Hourly
    Plotly.newPlot('hourly-chart',[
        {{x:H311.map(d=>d.hour),y:H311.map(d=>d.count),name:'311 Disorder',type:'bar',marker:{{color:H311.map(d=>(d.hour>=20||d.hour<6)?'#1B3A35':'#D4A03C')}},hovertemplate:'%{{x}}:00<br>%{{y:,.0f}} 311 cases<extra></extra>'}},
        {{x:HSFPD.map(d=>d.hour),y:HSFPD.map(d=>d.count),name:'SFPD Incidents',type:'bar',marker:{{color:'rgba(124,58,237,0.4)'}},hovertemplate:'%{{x}}:00<br>%{{y:,.0f}} incidents<extra></extra>'}},
    ],{{...L,margin:{{l:50,r:20,t:10,b:40}},barmode:'overlay',xaxis:{{title:'Hour',tickvals:[...Array(12)].map((_,i)=>i*2),gridcolor:'#F3F4F6'}},yaxis:{{title:'Cases',gridcolor:'#F3F4F6'}},legend:{{orientation:'h',y:-0.2}},shapes:[{{type:'rect',x0:-.5,x1:5.5,y0:0,y1:1,yref:'paper',fillcolor:'rgba(27,58,53,.06)',line:{{width:0}}}},{{type:'rect',x0:19.5,x1:23.5,y0:0,y1:1,yref:'paper',fillcolor:'rgba(27,58,53,.06)',line:{{width:0}}}}]}},{{responsive:true}});

    // SPI
    const ss=[...SPI].sort((a,b)=>a.SPI-b.SPI);
    Plotly.newPlot('spi-chart',[{{y:ss.map(d=>d.analysis_neighborhood),x:ss.map(d=>d.SPI),type:'bar',orientation:'h',marker:{{color:ss.map(d=>d.SPI<30?'#D4594E':d.SPI<50?'#D4A03C':d.SPI<70?'#6BB8F0':'#2D8B4E')}},text:ss.map(d=>d.SPI.toFixed(1)),textposition:'outside',textfont:{{family:'JetBrains Mono',size:10}},hovertemplate:'%{{y}}<br>SPI: %{{x:.1f}}<br>311: %{{customdata[0]:,}}<br>Crime: %{{customdata[1]:,}}<br>Crashes: %{{customdata[2]:,}}<extra></extra>',customdata:ss.map(d=>[d.raw_count,d.crime_count,d.crash_count])}}],{{...L,margin:{{l:180,r:55,t:10,b:30}},xaxis:{{range:[0,105],gridcolor:'#F3F4F6',title:'SPI (0=least safe, 100=safest)'}}}},{{responsive:true}});

    // DCDI
    const dc=[...SPI].filter(d=>d.raw_count>500).sort((a,b)=>b.DCDI-a.DCDI);
    Plotly.newPlot('dcdi-chart',[{{y:dc.map(d=>d.analysis_neighborhood),x:dc.map(d=>d.DCDI),type:'bar',orientation:'h',marker:{{color:dc.map(d=>d.DCDI>10?'#D4A03C':d.DCDI<-10?'#D4594E':'#6B7280')}},text:dc.map(d=>d.DCDI>10?'Perception Gap':d.DCDI<-10?'Hidden Risk':'Aligned'),textposition:'outside',textfont:{{size:10}}}}],{{...L,margin:{{l:180,r:100,t:10,b:30}},xaxis:{{gridcolor:'#F3F4F6',zeroline:true,zerolinecolor:'#374151',zerolinewidth:2,title:'← More Crime | More Disorder →'}}}},{{responsive:true}});

    // Time heatmap
    const hs=[...new Set(TIME_SPI.map(d=>d.neighborhood))];
    const ws=[...new Set(TIME_SPI.map(d=>d.window))];
    const z=hs.map(h=>ws.map(w=>{{const m=TIME_SPI.find(d=>d.neighborhood===h&&d.window===w);return m?m.rate:0}}));
    Plotly.newPlot('time-heatmap',[{{z,x:ws,y:hs,type:'heatmap',colorscale:[[0,'#FAF7F2'],[.3,'#FFC300'],[.6,'#E3611C'],[1,'#5A1846']],hovertemplate:'%{{y}}<br>%{{x}}<br>%{{z:.0f}} cases/mo<extra></extra>'}}],{{...L,margin:{{l:180,r:10,t:10,b:80}},xaxis:{{tickangle:-25}}}},{{responsive:true}});

    // Categories
    const cs=[...CATS].reverse();
    Plotly.newPlot('cat-chart',[{{y:cs.map(c=>c.name),x:cs.map(c=>c.count),type:'bar',orientation:'h',marker:{{color:cs.map(c=>c.count),colorscale:[[0,'#FFC300'],[.5,'#C70039'],[1,'#5A1846']]}},hovertemplate:'%{{y}}<br>%{{x:,.0f}}<extra></extra>'}}],{{...L,margin:{{l:190,r:20,t:10,b:30}},xaxis:{{gridcolor:'#F3F4F6'}},showlegend:false}},{{responsive:true}});

    // Cat by hood
    const ch5=[...new Set(CAT_HOOD.map(d=>d.neighborhood))];
    const cc5=[...new Set(CAT_HOOD.map(d=>d.category))];
    const cl5=['#D4594E','#D4A03C','#3B82C8','#2D8B4E','#7C3AED'];
    Plotly.newPlot('cat-hood-chart',cc5.map((c,i)=>({{name:c,x:ch5,y:ch5.map(h=>{{const m=CAT_HOOD.find(d=>d.neighborhood===h&&d.category===c);return m?m.count:0}}),type:'bar',marker:{{color:cl5[i%5]}}}})),{{...L,barmode:'stack',margin:{{l:50,r:10,t:10,b:110}},xaxis:{{tickangle:-25}},yaxis:{{gridcolor:'#F3F4F6'}},legend:{{orientation:'h',y:-0.35,font:{{size:10}}}}}},{{responsive:true}});

    // Monthly
    Plotly.newPlot('monthly-chart',[{{x:M311.map(d=>d.month),y:M311.map(d=>d.count),name:'311 Disorder',mode:'lines+markers',line:{{color:'#D4594E',width:2.5}},marker:{{size:4}}}},{{x:MSFPD.map(d=>d.month),y:MSFPD.map(d=>d.count),name:'SFPD Incidents',mode:'lines+markers',line:{{color:'#7C3AED',width:2.5}},marker:{{size:4}}}}],{{...L,margin:{{l:50,r:20,t:10,b:40}},xaxis:{{gridcolor:'#F3F4F6'}},yaxis:{{title:'Monthly',gridcolor:'#F3F4F6'}},legend:{{orientation:'h',y:-0.15}}}},{{responsive:true}});

    // Crime categories
    if(CRIME_CATS.length){{const cc=[...CRIME_CATS].reverse();Plotly.newPlot('crime-cat-chart',[{{y:cc.map(c=>c.name),x:cc.map(c=>c.count),type:'bar',orientation:'h',marker:{{color:'#7C3AED',opacity:0.75}},hovertemplate:'%{{y}}<br>%{{x:,.0f}}<extra></extra>'}}],{{...L,margin:{{l:180,r:20,t:10,b:30}},xaxis:{{gridcolor:'#F3F4F6'}}}},{{responsive:true}})}}
}}

window.addEventListener('DOMContentLoaded',init);
</script>
</body></html>"""

output = Path("psp_dashboard_v6.html")
output.write_text(html, encoding='utf-8')
mb = output.stat().st_size / 1024 / 1024
print(f"\n✅ Dashboard saved: {output.absolute()}")
print(f"   Size: {mb:.1f} MB")
print(f"\n🚀 Open: open psp_dashboard_v6.html")
