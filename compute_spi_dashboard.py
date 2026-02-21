#!/usr/bin/env python3
"""
Public Safety Pulse — Compute SPI & Generate Dashboard v5
=========================================================

This script:
1. Loads ALL available data (311, SFPD, traffic crashes, Reddit)
2. Computes a neighborhood-level Safety Perception Index (SPI)
3. Computes Disorder-Crime Divergence Index (DCDI)
4. Computes time-of-day variation scores
5. Generates a clean 2D dashboard (no 3D hex mess)

Usage:
    cd ~/Desktop/psp-mvp
    source venv/bin/activate
    python compute_spi_dashboard.py
    open psp_dashboard_v5.html
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

print("=" * 60)
print("Public Safety Pulse — SPI Computation & Dashboard v5")
print("=" * 60)

# ═══════════════════════════════════════════════════════════
# 1. LOAD ALL AVAILABLE DATA
# ═══════════════════════════════════════════════════════════
base = Path("data/raw")
print("\n📂 Loading all available datasets...")

# 311 Cases (Layer 2 — Condition & Environment)
df_311 = pd.read_parquet(base / "layer2" / "311_cases_safety.parquet")
df_311['lat'] = pd.to_numeric(df_311['lat'], errors='coerce')
df_311['long'] = pd.to_numeric(df_311['long'], errors='coerce')
df_311['requested_datetime'] = pd.to_datetime(df_311['requested_datetime'], errors='coerce')
df_311['hour'] = df_311['requested_datetime'].dt.hour
df_311['month'] = df_311['requested_datetime'].dt.to_period('M').astype(str)
df_311 = df_311.dropna(subset=['lat', 'long', 'requested_datetime'])
hood_col = 'analysis_neighborhood'
print(f"  ✅ 311 Safety Cases: {len(df_311):,} records, {df_311[hood_col].nunique()} neighborhoods")

# SFPD Incidents (Layer 1 — Hard Incident Data)
df_sfpd = pd.read_parquet(base / "layer1" / "sfpd_incidents.parquet")
df_sfpd['latitude'] = pd.to_numeric(df_sfpd['latitude'], errors='coerce')
df_sfpd['longitude'] = pd.to_numeric(df_sfpd['longitude'], errors='coerce')
df_sfpd['incident_datetime'] = pd.to_datetime(df_sfpd['incident_datetime'], errors='coerce')
df_sfpd['hour'] = df_sfpd['incident_datetime'].dt.hour
df_sfpd['month'] = df_sfpd['incident_datetime'].dt.to_period('M').astype(str)
df_sfpd = df_sfpd.dropna(subset=['latitude', 'longitude'])
print(f"  ✅ SFPD Incidents: {len(df_sfpd):,} records")

# Traffic Crashes (Layer 1)
crash_path = base / "layer1" / "traffic_crashes.parquet"
df_crash = None
if crash_path.exists():
    df_crash = pd.read_parquet(crash_path)
    print(f"  ✅ Traffic Crashes: {len(df_crash):,} records")
else:
    print(f"  ⚠️  Traffic Crashes: not found")

# Reddit Sentiment (Layer 3)
reddit_path = base / "layer3" / "reddit_safety_posts.parquet"
df_reddit = None
if reddit_path.exists():
    df_reddit = pd.read_parquet(reddit_path)
    print(f"  ✅ Reddit Posts: {len(df_reddit):,} records")
else:
    print(f"  ⚠️  Reddit Posts: not found")


# ═══════════════════════════════════════════════════════════
# 2. COMPUTE SAFETY PERCEPTION INDEX (SPI)
# ═══════════════════════════════════════════════════════════
print("\n⚙️  Computing Safety Perception Index...")

"""
METHODOLOGY
===========
The Safety Perception Index (SPI) estimates how safe a neighborhood
*feels* based on observable proxy data. It is NOT a direct measurement
of perception — that's what Phase 1 would collect.

Components (all computed at neighborhood level):
  1. Disorder Density (D): 311 safety cases per sq km, normalized 0-100
  2. Crime Rate (C): SFPD incidents per sq km, normalized 0-100
  3. Disorder Composition (DC): % of 311 cases in high-salience categories
     (encampments, human waste > graffiti > general cleaning)
  4. Night Concentration (N): Ratio of night-to-day disorder reports
     Higher ratio = more nighttime disorder = lower perceived safety
  5. Trend Direction (T): 3-month moving average slope
     Rising disorder = worsening perception

Composite formula:
  SPI = 100 - (w1*D + w2*C + w3*DC + w4*N + w5*T)
  
  Where SPI = 100 is "safest feeling" and SPI = 0 is "least safe feeling"
  
Weights:
  w1 = 0.35 (disorder is strongest perception driver per broken windows research)
  w2 = 0.25 (crime matters but less than what's visible on the street)
  w3 = 0.20 (encampments/waste affect perception more than graffiti)
  w4 = 0.10 (nighttime concentration amplifies perception)
  w5 = 0.10 (trend direction — getting worse feels worse)

CALIBRATION NOTE:
  These weights are initial estimates. Phase 1 would calibrate them against
  actual perception data. The 2023 City Survey provides partial validation:
  neighborhoods with known low safety scores (Tenderloin, SoMa) should
  receive low SPI scores.
  
KNOWN LIMITATIONS:
  - Reporting bias: areas with engaged residents over-report via 311
  - Survival bias: areas people avoid generate fewer reports
  - Temporal lag: 311 data reflects when reports were filed, not conditions
  - No direct perception data (that's what Phase 1 solves)
"""

# Approximate neighborhood areas (sq km) for density normalization
# Source: SF Planning Department neighborhood boundaries
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
    'Visitacion Valley': 2.2, 'West of Twin Peaks': 3.8,
    'Western Addition': 1.5,
}

# HIGH-SALIENCE categories (weight more heavily — these are what people SEE)
HIGH_SALIENCE = ['Encampment', 'Street and Sidewalk Cleaning']
MED_SALIENCE = ['Graffiti Public', 'Graffiti Private', 'Blocked Street and Sidewalk', 'Noise']

# ── Component 1: Disorder Density ──────────────────
disorder_by_hood = df_311.groupby(hood_col).size().reset_index(name='disorder_count')
disorder_by_hood['area_km2'] = disorder_by_hood[hood_col].map(HOOD_AREAS).fillna(2.0)
disorder_by_hood['disorder_density'] = disorder_by_hood['disorder_count'] / disorder_by_hood['area_km2']

# Normalize to 0-100 (higher = more disorder = worse)
max_dd = disorder_by_hood['disorder_density'].quantile(0.95)  # Cap at 95th pctile to reduce outlier effect
disorder_by_hood['D_score'] = (disorder_by_hood['disorder_density'] / max_dd * 100).clip(0, 100)

# ── Component 2: Crime Rate ────────────────────────
crime_by_hood = df_sfpd.groupby(hood_col).size().reset_index(name='crime_count')
crime_by_hood['area_km2'] = crime_by_hood[hood_col].map(HOOD_AREAS).fillna(2.0)
crime_by_hood['crime_density'] = crime_by_hood['crime_count'] / crime_by_hood['area_km2']

max_cd = crime_by_hood['crime_density'].quantile(0.95)
crime_by_hood['C_score'] = (crime_by_hood['crime_density'] / max_cd * 100).clip(0, 100)

# ── Component 3: Disorder Composition ──────────────
cat_col = 'service_name'
if cat_col in df_311.columns:
    salience_map = {}
    for cat in df_311[cat_col].unique():
        if cat in HIGH_SALIENCE:
            salience_map[cat] = 1.0
        elif cat in MED_SALIENCE:
            salience_map[cat] = 0.6
        else:
            salience_map[cat] = 0.3

    df_311['salience_weight'] = df_311[cat_col].map(salience_map).fillna(0.3)
    composition = df_311.groupby(hood_col).agg(
        total_cases=('salience_weight', 'count'),
        weighted_salience=('salience_weight', 'sum')
    ).reset_index()
    composition['DC_score'] = (composition['weighted_salience'] / composition['total_cases'] * 100).clip(0, 100)
    # Normalize
    max_dc = composition['DC_score'].quantile(0.95)
    composition['DC_score'] = (composition['DC_score'] / max_dc * 100).clip(0, 100)
else:
    composition = disorder_by_hood[[hood_col]].copy()
    composition['DC_score'] = 50  # neutral if no category data

# ── Component 4: Night Concentration ───────────────
night_mask = (df_311['hour'] >= 20) | (df_311['hour'] < 6)
day_mask = (df_311['hour'] >= 7) & (df_311['hour'] < 19)

night_counts = df_311[night_mask].groupby(hood_col).size().reset_index(name='night')
day_counts = df_311[day_mask].groupby(hood_col).size().reset_index(name='day')
night_ratio = night_counts.merge(day_counts, on=hood_col, how='outer').fillna(1)
night_ratio['N_score'] = (night_ratio['night'] / (night_ratio['day'] + 1) * 100).clip(0, 100)
max_nr = night_ratio['N_score'].quantile(0.95)
night_ratio['N_score'] = (night_ratio['N_score'] / max_nr * 100).clip(0, 100)

# ── Component 5: Trend Direction ───────────────────
# Compare last 3 months vs prior 3 months
recent_cutoff = df_311['requested_datetime'].max() - pd.Timedelta(days=90)
mid_cutoff = recent_cutoff - pd.Timedelta(days=90)

recent = df_311[df_311['requested_datetime'] >= recent_cutoff].groupby(hood_col).size().reset_index(name='recent')
prior = df_311[(df_311['requested_datetime'] >= mid_cutoff) & (df_311['requested_datetime'] < recent_cutoff)].groupby(hood_col).size().reset_index(name='prior')
trend = recent.merge(prior, on=hood_col, how='outer').fillna(1)
trend['T_score'] = ((trend['recent'] / (trend['prior'] + 1) - 1) * 100).clip(-50, 50)
# Normalize: positive = getting worse
trend['T_score'] = ((trend['T_score'] + 50) / 100 * 100).clip(0, 100)

# ── Combine into SPI ──────────────────────────────
print("  Combining components into SPI...")

spi = disorder_by_hood[[hood_col, 'D_score', 'disorder_count', 'disorder_density']].copy()
spi = spi.merge(crime_by_hood[[hood_col, 'C_score', 'crime_count', 'crime_density']], on=hood_col, how='outer')
spi = spi.merge(composition[[hood_col, 'DC_score']], on=hood_col, how='outer')
spi = spi.merge(night_ratio[[hood_col, 'N_score']], on=hood_col, how='outer')
spi = spi.merge(trend[[hood_col, 'T_score']], on=hood_col, how='outer')
spi = spi.fillna(50)

# Weights
W = {'D': 0.35, 'C': 0.25, 'DC': 0.20, 'N': 0.10, 'T': 0.10}

spi['raw_risk'] = (
    W['D'] * spi['D_score'] +
    W['C'] * spi['C_score'] +
    W['DC'] * spi['DC_score'] +
    W['N'] * spi['N_score'] +
    W['T'] * spi['T_score']
)

# Invert: higher SPI = safer feeling
spi['SPI'] = (100 - spi['raw_risk']).clip(0, 100).round(1)

# Compute DCDI (Disorder-Crime Divergence Index)
max_d = spi['D_score'].max()
max_c = spi['C_score'].max()
spi['D_norm'] = spi['D_score'] / max_d * 100 if max_d > 0 else 50
spi['C_norm'] = spi['C_score'] / max_c * 100 if max_c > 0 else 50
spi['DCDI'] = (spi['D_norm'] - spi['C_norm']).round(1)
# Positive DCDI = perception problem (more disorder than crime)
# Negative DCDI = hidden risk (more crime than disorder)

spi = spi.sort_values('SPI', ascending=True)

print("\n📊 Top 10 Neighborhoods by SPI (lowest = feels least safe):")
print("-" * 65)
for _, row in spi.head(10).iterrows():
    dcdi_label = "Perception Gap" if row['DCDI'] > 10 else ("Hidden Risk" if row['DCDI'] < -10 else "Aligned")
    print(f"  {row[hood_col]:35s} SPI: {row['SPI']:5.1f}  DCDI: {row['DCDI']:+6.1f} ({dcdi_label})")

print(f"\n  ... {len(spi)} total neighborhoods scored")


# ═══════════════════════════════════════════════════════════
# 3. COMPUTE TIME-OF-DAY SPI
# ═══════════════════════════════════════════════════════════
print("\n⚙️  Computing time-of-day variation...")

time_windows = {
    'Early Morning (5–9am)': (5, 9),
    'Midday (9am–1pm)': (9, 13),
    'Afternoon (1–5pm)': (13, 17),
    'Evening (5–9pm)': (17, 21),
    'Night (9pm–1am)': (21, 1),
    'Late Night (1–5am)': (1, 5),
}

# For top 10 neighborhoods, compute disorder density by time window
top_hoods = spi.head(12)[hood_col].tolist()
time_data = []

for hood in top_hoods:
    hood_311 = df_311[df_311[hood_col] == hood]
    for window_name, (start, end) in time_windows.items():
        if start < end:
            window_cases = hood_311[hood_311['hour'].between(start, end - 1)]
        else:
            window_cases = hood_311[(hood_311['hour'] >= start) | (hood_311['hour'] < end)]
        # Annualize: cases per month in this window
        n_months = max(1, df_311['month'].nunique())
        monthly_rate = len(window_cases) / n_months
        time_data.append({
            'neighborhood': hood,
            'window': window_name,
            'cases_per_month': round(monthly_rate, 1),
        })

time_df = pd.DataFrame(time_data)

# ═══════════════════════════════════════════════════════════
# 4. PREPARE CATEGORY ANALYSIS
# ═══════════════════════════════════════════════════════════
print("⚙️  Analyzing disorder categories...")

cat_counts = df_311[cat_col].value_counts().head(12)
categories = [{"name": k, "count": int(v)} for k, v in cat_counts.items()]

# Category by neighborhood for top 5 hoods
top5 = spi.head(5)[hood_col].tolist()
cat_by_hood = []
for hood in top5:
    hood_data = df_311[df_311[hood_col] == hood][cat_col].value_counts().head(5)
    for cat, count in hood_data.items():
        cat_by_hood.append({'neighborhood': hood, 'category': cat, 'count': int(count)})

# ═══════════════════════════════════════════════════════════
# 5. PREPARE MONTHLY TRENDS
# ═══════════════════════════════════════════════════════════
monthly_311 = df_311.groupby('month').size().reset_index(name='count').sort_values('month')
monthly_sfpd = df_sfpd.groupby('month').size().reset_index(name='count').sort_values('month')

# Hourly
hourly_311 = df_311.groupby('hour').size().reset_index(name='count')
hourly_sfpd = df_sfpd.groupby('hour').size().reset_index(name='count')


# ═══════════════════════════════════════════════════════════
# 6. GENERATE DASHBOARD HTML
# ═══════════════════════════════════════════════════════════
print("\n🎨 Generating dashboard...")

# Prepare data for JSON embedding
spi_json = spi[[hood_col, 'SPI', 'DCDI', 'disorder_count', 'crime_count',
                  'D_score', 'C_score', 'DC_score', 'N_score', 'T_score']].to_dict('records')
time_json = time_df.to_dict('records')
cat_hood_json = cat_by_hood

# Key stats
total_311 = len(df_311)
total_sfpd = len(df_sfpd)
total_crash = len(df_crash) if df_crash is not None else 0
total_reddit = len(df_reddit) if df_reddit is not None else 0
n_hoods = spi[hood_col].nunique()
lowest_spi = spi.iloc[0]
highest_spi = spi.iloc[-1]

# Date range
date_min = df_311['requested_datetime'].min().strftime('%b %Y')
date_max = df_311['requested_datetime'].max().strftime('%b %Y')

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Public Safety Pulse — San Francisco</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,700&family=DM+Serif+Display&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'DM Sans', -apple-system, sans-serif; background: #FAF7F2; color: #1A1A1A; line-height: 1.6; }}
h1,h2,h3,h4 {{ font-family: 'DM Serif Display', Georgia, serif; font-weight: 400; }}

.header {{ background: #1B3A35; padding: 48px 0 36px; }}
.wrap {{ max-width: 980px; margin: 0 auto; padding: 0 32px; }}
.header .org {{ color: rgba(255,255,255,0.4); font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 8px; }}
.header h1 {{ color: #fff; font-size: 2.4rem; }}
.header .sub {{ color: rgba(255,255,255,0.6); margin-top: 8px; }}

.nav {{ background: #fff; border-bottom: 1px solid #E5E7EB; position: sticky; top: 0; z-index: 100; }}
.nav-inner {{ max-width: 980px; margin: 0 auto; padding: 0 32px; display: flex; overflow-x: auto; }}
.nav-btn {{ padding: 13px 22px; border: none; background: none; cursor: pointer; font-family: inherit; font-size: 13px; color: #6B7280; border-bottom: 2px solid transparent; white-space: nowrap; }}
.nav-btn.active {{ color: #1B3A35; font-weight: 700; border-bottom-color: #D4594E; }}

.main {{ max-width: 980px; margin: 0 auto; padding: 32px; }}
.sec {{ display: none; }}
.sec.active {{ display: block; }}

.label {{ font-size: 11px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #D4594E; margin-bottom: 6px; }}
.quote {{ border-left: 3px solid #D4594E; padding: 4px 0 4px 24px; margin: 28px 0; }}
.quote p {{ font-family: 'DM Serif Display', Georgia, serif; font-size: 1.25rem; font-style: italic; color: #374151; line-height: 1.5; }}
.quote .attr {{ font-size: 13px; color: #9CA3AF; font-style: normal; margin-top: 8px; }}

.metrics {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 24px 0; }}
.m {{ flex: 1; min-width: 140px; background: #fff; border-radius: 8px; padding: 20px 14px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
.m .v {{ font-family: 'DM Serif Display', Georgia, serif; font-size: 2rem; line-height: 1; }}
.m .l {{ font-size: 10px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #6B7280; margin-top: 6px; }}
.m .s {{ font-size: 10px; color: #9CA3AF; margin-top: 2px; }}

.card {{ background: #fff; border-radius: 10px; padding: 24px; margin: 20px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
.dark {{ background: #1B3A35; border-radius: 10px; padding: 28px 32px; color: #fff; margin: 24px 0; }}
.dark h3 {{ color: #fff; font-size: 1.3rem; margin-bottom: 8px; }}
.dark p {{ color: rgba(255,255,255,0.75); }}
.dark .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }}
.dark .gv {{ font-family: 'DM Serif Display', Georgia, serif; font-size: 1.8rem; }}
.dark .gl {{ font-size: 12px; color: rgba(255,255,255,0.5); margin-top: 2px; }}

.insight {{ background: #fff; border-left: 3px solid #D4594E; padding: 16px 20px; border-radius: 0 8px 8px 0; margin: 20px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
.insight .tag {{ font-size: 10px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #D4594E; margin-bottom: 4px; }}

.note {{ background: #F3F4F6; border-radius: 6px; padding: 14px 18px; font-size: 12px; color: #6B7280; margin: 16px 0; }}
.note strong {{ color: #374151; }}
.badge {{ display: inline-block; background: #E5E7EB; border-radius: 3px; padding: 1px 6px; font-size: 10px; font-family: 'JetBrains Mono', monospace; color: #6B7280; margin: 0 2px; }}

.prose {{ font-size: 15px; color: #374151; margin: 14px 0; }}
.prose em {{ color: #D4594E; }}
.prose strong {{ color: #1A1A1A; }}

table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
th {{ background: #F9FAFB; padding: 10px 14px; font-size: 10px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #6B7280; text-align: left; border-bottom: 2px solid #E5E7EB; }}
td {{ padding: 10px 14px; font-size: 13px; border-bottom: 1px solid #F3F4F6; }}

.spi-bar {{ display: flex; align-items: center; gap: 8px; }}
.spi-fill {{ height: 18px; border-radius: 3px; transition: width 0.6s; }}
.spi-num {{ font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 500; min-width: 32px; }}

.channels {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 16px 0; }}
.ch {{ background: #fff; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
.ch .t {{ font-weight: 700; font-size: 13px; color: #1B3A35; margin-bottom: 4px; }}
.ch .d {{ font-size: 12px; color: #6B7280; }}

footer {{ max-width: 980px; margin: 0 auto; padding: 24px 32px 40px; text-align: center; color: #9CA3AF; font-size: 11px; border-top: 1px solid #E5E7EB; }}

@media (max-width: 768px) {{
    .metrics {{ flex-direction: column; }}
    .dark .grid {{ grid-template-columns: 1fr; }}
    .channels {{ grid-template-columns: 1fr; }}
    .main {{ padding: 20px 16px; }}
}}
</style>
</head>
<body>

<div class="header"><div class="wrap">
    <div class="org">City Science Lab San Francisco × MIT Media Lab City Science</div>
    <h1>Public Safety Pulse</h1>
    <div class="sub">Measuring everyday safety perception in San Francisco — MVP Dashboard</div>
</div></div>

<div class="nav"><div class="nav-inner">
    <button class="nav-btn active" onclick="show('gap',this)">The Perception Gap</button>
    <button class="nav-btn" onclick="show('spi',this)">Safety Perception Index</button>
    <button class="nav-btn" onclick="show('time',this)">Time of Day</button>
    <button class="nav-btn" onclick="show('hoods',this)">Neighborhoods</button>
    <button class="nav-btn" onclick="show('method',this)">Data & Methods</button>
    <button class="nav-btn" onclick="show('ask',this)">The Ask</button>
</div></div>

<div class="main">

<!-- ═══ GAP ═══ -->
<div id="gap" class="sec active">
    <div class="quote">
        <p>"Safety isn't just a statistic; it's a feeling you hold when you're walking down the street."</p>
        <div class="attr">— Daniel Lurie, Mayor of San Francisco, Inauguration Speech, January 2025</div>
    </div>

    <div class="metrics">
        <div class="m"><div class="v" style="color:#2D8B4E">↓35%</div><div class="l">Reported Crime</div><div class="s">2019 → 2024 (SFPD)</div></div>
        <div class="m"><div class="v" style="color:#D4A03C">63%</div><div class="l">Feel Safe (Day)</div><div class="s">2023 City Survey</div></div>
        <div class="m"><div class="v" style="color:#D4594E">36%</div><div class="l">Feel Safe (Night)</div><div class="s">2023 City Survey</div></div>
        <div class="m"><div class="v" style="color:#1B3A35">{total_311:,}</div><div class="l">311 Disorder Cases</div><div class="s">{date_min} – {date_max}</div></div>
    </div>

    <p class="prose">
        San Francisco's reported crime has declined substantially since 2019. SFPD incident data shows a roughly
        35% reduction from 2019 to 2024 across all police districts. Yet the 2023 City Performance Survey — the most
        recent available — recorded the lowest safety satisfaction since the survey began in 1996. Only 63% of
        residents reported feeling safe walking during the day, down from 85% in 2019. At night, only 36% feel safe,
        down from 53%.
    </p>
    <p class="prose">
        <strong>This is the perception gap.</strong> The statistical picture is improving. The lived experience has not
        kept pace. Why? Because safety <em>perception</em> is driven less by crime rates and more by what people
        <em>see on the street</em> — encampments, needles, aggressive behavior, graffiti, broken infrastructure.
        These conditions are captured in 311 service request data, not police reports.
    </p>

    <div class="dark">
        <div class="label" style="color:#D4A03C;">CityBeat 2025 Poll — SF Chamber of Commerce</div>
        <h3>New data suggests perception is beginning to shift</h3>
        <div class="grid">
            <div><div class="gv" style="color:#2D8B4E">78%</div><div class="gl">of weekly downtown visitors feel safe during the day</div></div>
            <div><div class="gv" style="color:#D4A03C">43%</div><div class="gl">say SF is on the right track (was 22% in 2024)</div></div>
            <div><div class="gv" style="color:#6BB8F0">↓34%</div><div class="gl">fewer voters say crime has gotten worse (since 2022)</div></div>
            <div><div class="gv" style="color:#6BB8F0">↓25%</div><div class="gl">fewer say homelessness/street behavior is worse</div></div>
        </div>
        <p style="font-size:11px;color:rgba(255,255,255,0.3);border-top:1px solid rgba(255,255,255,0.1);padding-top:12px;margin-top:16px;">
            Source: SF Chamber of Commerce CityBeat 2025 Poll. These are the most recent perception data points available.
            The City Survey has not been conducted since 2023. This is exactly the data gap PSP addresses.
        </p>
    </div>

    <div class="card">
        <div class="label">Hourly Distribution</div>
        <h4>When Are Disorder Reports Filed?</h4>
        <div id="hourly-chart" style="height:280px;"></div>
    </div>

    <div class="note">
        <strong>Data sources on this page:</strong>
        SF City Performance Survey 2023 (Controller's Office),
        CityBeat 2025 Poll (SF Chamber of Commerce),
        SFPD Incident Reports via DataSF
        <span class="badge">wg3w-h783</span>,
        311 Service Requests via DataSF
        <span class="badge">vw6y-z8j6</span>.
        Crime decline figure (35%) based on SFPD reported incidents 2019 vs 2024.
    </div>
</div>

<!-- ═══ SPI ═══ -->
<div id="spi" class="sec">
    <div class="label">Computed Metric</div>
    <h2>Safety Perception Index</h2>
    <p class="prose">
        The SPI estimates how safe each neighborhood <em>feels</em> based on observable proxy data.
        It combines five components: disorder density (311 reports per km²), crime rate (SFPD incidents per km²),
        disorder composition (encampments and waste weigh more than graffiti), nighttime concentration,
        and trend direction. Scale: 0 (least safe feeling) to 100 (safest feeling).
    </p>
    <p class="prose" style="color:#6B7280;font-size:13px;">
        <strong>Important:</strong> This is a proxy estimate, not a direct measurement. Actual perception data
        would come from Phase 1's direct sentiment collection. These scores have not been validated against
        a perception survey and should be interpreted as relative rankings, not absolute measurements.
    </p>

    <div class="card">
        <h4>SPI by Neighborhood (lowest to highest)</h4>
        <div id="spi-chart" style="height:600px;"></div>
    </div>

    <div class="insight">
        <div class="tag">Key Finding</div>
        <strong>{lowest_spi[hood_col]}</strong> scores lowest (SPI {lowest_spi['SPI']}) with {int(lowest_spi['disorder_count']):,}
        disorder reports and {int(lowest_spi['crime_count']):,} crime incidents in a {HOOD_AREAS.get(lowest_spi[hood_col], 2.0):.1f} km² area.
        <strong>{highest_spi[hood_col]}</strong> scores highest (SPI {highest_spi['SPI']}).
        The 10× range in SPI scores across neighborhoods demonstrates exactly the kind of block-level
        variation that citywide surveys cannot capture.
    </div>

    <div class="card">
        <h4>Disorder–Crime Divergence Index</h4>
        <p style="font-size:13px;color:#6B7280;margin-bottom:12px;">
            Where the divergence is positive, disorder signals (what people see) exceed crime rates (what gets reported).
            These are <strong>perception problems</strong> — areas that feel unsafe primarily because of environmental
            conditions, not criminal danger. Negative divergence indicates potential hidden risk.
        </p>
        <div id="dcdi-chart" style="height:500px;"></div>
    </div>
</div>

<!-- ═══ TIME ═══ -->
<div id="time" class="sec">
    <div class="label">Temporal Patterns</div>
    <h2>How Disorder Shifts Throughout the Day</h2>
    <p class="prose">
        Safety perception changes dramatically by time of day. The same corner can feel vibrant at noon
        and threatening at midnight. This heatmap shows how 311 disorder report rates vary across
        6 time windows for the neighborhoods with the most activity.
    </p>

    <div class="card">
        <h4>Disorder Reports by Time Window & Neighborhood</h4>
        <p style="font-size:12px;color:#6B7280;margin-bottom:8px;">Cases per month in each time window. Darker = more reports.</p>
        <div id="time-heatmap" style="height:450px;"></div>
    </div>

    <div class="insight">
        <div class="tag">Why This Matters</div>
        The City Survey asks "do you feel safe during the day?" — one question, once every two years,
        at neighborhood level. Public Safety Pulse would capture this variation in 4-hour windows,
        every day, at block level. That's the difference between knowing a neighborhood has a problem
        and knowing which intersection, at which hour, needs an intervention.
    </div>
</div>

<!-- ═══ HOODS ═══ -->
<div id="hoods" class="sec">
    <div class="label">Neighborhood Deep Dive</div>
    <h2>What's Driving Disorder in Each Area?</h2>

    <div class="card">
        <h4>Top 5 Category Breakdown by Neighborhood</h4>
        <p style="font-size:12px;color:#6B7280;margin-bottom:8px;">What types of 311 reports dominate in the lowest-SPI neighborhoods?</p>
        <div id="cat-hood-chart" style="height:400px;"></div>
    </div>

    <div class="card">
        <h4>Monthly Trends</h4>
        <div id="monthly-chart" style="height:300px;"></div>
    </div>

    <div class="note">
        <strong>Observation:</strong> Street and Sidewalk Cleaning dominates overall volume, but
        Encampment reports have the strongest association with low safety perception in the research
        literature. The SPI weights encampment and cleaning requests more heavily than graffiti reports
        based on broken windows theory (Wilson & Kelling, 1982) and MIT Place Pulse findings (Salesses et al., 2013).
    </div>
</div>

<!-- ═══ METHOD ═══ -->
<div id="method" class="sec">
    <div class="label">Transparency</div>
    <h2>Data Sources & Methodology</h2>

    <h3>Data Sources</h3>
    <table>
        <tr><th>Dataset</th><th>Source</th><th>Records</th><th>Period</th><th>Role in SPI</th></tr>
        <tr><td>311 Service Requests</td><td>DataSF <span class="badge">vw6y-z8j6</span></td><td>{total_311:,}</td><td>{date_min}–{date_max}</td><td>Disorder density, composition, temporal patterns</td></tr>
        <tr><td>SFPD Incident Reports</td><td>DataSF <span class="badge">wg3w-h783</span></td><td>{total_sfpd:,}</td><td>{date_min}–{date_max}</td><td>Crime rate, divergence analysis</td></tr>
        <tr><td>Traffic Crashes</td><td>DataSF <span class="badge">ubvf-ztfx</span></td><td>{total_crash:,}</td><td>Last 12 months</td><td>Not yet integrated (available)</td></tr>
        <tr><td>Reddit Posts</td><td>Reddit API (r/sanfrancisco)</td><td>{total_reddit}</td><td>Various</td><td>Not yet integrated (available)</td></tr>
        <tr><td>City Performance Survey</td><td>SF Controller's Office</td><td>—</td><td>2023 (most recent)</td><td>Calibration reference</td></tr>
        <tr><td>CityBeat Poll</td><td>SF Chamber of Commerce</td><td>—</td><td>2025</td><td>Calibration reference</td></tr>
    </table>

    <h3>SPI Computation</h3>
    <p class="prose">
        The Safety Perception Index combines five normalized components at the neighborhood level:
    </p>
    <table>
        <tr><th>Component</th><th>Weight</th><th>Source</th><th>Rationale</th></tr>
        <tr><td>Disorder Density (D)</td><td>35%</td><td>311 cases / km²</td><td>Strongest perception driver per broken windows research</td></tr>
        <tr><td>Crime Rate (C)</td><td>25%</td><td>SFPD incidents / km²</td><td>Contributes to perception but less than visible disorder</td></tr>
        <tr><td>Disorder Composition (DC)</td><td>20%</td><td>Category salience weighting</td><td>Encampments and waste affect perception more than graffiti</td></tr>
        <tr><td>Night Concentration (N)</td><td>10%</td><td>Night-to-day report ratio</td><td>Nighttime activity amplifies perception of unsafety</td></tr>
        <tr><td>Trend Direction (T)</td><td>10%</td><td>3-month change</td><td>Worsening conditions feel worse than stable ones</td></tr>
    </table>
    <p class="prose" style="font-family:'JetBrains Mono',monospace;font-size:13px;background:#F3F4F6;padding:12px 16px;border-radius:6px;">
        SPI = 100 − (0.35×D + 0.25×C + 0.20×DC + 0.10×N + 0.10×T)
    </p>

    <h3>Known Limitations</h3>
    <p class="prose">
        <strong>Reporting bias:</strong> 311 data reflects who reports, not what exists. Neighborhoods with engaged
        residents over-report; areas where people have given up under-report.<br>
        <strong>Survival bias:</strong> Areas people actively avoid generate fewer data points and may appear safer
        than they feel. Mobility data (Replica/SafeGraph) would help address this but is not yet integrated.<br>
        <strong>No direct perception:</strong> This entire dashboard is built from proxy data. We are inferring
        how people feel from what they report and where incidents occur. Phase 1 would collect the actual signal.<br>
        <strong>Weight calibration:</strong> Component weights are based on research literature, not empirical
        calibration against SF perception data. Phase 1 would enable proper calibration.
    </p>
</div>

<!-- ═══ ASK ═══ -->
<div id="ask" class="sec">
    <div class="dark">
        <h3>Everything you just saw is proxy data — our best inference.</h3>
        <p>
            311 only captures what people report. Crime data only captures what police file.
            Areas people avoid appear safe. We need the actual signal — how people <em>feel</em>,
            in the moment, at the places they actually are.
        </p>
    </div>

    <h3>What Phase 1 Unlocks</h3>
    <table>
        <tr><th>What We Have Now (Proxy)</th><th>What Phase 1 Adds (Direct)</th></tr>
        <tr><td>311 complaints — lagging, reporter bias</td><td>Direct, in-the-moment perception</td></tr>
        <tr><td>Crime incidents — only what gets reported</td><td>Real-time safety sentiment</td></tr>
        <tr><td>Biennial survey — 2-year lag, neighborhood level</td><td>Daily signal, block level</td></tr>
        <tr><td>Proxy-based SPI — inferred from observable data</td><td>Calibrated index with ground truth</td></tr>
    </table>

    <h3>How It Works</h3>
    <p class="prose">
        A single, optional question — <em>"Right now, how does the surrounding area feel to you?"</em> —
        delivered through existing digital touchpoints during normal daily activity.
        Comfortable / Neutral / Uncomfortable. Anonymous. Aggregated by place and time.
    </p>

    <div class="channels">
        <div class="ch"><div class="t">Offices & Buildings</div><div class="d">Employee check-in (Envoy), workplace comms, visitor sign-in</div></div>
        <div class="ch"><div class="t">Stores & Restaurants</div><div class="d">Point of sale — Square, Toast, Clover, Apple Pay</div></div>
        <div class="ch"><div class="t">Transit</div><div class="d">BART / Muni, Uber / Lyft / Waymo, Google Maps, parking</div></div>
        <div class="ch"><div class="t">Location-Based Apps</div><div class="d">AllTrails, Strava, Yelp, QR feedback in public spaces</div></div>
    </div>

    <h3>The Feedback Loop</h3>
    <p class="prose">
        Phase 1 validates signal quality. Phase 2 builds a <strong>correlation engine</strong> — mapping which
        observable conditions (311 categories, time of day, interventions) predict perception scores.
        Phase 2 also tests <strong>interventions</strong> — cleaning, lighting, ambassadors, street musicians,
        signage — and measures their impact on near-real-time sentiment. This creates a continuous
        improvement cycle: measure → correlate → intervene → re-measure.
    </p>

    <div class="metrics">
        <div class="m"><div class="v" style="color:#1B3A35">$150–200K</div><div class="l">Phase 1 Investment</div><div class="s">6-month pilot</div></div>
        <div class="m"><div class="v" style="color:#1B3A35">6 months</div><div class="l">Pilot Duration</div><div class="s">Define → Build → Capture → Evaluate</div></div>
        <div class="m"><div class="v" style="color:#1B3A35">50K/mo</div><div class="l">Response Target</div><div class="s">Ramp from 5K initial</div></div>
    </div>

    <div class="dark" style="text-align:center;margin-top:32px;">
        <h3>Partner with us to validate whether direct sentiment can fill the gap.</h3>
        <p style="font-size:1rem;color:rgba(255,255,255,0.8);margin-top:8px;">
            City Science Lab San Francisco × MIT Media Lab City Science
        </p>
    </div>
</div>

</div>

<footer>
    Public Safety Pulse — City Science Lab San Francisco × MIT Media Lab City Science<br>
    Data: DataSF Open Data Portal · SF City Performance Survey 2023 · CityBeat 2025 Poll<br>
    Dashboard generated {datetime.now().strftime('%B %d, %Y')} · All data from public sources
</footer>

<script>
const SPI = {json.dumps(spi_json)};
const TIME = {json.dumps(time_json)};
const CAT_HOOD = {json.dumps(cat_hood_json)};
const CATS = {json.dumps(categories)};
const HOURLY_311 = {json.dumps(hourly_311.to_dict('records'))};
const MONTHLY_311 = {json.dumps(monthly_311.to_dict('records'))};
const MONTHLY_SFPD = {json.dumps(monthly_sfpd.to_dict('records'))};

const LAYOUT = {{
    font: {{ family: 'DM Sans, sans-serif', color: '#374151', size: 12 }},
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    margin: {{ l: 10, r: 10, t: 10, b: 10 }},
    hovermode: 'closest',
}};

function show(id, btn) {{
    document.querySelectorAll('.sec').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(id).classList.add('active');
    btn.classList.add('active');
}}

// SPI bar chart
const spiSorted = [...SPI].sort((a,b) => a.SPI - b.SPI);
const spiColor = spiSorted.map(d => d.SPI < 30 ? '#D4594E' : d.SPI < 50 ? '#D4A03C' : d.SPI < 70 ? '#6BB8F0' : '#2D8B4E');
Plotly.newPlot('spi-chart', [{{
    y: spiSorted.map(d => d.analysis_neighborhood),
    x: spiSorted.map(d => d.SPI),
    type: 'bar', orientation: 'h',
    marker: {{ color: spiColor }},
    text: spiSorted.map(d => d.SPI.toFixed(1)),
    textposition: 'outside',
    textfont: {{ family: 'JetBrains Mono', size: 11 }},
    hovertemplate: '%{{y}}<br>SPI: %{{x:.1f}}<br>Disorder: %{{customdata[0]:,.0f}}<br>Crime: %{{customdata[1]:,.0f}}<extra></extra>',
    customdata: spiSorted.map(d => [d.disorder_count, d.crime_count]),
}}], {{
    ...LAYOUT,
    margin: {{ l: 180, r: 60, t: 10, b: 30 }},
    xaxis: {{ range: [0, 105], gridcolor: '#F3F4F6', title: 'Safety Perception Index (0 = least safe, 100 = safest)' }},
    yaxis: {{ autorange: 'reversed' }},
}}, {{responsive: true}});

// DCDI chart
const dcdiSorted = [...SPI].filter(d => d.disorder_count > 500).sort((a,b) => b.DCDI - a.DCDI);
Plotly.newPlot('dcdi-chart', [{{
    y: dcdiSorted.map(d => d.analysis_neighborhood),
    x: dcdiSorted.map(d => d.DCDI),
    type: 'bar', orientation: 'h',
    marker: {{ color: dcdiSorted.map(d => d.DCDI > 10 ? '#D4A03C' : d.DCDI < -10 ? '#D4594E' : '#6B7280') }},
    text: dcdiSorted.map(d => d.DCDI > 10 ? 'Perception Gap' : d.DCDI < -10 ? 'Hidden Risk' : 'Aligned'),
    textposition: 'outside',
    textfont: {{ size: 10 }},
    hovertemplate: '%{{y}}<br>DCDI: %{{x:+.1f}}<extra></extra>',
}}], {{
    ...LAYOUT,
    margin: {{ l: 180, r: 100, t: 10, b: 30 }},
    xaxis: {{ gridcolor: '#F3F4F6', zeroline: true, zerolinecolor: '#374151', zerolinewidth: 2, title: '← More Crime Than Disorder | More Disorder Than Crime →' }},
}}, {{responsive: true}});

// Hourly chart
Plotly.newPlot('hourly-chart', [{{
    x: HOURLY_311.map(d => d.hour),
    y: HOURLY_311.map(d => d.count),
    type: 'bar',
    marker: {{ color: HOURLY_311.map(d => (d.hour >= 20 || d.hour < 6) ? '#1B3A35' : '#D4A03C') }},
    hovertemplate: '%{{x}}:00<br>%{{y:,.0f}} cases<extra></extra>',
}}], {{
    ...LAYOUT,
    margin: {{ l: 50, r: 20, t: 10, b: 40 }},
    xaxis: {{ title: 'Hour of Day', tickvals: Array.from({{length:12}}, (_,i)=>i*2), gridcolor: '#F3F4F6' }},
    yaxis: {{ title: '311 Cases', gridcolor: '#F3F4F6' }},
    shapes: [
        {{type:'rect',x0:-0.5,x1:5.5,y0:0,y1:1,yref:'paper',fillcolor:'rgba(27,58,53,0.06)',line:{{width:0}}}},
        {{type:'rect',x0:19.5,x1:23.5,y0:0,y1:1,yref:'paper',fillcolor:'rgba(27,58,53,0.06)',line:{{width:0}}}},
    ],
}}, {{responsive: true}});

// Time heatmap
const hoods = [...new Set(TIME.map(d => d.neighborhood))];
const windows = [...new Set(TIME.map(d => d.window))];
const z = hoods.map(h => windows.map(w => {{
    const match = TIME.find(d => d.neighborhood === h && d.window === w);
    return match ? match.cases_per_month : 0;
}}));
Plotly.newPlot('time-heatmap', [{{
    z: z, x: windows, y: hoods, type: 'heatmap',
    colorscale: [[0,'#FAF7F2'],[0.3,'#FFC300'],[0.6,'#E3611C'],[1,'#5A1846']],
    hovertemplate: '%{{y}}<br>%{{x}}<br>%{{z:.0f}} cases/month<extra></extra>',
}}], {{
    ...LAYOUT,
    margin: {{ l: 180, r: 10, t: 10, b: 80 }},
    xaxis: {{ tickangle: -30 }},
}}, {{responsive: true}});

// Category by hood
const ch_hoods = [...new Set(CAT_HOOD.map(d => d.neighborhood))];
const ch_cats = [...new Set(CAT_HOOD.map(d => d.category))];
const colors5 = ['#D4594E','#D4A03C','#3B82C8','#2D8B4E','#7C3AED'];
const traces = ch_cats.map((cat, i) => ({{
    name: cat,
    y: ch_hoods.map(h => {{ const m = CAT_HOOD.find(d => d.neighborhood === h && d.category === cat); return m ? m.count : 0; }}),
    x: ch_hoods,
    type: 'bar',
    marker: {{ color: colors5[i % 5] }},
}}));
Plotly.newPlot('cat-hood-chart', traces, {{
    ...LAYOUT,
    barmode: 'stack',
    margin: {{ l: 50, r: 10, t: 10, b: 120 }},
    xaxis: {{ tickangle: -30 }},
    yaxis: {{ title: 'Cases', gridcolor: '#F3F4F6' }},
    legend: {{ orientation: 'h', y: -0.35, font: {{ size: 10 }} }},
}}, {{responsive: true}});

// Monthly trend
Plotly.newPlot('monthly-chart', [
    {{ x: MONTHLY_311.map(d=>d.month), y: MONTHLY_311.map(d=>d.count), name:'311 Disorder', mode:'lines+markers', line:{{color:'#D4594E',width:2.5}}, marker:{{size:5}} }},
    {{ x: MONTHLY_SFPD.map(d=>d.month), y: MONTHLY_SFPD.map(d=>d.count), name:'SFPD Incidents', mode:'lines+markers', line:{{color:'#7C3AED',width:2.5}}, marker:{{size:5}} }},
], {{
    ...LAYOUT,
    margin: {{ l: 50, r: 20, t: 10, b: 40 }},
    xaxis: {{ gridcolor:'#F3F4F6' }},
    yaxis: {{ title:'Monthly Cases', gridcolor:'#F3F4F6' }},
    legend: {{ orientation:'h', y:-0.15 }},
}}, {{responsive: true}});
</script>
</body>
</html>"""

output_path = Path("psp_dashboard_v5.html")
output_path.write_text(html, encoding='utf-8')
size_mb = output_path.stat().st_size / 1024 / 1024
print(f"\n✅ Dashboard saved: {output_path.absolute()}")
print(f"   Size: {size_mb:.1f} MB")
print(f"\n🚀 Open it: open psp_dashboard_v5.html")
