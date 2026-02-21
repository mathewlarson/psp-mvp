#!/usr/bin/env python3
"""
Public Safety Pulse — Dashboard Generator v4
Reads parquet data files and generates a standalone HTML dashboard.

Usage:
    cd ~/Desktop/psp-mvp
    source venv/bin/activate
    python generate_dashboard.py

Output: dashboard_v4.html (open in browser)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

print("=" * 60)
print("Public Safety Pulse — Dashboard Generator v4")
print("=" * 60)

# ── Load Data ──────────────────────────────────────────────
base = Path("data/raw")

print("\n📂 Loading data...")

# 311 Cases
df_311 = pd.read_parquet(base / "layer2" / "311_cases_safety.parquet")
df_311['lat'] = pd.to_numeric(df_311['lat'], errors='coerce')
df_311['long'] = pd.to_numeric(df_311['long'], errors='coerce')
df_311['requested_datetime'] = pd.to_datetime(df_311['requested_datetime'], errors='coerce')
df_311['hour'] = df_311['requested_datetime'].dt.hour
df_311 = df_311.dropna(subset=['lat', 'long', 'requested_datetime'])
print(f"  ✅ 311 Cases: {len(df_311):,} records")

# SFPD Incidents
df_sfpd = pd.read_parquet(base / "layer1" / "sfpd_incidents.parquet")
df_sfpd['latitude'] = pd.to_numeric(df_sfpd['latitude'], errors='coerce')
df_sfpd['longitude'] = pd.to_numeric(df_sfpd['longitude'], errors='coerce')
df_sfpd['incident_datetime'] = pd.to_datetime(df_sfpd['incident_datetime'], errors='coerce')
df_sfpd['hour'] = df_sfpd['incident_datetime'].dt.hour
df_sfpd = df_sfpd.dropna(subset=['latitude', 'longitude'])
print(f"  ✅ SFPD Incidents: {len(df_sfpd):,} records")

# ── Pre-aggregate for performance ─────────────────────────
print("\n⚙️  Pre-aggregating data...")

# Time windows
time_windows = {
    'all': (0, 24),
    'early': (5, 9),
    'midday': (9, 13),
    'afternoon': (13, 17),
    'evening': (17, 21),
    'night': (21, 1),
    'latenight': (1, 5),
}

def filter_hours(df, hour_col, start, end):
    if start == 0 and end == 24:
        return df
    if start < end:
        return df[df[hour_col].between(start, end - 1)]
    else:
        return df[(df[hour_col] >= start) | (df[hour_col] < end)]

# Sample points for each time window (max 80K per window for performance)
MAX_POINTS = 80000

disorder_by_window = {}
crime_by_window = {}

for name, (start, end) in time_windows.items():
    d = filter_hours(df_311, 'hour', start, end)
    if len(d) > MAX_POINTS:
        d = d.sample(MAX_POINTS, random_state=42)
    disorder_by_window[name] = d[['lat', 'long']].round(5).values.tolist()
    
    c = filter_hours(df_sfpd, 'hour', start, end)
    if len(c) > MAX_POINTS:
        c = c.sample(MAX_POINTS, random_state=42)
    crime_by_window[name] = c[['latitude', 'longitude']].round(5).values.tolist()
    print(f"  {name}: {len(disorder_by_window[name]):,} disorder / {len(crime_by_window[name]):,} crime points")

# Neighborhood stats
hood_col = 'analysis_neighborhood'
hood_311 = df_311.groupby(hood_col).size().reset_index(name='disorder')
hood_sfpd = df_sfpd.groupby(hood_col).size().reset_index(name='crime')
hood_stats = hood_311.merge(hood_sfpd, left_on=hood_col, right_on=hood_col, how='outer').fillna(0)
hood_stats.columns = ['name', 'disorder', 'crime']
hood_stats['disorder'] = hood_stats['disorder'].astype(int)
hood_stats['crime'] = hood_stats['crime'].astype(int)
hood_stats = hood_stats.sort_values('disorder', ascending=False).head(20)
hood_stats_json = hood_stats.to_dict('records')

# Category breakdown
cat_counts = df_311['service_name'].value_counts().head(12)
categories_json = [{"name": k, "count": int(v)} for k, v in cat_counts.items()]

# Hourly distribution
hourly_311 = df_311.groupby('hour').size().reset_index(name='count')
hourly_sfpd = df_sfpd.groupby('hour').size().reset_index(name='count')
hourly_json = {
    'disorder': hourly_311.to_dict('records'),
    'crime': hourly_sfpd.to_dict('records'),
}

# Monthly trends
df_311['month'] = df_311['requested_datetime'].dt.to_period('M').astype(str)
df_sfpd['month'] = df_sfpd['incident_datetime'].dt.to_period('M').astype(str)
monthly_311 = df_311.groupby('month').size().reset_index(name='count').sort_values('month')
monthly_sfpd = df_sfpd.groupby('month').size().reset_index(name='count').sort_values('month')

monthly_json = {
    'disorder': monthly_311.to_dict('records'),
    'crime': monthly_sfpd.to_dict('records'),
}

print(f"\n📊 Total: {len(df_311):,} disorder records, {len(df_sfpd):,} crime records")

# ── Generate HTML ─────────────────────────────────────────
print("\n🎨 Generating dashboard HTML...")

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
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'DM Sans', -apple-system, sans-serif; background: #FAF7F2; color: #1A1A1A; }}
h1, h2, h3, h4 {{ font-family: 'DM Serif Display', Georgia, serif; font-weight: 400; }}

/* Header */
.header {{ background: #1B3A35; padding: 40px 0 32px; }}
.header-inner {{ max-width: 1060px; margin: 0 auto; padding: 0 32px; }}
.header .org {{ color: rgba(255,255,255,0.4); font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 8px; }}
.header h1 {{ color: #fff; font-size: 2.6rem; line-height: 1.1; }}
.header .subtitle {{ color: rgba(255,255,255,0.65); font-size: 1rem; margin-top: 8px; }}

/* Navigation */
.nav {{ background: #fff; border-bottom: 1px solid #E5E7EB; position: sticky; top: 0; z-index: 1000; }}
.nav-inner {{ max-width: 1060px; margin: 0 auto; padding: 0 32px; display: flex; gap: 0; overflow-x: auto; }}
.nav-btn {{ padding: 14px 24px; border: none; background: none; cursor: pointer; font-family: 'DM Sans', sans-serif; font-size: 13px; font-weight: 500; color: #6B7280; border-bottom: 2px solid transparent; transition: all 0.2s; white-space: nowrap; }}
.nav-btn:hover {{ color: #374151; }}
.nav-btn.active {{ color: #1B3A35; font-weight: 700; border-bottom-color: #1B3A35; }}

/* Main content */
.main {{ max-width: 1060px; margin: 0 auto; padding: 32px; }}
.section {{ display: none; }}
.section.active {{ display: block; }}

/* Section label */
.section-label {{ font-family: 'DM Sans', sans-serif; font-size: 11px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #D4594E; margin-bottom: 6px; }}

/* Quote */
.quote {{ border-left: 3px solid #D4594E; padding-left: 24px; margin: 32px 0; }}
.quote .text {{ font-family: 'DM Serif Display', Georgia, serif; font-size: 1.3rem; font-style: italic; color: #374151; line-height: 1.5; }}
.quote .attr {{ font-size: 13px; color: #9CA3AF; margin-top: 10px; font-style: normal; }}

/* Metric cards */
.metrics {{ display: flex; gap: 14px; flex-wrap: wrap; margin: 28px 0; }}
.metric {{ flex: 1; min-width: 155px; background: #fff; border-radius: 10px; padding: 22px 18px; text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
.metric .value {{ font-family: 'DM Serif Display', Georgia, serif; font-size: 2.2rem; line-height: 1; }}
.metric .label {{ font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #6B7280; margin-top: 8px; }}
.metric .sub {{ font-size: 11px; color: #9CA3AF; margin-top: 3px; }}

/* Insight box */
.insight {{ background: #fff; border-left: 3px solid #D4594E; padding: 18px 24px; border-radius: 0 10px 10px 0; margin: 24px 0; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }}
.insight .tag {{ font-size: 11px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #D4594E; margin-bottom: 6px; }}

/* Dark card */
.dark-card {{ background: #1B3A35; border-radius: 12px; padding: 32px; color: #fff; margin: 28px 0; }}
.dark-card h3 {{ color: #fff; font-size: 1.35rem; margin-bottom: 10px; line-height: 1.3; }}
.dark-card p {{ color: rgba(255,255,255,0.75); line-height: 1.6; }}
.dark-card .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }}
.dark-card .grid .val {{ font-family: 'DM Serif Display', Georgia, serif; font-size: 2rem; line-height: 1; }}
.dark-card .grid .lbl {{ font-size: 12px; color: rgba(255,255,255,0.55); margin-top: 4px; line-height: 1.3; }}

/* Map container */
.map-container {{ position: relative; width: 100%; height: 500px; border-radius: 12px; overflow: hidden; margin: 20px 0; box-shadow: 0 2px 12px rgba(0,0,0,0.1); }}
.map-container canvas {{ border-radius: 12px; }}

/* Map controls */
.map-controls {{ display: flex; gap: 12px; flex-wrap: wrap; align-items: center; margin: 16px 0; }}
.map-controls select, .map-controls button {{ font-family: 'DM Sans', sans-serif; font-size: 13px; padding: 8px 16px; border-radius: 6px; border: 1px solid #D1D5DB; background: #fff; cursor: pointer; color: #374151; }}
.map-controls button.active {{ background: #1B3A35; color: #fff; border-color: #1B3A35; }}
.map-controls button:hover {{ background: #F3F4F6; }}
.map-controls button.active:hover {{ background: #264a44; }}

/* Play button */
.play-btn {{ width: 38px; height: 38px; border-radius: 50%; border: none; background: #1B3A35; color: #fff; cursor: pointer; font-size: 14px; display: flex; align-items: center; justify-content: center; transition: background 0.3s; }}
.play-btn.playing {{ background: #D4594E; }}

/* Time indicator */
.time-badge {{ position: absolute; top: 16px; right: 16px; background: rgba(0,0,0,0.7); backdrop-filter: blur(8px); border-radius: 8px; padding: 10px 16px; color: #fff; z-index: 10; }}
.time-badge .label {{ font-size: 9px; text-transform: uppercase; letter-spacing: 0.1em; color: rgba(255,255,255,0.5); }}
.time-badge .value {{ font-family: 'DM Serif Display', Georgia, serif; font-size: 18px; margin-top: 2px; }}

/* Legend */
.legend {{ position: absolute; bottom: 16px; left: 16px; background: rgba(0,0,0,0.65); backdrop-filter: blur(8px); border-radius: 8px; padding: 10px 14px; display: flex; gap: 12px; z-index: 10; }}
.legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 11px; color: rgba(255,255,255,0.8); }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}

/* Method note */
.method {{ background: #F3F4F6; border-radius: 8px; padding: 16px 20px; font-size: 13px; color: #6B7280; line-height: 1.6; margin: 16px 0; }}
.method strong {{ color: #374151; }}
.source-badge {{ display: inline-block; background: #E5E7EB; border-radius: 3px; padding: 1px 6px; font-size: 11px; font-family: 'JetBrains Mono', monospace; color: #6B7280; margin: 0 3px; }}

/* Charts */
.chart-container {{ background: #fff; border-radius: 10px; padding: 24px; margin: 20px 0; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}

/* Side by side maps */
.map-pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 20px 0; }}
.map-pair .map-container {{ height: 400px; }}
.map-label {{ font-size: 13px; font-weight: 600; color: #374151; margin-bottom: 8px; }}

/* Compare table */
.compare-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
.compare-table th {{ background: #F9FAFB; padding: 12px 16px; font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #6B7280; text-align: left; border-bottom: 2px solid #E5E7EB; }}
.compare-table td {{ padding: 12px 16px; font-size: 14px; border-bottom: 1px solid #F3F4F6; }}
.compare-table td:last-child {{ color: #1B3A35; font-weight: 600; }}

/* Channels grid */
.channels {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 20px 0; }}
.channel {{ background: #fff; border-radius: 8px; padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
.channel .ch-title {{ font-weight: 700; font-size: 13px; color: #1B3A35; margin-bottom: 6px; }}
.channel .ch-desc {{ font-size: 12px; color: #6B7280; line-height: 1.4; }}

/* Prose */
.prose {{ font-size: 15px; line-height: 1.7; color: #374151; margin: 16px 0; }}
.prose em {{ color: #D4594E; font-style: italic; }}

/* Responsive */
@media (max-width: 768px) {{
    .metrics {{ flex-direction: column; }}
    .map-pair {{ grid-template-columns: 1fr; }}
    .channels {{ grid-template-columns: 1fr 1fr; }}
    .dark-card .grid {{ grid-template-columns: 1fr 1fr; }}
    .main {{ padding: 20px 16px; }}
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: #D1D5DB; border-radius: 3px; }}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
    <div class="header-inner">
        <div class="org">City Science Lab San Francisco × MIT Media Lab City Science</div>
        <h1>Public Safety Pulse</h1>
        <div class="subtitle">Measuring everyday safety perception in San Francisco</div>
    </div>
</div>

<!-- NAVIGATION -->
<div class="nav">
    <div class="nav-inner">
        <button class="nav-btn active" onclick="showSection('gap')">The Perception Gap</button>
        <button class="nav-btn" onclick="showSection('pulse')">The Pulse Map</button>
        <button class="nav-btn" onclick="showSection('time')">Day vs Night</button>
        <button class="nav-btn" onclick="showSection('hoods')">Neighborhoods</button>
        <button class="nav-btn" onclick="showSection('ask')">The Ask</button>
    </div>
</div>

<!-- MAIN CONTENT -->
<div class="main">

<!-- ═══════════════════════════════════════════ -->
<!-- TAB 1: THE PERCEPTION GAP                  -->
<!-- ═══════════════════════════════════════════ -->
<div id="gap" class="section active">
    <div class="quote">
        <div class="text">"Safety isn't just a statistic; it's a feeling you hold when you're walking down the street."</div>
        <div class="attr">— Daniel Lurie, Mayor of San Francisco, Inauguration Speech, January 2025</div>
    </div>

    <div class="metrics">
        <div class="metric">
            <div class="value" style="color:#2D8B4E">↓45%</div>
            <div class="label">Crime</div>
            <div class="sub">2019 → 2025</div>
        </div>
        <div class="metric">
            <div class="value" style="color:#D4A03C">63%</div>
            <div class="label">Feel Safe (Day)</div>
            <div class="sub">2023 City Survey</div>
        </div>
        <div class="metric">
            <div class="value" style="color:#D4594E">36%</div>
            <div class="label">Feel Safe (Night)</div>
            <div class="sub">2023 City Survey</div>
        </div>
        <div class="metric">
            <div class="value" style="color:#1B3A35">{len(df_311):,}</div>
            <div class="label">311 Disorder Cases</div>
            <div class="sub">Safety-filtered, 12 months</div>
        </div>
        <div class="metric">
            <div class="value" style="color:#7C3AED">{len(df_sfpd):,}</div>
            <div class="label">SFPD Incidents</div>
            <div class="sub">12 months</div>
        </div>
    </div>

    <div class="chart-container">
        <div class="section-label">Resident Sentiment</div>
        <h3>The Perception Gap in Twenty-Seven Years of Data</h3>
        <div id="gap-chart" style="height:380px;"></div>
        <div class="method">
            <strong>Source:</strong> SF Controller's Office, City Performance Survey, 1996–2023.
            General satisfaction (red) fell to an all-time low of 2.98 in 2023. Crime has continued falling since.
        </div>
    </div>

    <div class="dark-card">
        <div class="section-label" style="color:#D4A03C;">CityBeat 2025 Poll — SF Chamber of Commerce</div>
        <h3>Perception is shifting — but we can only see it in expensive one-off polls</h3>
        <div class="grid">
            <div><div class="val" style="color:#2D8B4E">78%</div><div class="lbl">feel safe downtown during the day</div></div>
            <div><div class="val" style="color:#D4A03C">43%</div><div class="lbl">say SF is on the right track (up from 22% in 2024)</div></div>
            <div><div class="val" style="color:#6BB8F0">↓34%</div><div class="lbl">fewer voters say crime has gotten worse</div></div>
            <div><div class="val" style="color:#6BB8F0">↓25%</div><div class="lbl">fewer say street behavior is worse</div></div>
        </div>
        <p style="font-size:11px;color:rgba(255,255,255,0.35);border-top:1px solid rgba(255,255,255,0.1);padding-top:14px;margin-top:20px;">
            Source: SF Chamber of Commerce CityBeat 2025 Poll, sponsored by United Airlines
        </p>
    </div>

    <h3>Why the Gap Persists</h3>
    <p class="prose">
        Total reported crime fell 44.9% from 2019 through 2025 — from 119,177 incidents to 65,707. Every single one of
        the city's ten police districts recorded a decline. Yet the City Survey hit its lowest safety score since 1996.
    </p>
    <p class="prose">
        The answer lies in composition. Larceny theft — car break-ins, shoplifting, package theft — drives roughly 60%
        of the statistical decline but is most susceptible to reporting changes. Meanwhile, the visible markers that shape
        how a street <em>feels</em> — encampments, needles, graffiti, aggressive behavior — are captured in 311 data,
        not crime data. The gap between what the statistics say and what people experience is the problem
        Public Safety Pulse solves.
    </p>
</div>

<!-- ═══════════════════════════════════════════ -->
<!-- TAB 2: THE PULSE MAP                       -->
<!-- ═══════════════════════════════════════════ -->
<div id="pulse" class="section">
    <div class="section-label">Safety Perception Index</div>
    <h2>How Does Each Block Feel?</h2>
    <p class="prose">
        250,000 geocoded 311 disorder reports rendered at block level. Taller, hotter columns = more reports about
        encampments, street cleaning, graffiti, and needles — a proxy for how unsafe an area <em>feels</em>.
        Hit play to watch hotspots shift across 24 hours.
    </p>

    <div class="map-controls">
        <button class="play-btn" id="play-btn" onclick="togglePlay()">▶</button>
        <select id="time-select" onchange="updateMap()">
            <option value="all">All Day</option>
            <option value="early">Early Morning (5–9am)</option>
            <option value="midday">Midday (9am–1pm)</option>
            <option value="afternoon">Afternoon (1–5pm)</option>
            <option value="evening">Evening (5–9pm)</option>
            <option value="night">Night (9pm–1am)</option>
            <option value="latenight">Late Night (1–5am)</option>
        </select>
        <select id="focus-select" onchange="updateMapView()">
            <option value="city">All San Francisco</option>
            <option value="tenderloin" selected>Tenderloin / Civic Center</option>
            <option value="union">Union Square / Downtown</option>
            <option value="soma">SoMa</option>
            <option value="mission">Mission District</option>
            <option value="fidi">Financial District</option>
        </select>
        <button id="layer-disorder" class="active" onclick="setLayer('disorder')">311 Disorder</button>
        <button id="layer-crime" onclick="setLayer('crime')">SFPD Crime</button>
        <button id="layer-both" onclick="setLayer('both')">Both</button>
    </div>

    <div class="map-container" id="pulse-map">
        <div class="time-badge">
            <div class="label">Viewing</div>
            <div class="value" id="time-label">All Day</div>
        </div>
        <div class="legend">
            <div class="legend-item"><div class="legend-dot" style="background:#FFC300"></div>Low</div>
            <div class="legend-item"><div class="legend-dot" style="background:#E3611C"></div>Medium</div>
            <div class="legend-item"><div class="legend-dot" style="background:#900C3F"></div>High</div>
            <div class="legend-item"><div class="legend-dot" style="background:#5A1846"></div>Critical</div>
        </div>
    </div>

    <div class="chart-container">
        <div class="section-label">Disorder Composition</div>
        <h4>What's Driving the Signal?</h4>
        <div id="category-chart" style="height:360px;"></div>
    </div>

    <div class="method">
        <strong>Methodology:</strong> Each hexagon aggregates geocoded reports within ~80m radius.
        Height and color = report density. Data: DataSF Socrata API, 12-month window.
        <span class="source-badge">vw6y-z8j6</span> 311 Cases
        <span class="source-badge">wg3w-h783</span> SFPD Incidents
    </div>
</div>

<!-- ═══════════════════════════════════════════ -->
<!-- TAB 3: DAY VS NIGHT                        -->
<!-- ═══════════════════════════════════════════ -->
<div id="time" class="section">
    <div class="section-label">Temporal Patterns</div>
    <h2>When Does the City Feel Unsafe?</h2>
    <p class="prose">
        The same block feels different at noon versus midnight. This is the variation the City Survey
        can't capture — it asks one question, once every two years. Public Safety Pulse captures it in
        <em>4-hour windows, every day, at block level</em>.
    </p>

    <div class="chart-container">
        <h4>Hourly Distribution of 311 Disorder Reports</h4>
        <div id="hourly-chart" style="height:300px;"></div>
    </div>

    <div class="map-pair">
        <div>
            <div class="map-label">☀️ Daytime (7am – 7pm)</div>
            <div class="map-container" id="day-map"></div>
        </div>
        <div>
            <div class="map-label">🌙 Nighttime (9pm – 5am)</div>
            <div class="map-container" id="night-map"></div>
        </div>
    </div>

    <div class="insight">
        <div class="tag">Why This Matters for Phase 1</div>
        Public Safety Pulse would capture this time-of-day variation at block level — telling you not just
        <em>where</em> people feel unsafe, but <em>when</em>. That's the difference between deploying an
        ambassador team to a neighborhood and deploying them to a specific intersection at 6pm on Thursdays.
    </div>
</div>

<!-- ═══════════════════════════════════════════ -->
<!-- TAB 4: NEIGHBORHOODS                       -->
<!-- ═══════════════════════════════════════════ -->
<div id="hoods" class="section">
    <div class="section-label">Where Perception ≠ Reality</div>
    <h2>Disorder vs Crime by Neighborhood</h2>
    <p class="prose">
        The <strong>Disorder–Crime Divergence</strong> shows where what people <em>see</em> (311 environmental reports)
        differs from the crime data. Neighborhoods with high divergence have a <em>perception problem</em> — they
        feel unsafe because of environmental conditions, not criminal danger. These are where cleaning, lighting,
        and ambassador programs have the highest ROI.
    </p>

    <div class="chart-container">
        <div id="divergence-chart" style="height:500px;"></div>
    </div>

    <div class="chart-container">
        <h4>Monthly Trends</h4>
        <div id="monthly-chart" style="height:300px;"></div>
    </div>
</div>

<!-- ═══════════════════════════════════════════ -->
<!-- TAB 5: THE ASK                             -->
<!-- ═══════════════════════════════════════════ -->
<div id="ask" class="section">
    <div class="dark-card">
        <h3>Everything you just saw is proxy data — our best guess.</h3>
        <p>
            311 only captures what people report. Crime data only captures what police file.
            Areas people avoid appear safe in the data. We need the actual signal.
        </p>
    </div>

    <h3>What Phase 1 Unlocks</h3>
    <table class="compare-table">
        <tr><th>What We Have Now (Proxy)</th><th>What Phase 1 Adds (Direct)</th></tr>
        <tr><td>311 complaints — lagging, reporter bias</td><td>Direct, in-the-moment perception</td></tr>
        <tr><td>Crime incidents — only what gets reported</td><td>Real-time safety sentiment</td></tr>
        <tr><td>Biennial survey — 2-year lag, neighborhood level</td><td>Daily signal, block level</td></tr>
        <tr><td>Review text mining — business-adjacent only</td><td>Universal coverage via existing touchpoints</td></tr>
        <tr><td>Foot traffic proxy — infers avoidance</td><td>Directly asks: "How does this area feel?"</td></tr>
    </table>

    <h3>The Solution: Low-Friction Sentiment Capture</h3>
    <p class="prose">
        A single, optional question — <em>"Right now, how does the surrounding area feel to you?"</em> —
        delivered through existing digital touchpoints during normal daily activity.
        Comfortable / Neutral / Uncomfortable. Anonymous. Aggregated by place and time.
    </p>

    <div class="channels">
        <div class="channel"><div class="ch-title">Offices & Buildings</div><div class="ch-desc">Employee check-in via Envoy, workplace comms, visitor sign-in</div></div>
        <div class="channel"><div class="ch-title">Stores & Venues</div><div class="ch-desc">Point of sale — Square, Toast, Clover. Also SNAP, Apple Pay, Google Wallet</div></div>
        <div class="channel"><div class="ch-title">Transit</div><div class="ch-desc">BART / Muni, Uber / Lyft / Waymo, Google Maps, parking apps</div></div>
        <div class="channel"><div class="ch-title">Mobile & Location</div><div class="ch-desc">AllTrails, Strava, Yelp, QR feedback kiosks in public spaces</div></div>
    </div>

    <div class="metrics">
        <div class="metric">
            <div class="value" style="color:#1B3A35">$150–200K</div>
            <div class="label">Phase 1 Investment</div>
            <div class="sub">6-month pilot</div>
        </div>
        <div class="metric">
            <div class="value" style="color:#1B3A35">6 months</div>
            <div class="label">Pilot Duration</div>
            <div class="sub">Define → Build → Capture → Evaluate</div>
        </div>
        <div class="metric">
            <div class="value" style="color:#1B3A35">50K/mo</div>
            <div class="label">Response Target</div>
            <div class="sub">Ramp from 5K initial</div>
        </div>
    </div>

    <h3>The Feedback Loop</h3>
    <p class="prose">
        Public Safety Pulse creates a <strong>powerful feedback loop</strong>: collect high-frequency ground truth
        on how safe people feel → feed it into a correlation engine that identifies which levers move perception →
        deploy targeted interventions (cleaning, lighting, ambassadors, music, signage) → measure the impact in
        near-real-time → optimize. This is how you turn data into measurably safer streets.
    </p>

    <div class="dark-card" style="text-align:center;">
        <h3>Partner with us to validate whether direct sentiment can fill the gap.</h3>
        <p style="font-size:1.1rem;color:rgba(255,255,255,0.85);margin-top:12px;">
            City Science Lab San Francisco × MIT Media Lab City Science
        </p>
    </div>
</div>

</div><!-- end .main -->

<!-- FOOTER -->
<div style="max-width:1060px;margin:0 auto;padding:20px 32px 40px;text-align:center;color:#9CA3AF;font-size:12px;border-top:1px solid #E5E7EB;">
    Public Safety Pulse — City Science Lab San Francisco × MIT Media Lab City Science<br>
    Data: DataSF Open Data Portal · SF City Performance Survey 2023 · CityBeat 2025 Poll · SFPD<br>
    <span class="source-badge">vw6y-z8j6</span>
    <span class="source-badge">wg3w-h783</span>
    <span class="source-badge">ubvf-ztfx</span>
</div>

<script>
// ── Data ──────────────────────────────────────────
const DISORDER_DATA = {json.dumps({k: v for k, v in disorder_by_window.items()})};
const CRIME_DATA = {json.dumps({k: v for k, v in crime_by_window.items()})};
const HOOD_STATS = {json.dumps(hood_stats_json)};
const CATEGORIES = {json.dumps(categories_json)};
const HOURLY = {json.dumps(hourly_json)};
const MONTHLY = {json.dumps(monthly_json)};

const TIME_LABELS = {{
    all: 'All Day', early: '5–9am', midday: '9am–1pm',
    afternoon: '1–5pm', evening: '5–9pm', night: '9pm–1am', latenight: '1–5am'
}};

const FOCUS_VIEWS = {{
    city: {{ lat: 37.760, lng: -122.440, zoom: 11.5, pitch: 30 }},
    tenderloin: {{ lat: 37.782, lng: -122.413, zoom: 15, pitch: 50 }},
    union: {{ lat: 37.788, lng: -122.407, zoom: 15, pitch: 50 }},
    soma: {{ lat: 37.778, lng: -122.400, zoom: 14.5, pitch: 45 }},
    mission: {{ lat: 37.760, lng: -122.418, zoom: 14.5, pitch: 45 }},
    fidi: {{ lat: 37.794, lng: -122.398, zoom: 15, pitch: 50 }},
}};

const WARM_RANGE = [[255,255,178],[254,204,92],[253,141,60],[240,59,32],[189,0,38],[128,0,38]];
const PURPLE_RANGE = [[218,208,235],[188,170,210],[158,130,190],[128,90,165],[98,50,140],[68,10,110]];

const CARTO_DARK = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';
const CARTO_LIGHT = 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json';

// ── Navigation ────────────────────────────────────
function showSection(id) {{
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(id).classList.add('active');
    event.target.classList.add('active');

    // Init maps when tab becomes visible
    if (id === 'pulse' && !window.pulseMapInit) initPulseMap();
    if (id === 'time' && !window.timeMapInit) initTimeMaps();
}}

// ── Pulse Map ─────────────────────────────────────
let currentLayer = 'disorder';
let currentTime = 'all';
let deckInstance = null;
let isPlaying = false;
let playInterval = null;
const timeKeys = ['early', 'midday', 'afternoon', 'evening', 'night', 'latenight'];

function makeHexLayer(points, colorRange, elevScale, opacity) {{
    return new deck.HexagonLayer({{
        id: 'hex-' + Math.random(),
        data: points,
        getPosition: d => [d[1], d[0]],
        radius: 80,
        elevationScale: elevScale,
        elevationRange: [0, 500],
        extruded: true,
        pickable: true,
        opacity: opacity,
        colorRange: colorRange,
        coverage: 0.85,
    }});
}}

function getLayers() {{
    const t = currentTime;
    let layers = [];
    if (currentLayer === 'disorder' || currentLayer === 'both') {{
        layers.push(makeHexLayer(DISORDER_DATA[t], WARM_RANGE, 3, currentLayer === 'both' ? 0.6 : 0.8));
    }}
    if (currentLayer === 'crime' || currentLayer === 'both') {{
        layers.push(makeHexLayer(CRIME_DATA[t], PURPLE_RANGE, 8, currentLayer === 'both' ? 0.5 : 0.8));
    }}
    return layers;
}}

function initPulseMap() {{
    const focus = FOCUS_VIEWS[document.getElementById('focus-select').value];
    const isNight = ['night', 'latenight', 'evening'].includes(currentTime);

    deckInstance = new deck.DeckGL({{
        container: 'pulse-map',
        initialViewState: {{
            longitude: focus.lng, latitude: focus.lat, zoom: focus.zoom,
            pitch: focus.pitch, bearing: -15,
        }},
        controller: true,
        layers: getLayers(),
        mapStyle: isNight ? CARTO_DARK : CARTO_LIGHT,
        getTooltip: ({{object}}) => object && `${{object.colorValue}} reports in this area`,
    }});
    window.pulseMapInit = true;
}}

function updateMap() {{
    currentTime = document.getElementById('time-select').value;
    document.getElementById('time-label').textContent = TIME_LABELS[currentTime];
    const isNight = ['night', 'latenight', 'evening'].includes(currentTime);

    if (deckInstance) {{
        deckInstance.setProps({{
            layers: getLayers(),
            mapStyle: isNight ? CARTO_DARK : CARTO_LIGHT,
        }});
    }}
}}

function updateMapView() {{
    const focus = FOCUS_VIEWS[document.getElementById('focus-select').value];
    if (deckInstance) {{
        deckInstance.setProps({{
            initialViewState: {{
                longitude: focus.lng, latitude: focus.lat, zoom: focus.zoom,
                pitch: focus.pitch, bearing: -15,
                transitionDuration: 1000,
            }},
        }});
    }}
}}

function setLayer(layer) {{
    currentLayer = layer;
    ['disorder', 'crime', 'both'].forEach(l => {{
        document.getElementById('layer-' + l).classList.toggle('active', l === layer);
    }});
    if (deckInstance) deckInstance.setProps({{ layers: getLayers() }});
}}

function togglePlay() {{
    const btn = document.getElementById('play-btn');
    if (isPlaying) {{
        clearInterval(playInterval);
        isPlaying = false;
        btn.textContent = '▶';
        btn.classList.remove('playing');
    }} else {{
        let idx = 0;
        isPlaying = true;
        btn.textContent = '⏸';
        btn.classList.add('playing');
        playInterval = setInterval(() => {{
            document.getElementById('time-select').value = timeKeys[idx];
            updateMap();
            idx = (idx + 1) % timeKeys.length;
        }}, 2000);
    }}
}}

// ── Time Maps ─────────────────────────────────────
function initTimeMaps() {{
    const dayData = DISORDER_DATA['midday'];
    const nightData = DISORDER_DATA['night'];
    const view = {{ longitude: -122.412, latitude: 37.782, zoom: 13.5, pitch: 45, bearing: -15 }};

    new deck.DeckGL({{
        container: 'day-map',
        initialViewState: view,
        controller: true,
        layers: [makeHexLayer(dayData, WARM_RANGE, 4, 0.8)],
        mapStyle: CARTO_LIGHT,
    }});
    new deck.DeckGL({{
        container: 'night-map',
        initialViewState: view,
        controller: true,
        layers: [makeHexLayer(nightData, WARM_RANGE, 8, 0.8)],
        mapStyle: CARTO_DARK,
    }});
    window.timeMapInit = true;
}}

// ── Charts ────────────────────────────────────────
function initCharts() {{
    // Perception Gap chart
    const years = [1996,1998,2000,2002,2004,2006,2008,2010,2012,2014,2016,2018,2019,2021,2023];
    const dayS = [3.8,4.0,4.0,3.2,3.3,3.5,4.2,4.2,4.2,4.3,4.35,4.2,4.2,3.5,3.3];
    const nightS = [3.25,3.1,3.0,2.9,3.2,3.2,3.3,3.3,3.3,3.5,3.55,3.3,3.2,3.0,2.9];
    const genS = [3.3,3.2,3.25,3.1,3.2,3.3,3.3,3.25,3.25,3.3,3.5,3.4,3.3,3.1,2.98];

    const layout = {{
        font: {{ family: 'DM Sans', color: '#374151' }},
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
        margin: {{ l:45, r:20, t:10, b:40 }},
        yaxis: {{ range:[2.5,4.6], gridcolor:'#F3F4F6' }},
        xaxis: {{ gridcolor:'#F3F4F6' }},
        legend: {{ orientation:'h', y:-0.12 }},
        hovermode: 'x unified',
        shapes: [2002,2008,2015,2020].map(yr => ({{
            type:'line', x0:yr, x1:yr, y0:2.5, y1:4.6,
            line: {{ color:'#E5E7EB', width:1, dash:'dash' }}
        }})),
        annotations: [
            {{x:2002,y:4.55,text:'Dot-com Bust',showarrow:false,font:{{size:9,color:'#9CA3AF'}}}},
            {{x:2008,y:4.55,text:'Great Recession',showarrow:false,font:{{size:9,color:'#9CA3AF'}}}},
            {{x:2015,y:4.55,text:'Tech Peak',showarrow:false,font:{{size:9,color:'#9CA3AF'}}}},
            {{x:2020,y:4.55,text:'COVID',showarrow:false,font:{{size:9,color:'#9CA3AF'}}}},
        ],
    }};

    Plotly.newPlot('gap-chart', [
        {{ x:years, y:dayS, name:'Safe (Day)', mode:'lines+markers', line:{{color:'#3B82C8',width:2.5}}, marker:{{size:5}} }},
        {{ x:years, y:nightS, name:'Safe (Night)', mode:'lines+markers', line:{{color:'#D4594E',width:2.5}}, marker:{{size:5}} }},
        {{ x:years, y:genS, name:'General Satisfaction', mode:'lines+markers', line:{{color:'#9CA3AF',width:2,dash:'dot'}}, marker:{{size:4}} }},
    ], layout, {{responsive:true}});

    // Category chart
    const cats = CATEGORIES.reverse();
    Plotly.newPlot('category-chart', [{{
        y: cats.map(c => c.name), x: cats.map(c => c.count),
        type: 'bar', orientation: 'h',
        marker: {{ color: cats.map(c => c.count), colorscale: [['0','#FFC300'],['0.5','#C70039'],['1','#5A1846']] }},
        hovertemplate: '%{{y}}<br>%{{x:,.0f}} cases<extra></extra>',
    }}], {{
        font: {{ family: 'DM Sans', color: '#374151' }},
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
        margin: {{ l:180, r:20, t:10, b:40 }},
        xaxis: {{ gridcolor:'#F3F4F6', title: 'Cases' }},
        yaxis: {{ title: null }},
        showlegend: false,
    }}, {{responsive:true}});

    // Hourly chart
    const h = HOURLY.disorder;
    Plotly.newPlot('hourly-chart', [{{
        x: h.map(d => d.hour), y: h.map(d => d.count),
        type: 'bar',
        marker: {{ color: h.map(d => (d.hour >= 20 || d.hour < 6) ? '#1B3A35' : '#D4A03C') }},
        hovertemplate: '%{{x}}:00<br>%{{y:,.0f}} cases<extra></extra>',
    }}], {{
        font: {{ family: 'DM Sans', color: '#374151' }},
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
        margin: {{ l:50, r:20, t:10, b:40 }},
        xaxis: {{ title: 'Hour of Day', tickvals: [0,2,4,6,8,10,12,14,16,18,20,22], gridcolor:'#F3F4F6' }},
        yaxis: {{ title: '311 Cases', gridcolor:'#F3F4F6' }},
        shapes: [
            {{type:'rect',x0:-0.5,x1:5.5,y0:0,y1:1,yref:'paper',fillcolor:'rgba(27,58,53,0.06)',line:{{width:0}}}},
            {{type:'rect',x0:19.5,x1:23.5,y0:0,y1:1,yref:'paper',fillcolor:'rgba(27,58,53,0.06)',line:{{width:0}}}},
        ],
    }}, {{responsive:true}});

    // Divergence chart
    const hoods = HOOD_STATS;
    const maxD = Math.max(...hoods.map(h => h.disorder));
    const maxC = Math.max(...hoods.map(h => h.crime));
    Plotly.newPlot('divergence-chart', [
        {{
            y: hoods.map(h => h.name), x: hoods.map(h => (h.disorder/maxD*100).toFixed(1)),
            name: '311 Disorder Index', type: 'bar', orientation: 'h',
            marker: {{ color: '#D4594E', opacity: 0.85 }},
        }},
        {{
            y: hoods.map(h => h.name), x: hoods.map(h => (h.crime/maxC*100).toFixed(1)),
            name: 'SFPD Crime Index', type: 'bar', orientation: 'h',
            marker: {{ color: '#7C3AED', opacity: 0.75 }},
        }},
    ], {{
        font: {{ family: 'DM Sans', color: '#374151' }},
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
        margin: {{ l:160, r:20, t:10, b:40 }},
        barmode: 'group',
        xaxis: {{ title: 'Relative Index (normalized)', gridcolor:'#F3F4F6' }},
        legend: {{ orientation:'h', y:-0.08 }},
    }}, {{responsive:true}});

    // Monthly trend
    const m311 = MONTHLY.disorder;
    const mSfpd = MONTHLY.crime;
    Plotly.newPlot('monthly-chart', [
        {{ x: m311.map(d=>d.month), y: m311.map(d=>d.count), name:'311 Disorder', mode:'lines+markers', line:{{color:'#D4594E',width:2.5}}, marker:{{size:5}} }},
        {{ x: mSfpd.map(d=>d.month), y: mSfpd.map(d=>d.count), name:'SFPD Incidents', mode:'lines+markers', line:{{color:'#7C3AED',width:2.5}}, marker:{{size:5}} }},
    ], {{
        font: {{ family: 'DM Sans', color: '#374151' }},
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
        margin: {{ l:50, r:20, t:10, b:40 }},
        xaxis: {{ gridcolor:'#F3F4F6' }},
        yaxis: {{ title:'Monthly Cases', gridcolor:'#F3F4F6' }},
        legend: {{ orientation:'h', y:-0.15 }},
    }}, {{responsive:true}});
}}

// ── Init ──────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {{
    initCharts();
}});
</script>
</body>
</html>"""

output_path = Path("dashboard_v4.html")
output_path.write_text(html, encoding='utf-8')
print(f"\n✅ Dashboard saved to: {output_path.absolute()}")
print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
print(f"\n🚀 Open it:")
print(f"   open dashboard_v4.html")
"""

NOTE: This file will be large because it embeds the data.
For production, you'd serve the data from separate JSON files.
"""
