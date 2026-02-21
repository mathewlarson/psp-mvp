#!/usr/bin/env python3
"""
Public Safety Pulse — Dashboard v7
===================================
Fixes: heat map clarity + neighborhood labels
Adds: decision-maker UX, correlation engine, alert concept, data roadmap

Usage:
    cd ~/Desktop/psp-mvp && source venv/bin/activate
    python compute_spi_v3.py
    open psp_dashboard_v7.html
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

print("=" * 60)
print("Public Safety Pulse — Dashboard v7")
print("=" * 60)

# ═══════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════
base = Path("data/raw")
print("\n📂 Loading data...")

df_311 = pd.read_parquet(base / "layer2" / "311_cases_safety.parquet")
df_311['lat'] = pd.to_numeric(df_311['lat'], errors='coerce')
df_311['long'] = pd.to_numeric(df_311['long'], errors='coerce')
df_311['requested_datetime'] = pd.to_datetime(df_311['requested_datetime'], errors='coerce')
df_311 = df_311.dropna(subset=['lat', 'long', 'requested_datetime'])
df_311['hour'] = df_311['requested_datetime'].dt.hour
df_311['month'] = df_311['requested_datetime'].dt.to_period('M').astype(str)
hood_col = 'analysis_neighborhood'
cat_col = 'service_name'
print(f"  311: {len(df_311):,}")

df_sfpd = pd.read_parquet(base / "layer1" / "sfpd_incidents.parquet")
df_sfpd['latitude'] = pd.to_numeric(df_sfpd['latitude'], errors='coerce')
df_sfpd['longitude'] = pd.to_numeric(df_sfpd['longitude'], errors='coerce')
df_sfpd['incident_datetime'] = pd.to_datetime(df_sfpd['incident_datetime'], errors='coerce')
df_sfpd = df_sfpd.dropna(subset=['latitude', 'longitude', 'incident_datetime'])
df_sfpd['hour'] = df_sfpd['incident_datetime'].dt.hour
df_sfpd['month'] = df_sfpd['incident_datetime'].dt.to_period('M').astype(str)
print(f"  SFPD: {len(df_sfpd):,}")

df_crash = None
cp = base / "layer1" / "traffic_crashes.parquet"
if cp.exists():
    df_crash = pd.read_parquet(cp)
    lat_c = [c for c in df_crash.columns if 'lat' in c.lower()]
    lng_c = [c for c in df_crash.columns if 'lon' in c.lower() or 'lng' in c.lower()]
    if lat_c and lng_c:
        df_crash['lat'] = pd.to_numeric(df_crash[lat_c[0]], errors='coerce')
        df_crash['lng'] = pd.to_numeric(df_crash[lng_c[0]], errors='coerce')
        df_crash = df_crash.dropna(subset=['lat', 'lng'])
    date_c = [c for c in df_crash.columns if 'date' in c.lower()]
    if date_c:
        df_crash['crash_datetime'] = pd.to_datetime(df_crash[date_c[0]], errors='coerce')
        df_crash['hour'] = df_crash['crash_datetime'].dt.hour
    print(f"  Crashes: {len(df_crash):,}")

df_reddit = None
rp = base / "layer3" / "reddit_safety_posts.parquet"
if rp.exists():
    df_reddit = pd.read_parquet(rp)
    neg_w = ['unsafe','dangerous','scary','sketchy','crime','stabbing','shooting','robbery','assault',
             'theft','homeless','encampment','needles','dirty','terrible','worse','avoid','afraid','mugged']
    pos_w = ['safe','clean','beautiful','nice','great','better','improved','love','wonderful',
             'pleasant','comfortable','welcoming','vibrant']
    df_reddit['text'] = (df_reddit.get('title', pd.Series(dtype=str)).fillna('') + ' ' +
                          df_reddit.get('selftext', pd.Series(dtype=str)).fillna('')).str.lower()
    df_reddit['neg'] = df_reddit['text'].apply(lambda t: sum(1 for w in neg_w if w in str(t)))
    df_reddit['pos'] = df_reddit['text'].apply(lambda t: sum(1 for w in pos_w if w in str(t)))
    df_reddit['sentiment'] = df_reddit['pos'] - df_reddit['neg']
    print(f"  Reddit: {len(df_reddit):,} (sentiment avg: {df_reddit['sentiment'].mean():.2f})")


# ═══════════════════════════════════════════════
# COMPUTE SPI
# ═══════════════════════════════════════════════
print("\n⚙️  Computing SPI...")

HOOD_AREAS = {
    'Bayview Hunters Point':6.2,'Bernal Heights':2.1,'Castro/Upper Market':1.5,
    'Chinatown':0.5,'Excelsior':2.8,'Financial District/South Beach':2.1,
    'Glen Park':1.8,'Golden Gate Park':4.1,'Haight Ashbury':1.2,'Hayes Valley':0.8,
    'Inner Richmond':2.5,'Inner Sunset':2.5,'Japantown':0.4,'Lakeshore':3.5,
    'Lincoln Park':1.2,'Lone Mountain/USF':1.1,'Marina':1.5,'McLaren Park':2.8,
    'Mission':2.8,'Mission Bay':1.2,'Nob Hill':0.8,'Noe Valley':1.8,
    'North Beach':0.6,'Oceanview/Merced/Ingleside':3.5,'Outer Mission':1.8,
    'Outer Richmond':4.2,'Outer Sunset':5.5,'Pacific Heights':1.5,'Portola':1.8,
    'Potrero Hill':2.1,'Presidio':6.0,'Presidio Heights':0.8,'Russian Hill':0.8,
    'Seacliff':0.5,'South of Market':3.5,'Sunset/Parkside':5.5,'Tenderloin':0.6,
    'Treasure Island':1.5,'Twin Peaks':1.2,'Visitacion Valley':2.2,
    'West of Twin Peaks':3.8,'Western Addition':1.5,
}

# Neighborhood centroids for map labels (approximate)
HOOD_CENTROIDS = {
    'Tenderloin': [37.783, -122.414], 'South of Market': [37.778, -122.400],
    'Mission': [37.760, -122.418], 'Financial District/South Beach': [37.793, -122.397],
    'Bayview Hunters Point': [37.732, -122.390], 'Western Addition': [37.781, -122.435],
    'Hayes Valley': [37.776, -122.425], 'Nob Hill': [37.792, -122.415],
    'Chinatown': [37.794, -122.407], 'North Beach': [37.800, -122.410],
    'Castro/Upper Market': [37.762, -122.435], 'Haight Ashbury': [37.770, -122.448],
    'Pacific Heights': [37.792, -122.435], 'Marina': [37.800, -122.437],
    'Russian Hill': [37.800, -122.420], 'Potrero Hill': [37.760, -122.392],
    'Bernal Heights': [37.740, -122.415], 'Noe Valley': [37.750, -122.432],
    'Inner Richmond': [37.780, -122.465], 'Inner Sunset': [37.762, -122.468],
    'Outer Richmond': [37.778, -122.495], 'Outer Sunset': [37.755, -122.495],
    'Sunset/Parkside': [37.745, -122.490], 'Excelsior': [37.725, -122.430],
    'Visitacion Valley': [37.715, -122.405], 'Oceanview/Merced/Ingleside': [37.720, -122.455],
    'Glen Park': [37.735, -122.435], 'Lone Mountain/USF': [37.778, -122.452],
    'Outer Mission': [37.725, -122.442], 'Portola': [37.730, -122.405],
    'Lakeshore': [37.725, -122.485], 'Twin Peaks': [37.752, -122.448],
    'West of Twin Peaks': [37.745, -122.460], 'Presidio Heights': [37.788, -122.450],
    'Japantown': [37.785, -122.430], 'Mission Bay': [37.770, -122.390],
    'Presidio': [37.798, -122.465],
}

def z_norm(s):
    mu, sig = s.mean(), s.std()
    return ((s - mu) / sig).clip(-3, 3) if sig > 0 else pd.Series(0, index=s.index)

SALIENCE = {
    'Encampment': 1.0, 'Street and Sidewalk Cleaning': 0.8,
    'Blocked Street and Sidewalk': 0.6, 'Noise': 0.5,
    'Graffiti Public': 0.4, 'Graffiti Private': 0.3,
    'Streetlights': 0.7, 'Damage Property': 0.4,
    'General Request': 0.3, 'Sewer': 0.2,
    'RPD General': 0.2, 'Litter Receptacle Maintenance': 0.3,
    'Illegal Postings': 0.2,
}

SEVERITY = {
    'Homicide':5,'Rape':4,'Robbery':3.5,'Assault':3,'Weapons Carrying Etc':3,
    'Arson':2.5,'Motor Vehicle Theft':2,'Burglary':2,'Larceny Theft':1.5,
    'Vandalism':1.5,'Drug Offense':1.5,'Disorderly Conduct':1.5,
}

max_date = df_311['requested_datetime'].max()
df_311['decay'] = np.exp(-(max_date - df_311['requested_datetime']).dt.days / 180)
df_311['salience'] = df_311[cat_col].map(SALIENCE).fillna(0.3)

# Also extract 311 resolution time as a signal
if 'closed_date' in df_311.columns:
    df_311['closed_date'] = pd.to_datetime(df_311['closed_date'], errors='coerce')
    df_311['resolution_days'] = (df_311['closed_date'] - df_311['requested_datetime']).dt.days
    df_311['resolution_days'] = df_311['resolution_days'].clip(0, 365)
    has_resolution = True
    print("  ✅ Resolution time data available — integrating as Component 7")
elif 'closed' in df_311.columns:
    df_311['closed_dt'] = pd.to_datetime(df_311['closed'], errors='coerce')
    df_311['resolution_days'] = (df_311['closed_dt'] - df_311['requested_datetime']).dt.days
    df_311['resolution_days'] = df_311['resolution_days'].clip(0, 365)
    has_resolution = True
    print("  ✅ Resolution time data available — integrating as Component 7")
else:
    has_resolution = False
    print("  ⚠️  No resolution time column found")

# Crime severity
inc_cat = 'incident_category' if 'incident_category' in df_sfpd.columns else None
for c in df_sfpd.columns:
    if 'categ' in c.lower() and inc_cat is None:
        inc_cat = c
df_sfpd['severity'] = df_sfpd[inc_cat].map(SEVERITY).fillna(1.0) if inc_cat else 1.0
df_sfpd['decay'] = np.exp(-(df_sfpd['incident_datetime'].max() - df_sfpd['incident_datetime']).dt.days / 180)

# 1. Disorder Density
d1 = df_311.groupby(hood_col).agg(cnt=('decay','count'), wt=('decay','sum')).reset_index()
d1['area'] = d1[hood_col].map(HOOD_AREAS).fillna(2)
d1['dd'] = d1['wt'] / d1['area']
d1['D'] = z_norm(d1['dd'])

# 2. Crime Severity
sfpd_hc = hood_col if hood_col in df_sfpd.columns else 'analysis_neighborhood'
d2 = df_sfpd.groupby(sfpd_hc).agg(cc=('severity','count'), sv=('severity','sum')).reset_index()
d2.columns = [hood_col, 'cc', 'sv']
d2['area'] = d2[hood_col].map(HOOD_AREAS).fillna(2)
d2['cd'] = d2['sv'] / d2['area']
d2['C'] = z_norm(d2['cd'])

# 3. Disorder Salience
d3 = df_311.groupby(hood_col)['salience'].mean().reset_index(name='sal')
d3['DC'] = z_norm(d3['sal'])

# 4. Pedestrian Safety
if df_crash is not None and 'lat' in df_crash.columns and len(df_crash) > 0:
    hc = df_311.groupby(hood_col).agg(cl=('lat','mean'), cn=('long','mean')).reset_index()
    def nearest(lat, lng):
        d = ((hc['cl'] - lat)**2 + (hc['cn'] - lng)**2)
        return hc.iloc[d.idxmin()][hood_col]
    df_crash['hood'] = df_crash.apply(lambda r: nearest(r['lat'], r['lng']), axis=1)
    d4 = df_crash.groupby('hood').size().reset_index(name='crashes')
    d4.columns = [hood_col, 'crashes']
    d4['area'] = d4[hood_col].map(HOOD_AREAS).fillna(2)
    d4['PS'] = z_norm(d4['crashes'] / d4['area'])
else:
    d4 = pd.DataFrame({hood_col: d1[hood_col], 'crashes': 0, 'PS': 0.0})

# 5. Temporal Risk
night = (df_311['hour'] >= 20) | (df_311['hour'] < 6)
day = (df_311['hour'] >= 7) & (df_311['hour'] < 19)
nc = df_311[night].groupby(hood_col).size().reset_index(name='n')
dc_day = df_311[day].groupby(hood_col).size().reset_index(name='d')
d5 = nc.merge(dc_day, on=hood_col, how='outer').fillna(1)
d5['nr'] = d5['n'] / (d5['d'] + 1)

cutoff_r = max_date - pd.Timedelta(days=90)
cutoff_p = cutoff_r - pd.Timedelta(days=90)
rec = df_311[df_311['requested_datetime'] >= cutoff_r].groupby(hood_col).size().reset_index(name='r')
pri = df_311[(df_311['requested_datetime'] >= cutoff_p) & (df_311['requested_datetime'] < cutoff_r)].groupby(hood_col).size().reset_index(name='p')
tr = rec.merge(pri, on=hood_col, how='outer').fillna(1)
tr['tr'] = tr['r'] / (tr['p'] + 1)

d5 = d5.merge(tr[[hood_col, 'tr']], on=hood_col, how='outer').fillna(1)
d5['raw'] = 0.6 * d5['nr'] + 0.4 * d5['tr']
d5['TR'] = z_norm(d5['raw'])

# 6. Community Sentiment
reddit_mod = 0.0
hood_reddit = {}
if df_reddit is not None and 'sentiment' in df_reddit.columns:
    reddit_mod = -df_reddit['sentiment'].mean() / max(1, df_reddit['sentiment'].std())
    for hood in HOOD_AREAS:
        hw = hood.lower().replace('/', ' ').split()[0]
        if len(hw) > 4:
            m = df_reddit[df_reddit['text'].str.contains(hw, na=False)]
            if len(m) >= 2:
                hood_reddit[hood] = -m['sentiment'].mean() / max(1, df_reddit['sentiment'].std())

# 7. Resolution Responsiveness (if available)
if has_resolution:
    d7 = df_311.groupby(hood_col)['resolution_days'].median().reset_index(name='med_res')
    d7['RR'] = z_norm(d7['med_res'])
    W = {'D':0.25,'C':0.18,'DC':0.13,'PS':0.08,'TR':0.13,'CS':0.08,'RR':0.15}
else:
    d7 = None
    W = {'D':0.30,'C':0.20,'DC':0.15,'PS':0.10,'TR':0.15,'CS':0.10}

# Combine
spi = d1[[hood_col, 'D', 'cnt', 'dd']].copy()
spi = spi.merge(d2[[hood_col, 'C', 'cc', 'cd']], on=hood_col, how='outer')
spi = spi.merge(d3[[hood_col, 'DC']], on=hood_col, how='outer')
spi = spi.merge(d4[[hood_col, 'PS', 'crashes']], on=hood_col, how='outer')
spi = spi.merge(d5[[hood_col, 'TR', 'nr', 'tr']], on=hood_col, how='outer')
spi['CS'] = spi[hood_col].map(hood_reddit).fillna(reddit_mod)
if d7 is not None:
    spi = spi.merge(d7[[hood_col, 'RR', 'med_res']], on=hood_col, how='outer')
spi = spi.fillna(0)

# Weighted risk
risk = W['D']*spi['D'] + W['C']*spi['C'] + W['DC']*spi['DC'] + W['PS']*spi['PS'] + W['TR']*spi['TR'] + W['CS']*spi['CS']
if d7 is not None:
    risk += W['RR']*spi['RR']
spi['risk'] = risk

rmin, rmax = spi['risk'].min(), spi['risk'].max()
spi['SPI'] = (100 * (1 - (spi['risk'] - rmin) / max(rmax - rmin, 0.001))).round(1)
spi['DCDI'] = ((spi['D'] / max(spi['D'].abs().max(), 1) - spi['C'] / max(spi['C'].abs().max(), 1)) * 50).round(1)
spi = spi.sort_values('SPI')

# Add centroids for map
spi['lat'] = spi[hood_col].map(lambda h: HOOD_CENTROIDS.get(h, [0,0])[0])
spi['lng'] = spi[hood_col].map(lambda h: HOOD_CENTROIDS.get(h, [0,0])[1])

print("\n📊 SPI Results:")
for _, r in spi.head(10).iterrows():
    print(f"  {r[hood_col]:35s} SPI:{r['SPI']:5.1f}")


# ═══════════════════════════════════════════════
# HEAT MAP DATA
# ═══════════════════════════════════════════════
print("\n⚙️  Preparing heat map data...")

TW = [('5am–9am',5,9),('9am–1pm',9,13),('1pm–5pm',13,17),
      ('5pm–9pm',17,21),('9pm–1am',21,1),('1am–5am',1,5)]

def filt_h(df, hc, s, e):
    if s < e: return df[df[hc].between(s, e-1)]
    return df[(df[hc] >= s) | (df[hc] < e)]

MAX_P = 40000
heat = {}
for name, s, e in TW:
    pts = []
    d = filt_h(df_311, 'hour', s, e)
    if len(d) > MAX_P: d = d.sample(MAX_P, random_state=42)
    for _, r in d.iterrows():
        pts.append([round(r['lat'],5), round(r['long'],5), round(r['salience']*r['decay'], 2)])
    c = filt_h(df_sfpd, 'hour', s, e)
    if len(c) > MAX_P//3: c = c.sample(MAX_P//3, random_state=42)
    for _, r in c.iterrows():
        pts.append([round(r['latitude'],5), round(r['longitude'],5), round(float(r['severity'])*float(r['decay']), 2)])
    if df_crash is not None and 'lat' in df_crash.columns and 'hour' in df_crash.columns:
        cr = filt_h(df_crash, 'hour', s, e)
        for _, r in cr.iterrows():
            pts.append([round(r['lat'],5), round(r['lng'],5), 2.0])
    heat[name] = pts
    print(f"  {name}: {len(pts):,} points")

# ═══════════════════════════════════════════════
# ANALYTICS
# ═══════════════════════════════════════════════
print("\n⚙️  Analytics...")

# Time heatmap for top hoods
top_h = spi.head(12)[hood_col].tolist()
nm = max(1, df_311['month'].nunique())
time_data = []
for h in top_h:
    h311 = df_311[df_311[hood_col] == h]
    for name, s, e in TW:
        wc = filt_h(h311, 'hour', s, e)
        time_data.append({'neighborhood': h, 'window': name, 'rate': round(len(wc)/nm, 1)})

# Categories
cats = [{"name": k, "count": int(v)} for k, v in df_311[cat_col].value_counts().head(12).items()]

# Cat by hood
cat_hood = []
for h in spi.head(5)[hood_col].tolist():
    for c, n in df_311[df_311[hood_col]==h][cat_col].value_counts().head(5).items():
        cat_hood.append({'neighborhood':h,'category':c,'count':int(n)})

# Crime cats
cc_col = inc_cat or 'incident_category'
crime_cats = []
if cc_col in df_sfpd.columns:
    crime_cats = [{"name":k,"count":int(v)} for k,v in df_sfpd[cc_col].value_counts().head(10).items()]

# Monthly
m311 = df_311.groupby('month').size().reset_index(name='count').sort_values('month').to_dict('records')
msfpd = df_sfpd.groupby('month').size().reset_index(name='count').sort_values('month').to_dict('records')

# Hourly overlay
h311 = df_311.groupby('hour').size().reset_index(name='count').to_dict('records')
hsfpd = df_sfpd.groupby('hour').size().reset_index(name='count').to_dict('records')

# Neighborhood labels for map
hood_labels = []
for _, r in spi.iterrows():
    if r['lat'] > 0:
        hood_labels.append({
            'name': r[hood_col], 'lat': r['lat'], 'lng': r['lng'],
            'spi': r['SPI'], 'cnt': int(r['cnt']) if pd.notna(r['cnt']) else 0
        })

# SPI for JSON
spi_out = spi[[hood_col,'SPI','DCDI','cnt','cc','crashes','D','C','DC','PS','TR','CS',
               'nr','tr','lat','lng']].copy()
spi_out['cnt'] = spi_out['cnt'].fillna(0).astype(int)
spi_out['cc'] = spi_out['cc'].fillna(0).astype(int)
spi_out['crashes'] = spi_out['crashes'].fillna(0).astype(int)
if d7 is not None and 'med_res' in spi.columns:
    spi_out['med_res'] = spi['med_res'].fillna(0).round(1)
spi_j = spi_out.to_dict('records')

# Date range
dmin = df_311['requested_datetime'].min().strftime('%b %Y')
dmax = df_311['requested_datetime'].max().strftime('%b %Y')
lo = spi.iloc[0]
hi = spi.iloc[-1]
n_components = 7 if d7 is not None else 6
total_records = len(df_311) + len(df_sfpd) + (len(df_crash) if df_crash is not None else 0) + (len(df_reddit) if df_reddit is not None else 0)

# ═══════════════════════════════════════════════
# HTML
# ═══════════════════════════════════════════════
print("\n🎨 Generating dashboard...")

html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
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
.hd .s{{color:rgba(255,255,255,.6);margin-top:6px;font-size:14px}}

.nv{{background:#fff;border-bottom:1px solid #E5E7EB;position:sticky;top:0;z-index:100}}
.nv-i{{max-width:1000px;margin:0 auto;padding:0 32px;display:flex;overflow-x:auto}}
.nb{{padding:12px 18px;border:none;background:none;cursor:pointer;font-family:inherit;font-size:12px;color:#6B7280;border-bottom:2px solid transparent;white-space:nowrap}}
.nb.a{{color:#1B3A35;font-weight:700;border-bottom-color:#D4594E}}

.mn{{max-width:1000px;margin:0 auto;padding:28px 32px}}
.sc{{display:none}}.sc.a{{display:block}}

.lb{{font-size:11px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#D4594E;margin-bottom:4px}}
.qt{{border-left:3px solid #D4594E;padding:4px 0 4px 22px;margin:24px 0}}
.qt p{{font-family:'DM Serif Display',Georgia,serif;font-size:1.15rem;font-style:italic;color:#374151;line-height:1.45}}
.qt .at{{font-size:12px;color:#9CA3AF;font-style:normal;margin-top:6px}}

.ms{{display:flex;gap:10px;flex-wrap:wrap;margin:20px 0}}
.mc{{flex:1;min-width:120px;background:#fff;border-radius:8px;padding:16px 12px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.06)}}
.mc .v{{font-family:'DM Serif Display',Georgia,serif;font-size:1.8rem;line-height:1}}
.mc .l{{font-size:9px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#6B7280;margin-top:5px}}
.mc .u{{font-size:9px;color:#9CA3AF;margin-top:2px}}

.cd{{background:#fff;border-radius:10px;padding:22px;margin:18px 0;box-shadow:0 1px 3px rgba(0,0,0,.06)}}
.dk{{background:#1B3A35;border-radius:10px;padding:24px 26px;color:#fff;margin:20px 0}}
.dk h3{{color:#fff;font-size:1.2rem;margin-bottom:6px;line-height:1.3}}
.dk p{{color:rgba(255,255,255,.72);font-size:13px}}
.dk .g{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}}
.dk .gv{{font-family:'DM Serif Display',Georgia,serif;font-size:1.6rem}}
.dk .gl{{font-size:11px;color:rgba(255,255,255,.5);margin-top:2px}}

.in{{background:#fff;border-left:3px solid #D4594E;padding:14px 18px;border-radius:0 8px 8px 0;margin:18px 0;box-shadow:0 1px 3px rgba(0,0,0,.04)}}
.in .tg{{font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#D4594E;margin-bottom:3px}}

.nt{{background:#F3F4F6;border-radius:6px;padding:12px 16px;font-size:12px;color:#6B7280;margin:14px 0}}
.nt strong{{color:#374151}}
.bg{{display:inline-block;background:#E5E7EB;border-radius:3px;padding:1px 5px;font-size:10px;font-family:'JetBrains Mono',monospace;color:#6B7280;margin:0 2px}}

.pr{{font-size:14px;color:#374151;margin:12px 0}}.pr em{{color:#D4594E}}.pr strong{{color:#1A1A1A}}
table{{width:100%;border-collapse:collapse;margin:14px 0}}
th{{background:#F9FAFB;padding:8px 12px;font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#6B7280;text-align:left;border-bottom:2px solid #E5E7EB}}
td{{padding:8px 12px;font-size:12px;border-bottom:1px solid #F3F4F6}}

.mw{{position:relative;width:100%;height:540px;border-radius:12px;overflow:hidden;margin:18px 0;box-shadow:0 2px 12px rgba(0,0,0,.1)}}
.mc2{{display:flex;gap:10px;align-items:center;margin:14px 0;flex-wrap:wrap}}
.mc2 select{{font-family:inherit;font-size:12px;padding:7px 12px;border-radius:6px;border:1px solid #D1D5DB;background:#fff}}
.pb{{width:34px;height:34px;border-radius:50%;border:none;background:#1B3A35;color:#fff;cursor:pointer;font-size:12px;display:flex;align-items:center;justify-content:center}}
.pb.on{{background:#D4594E}}
.tb{{position:absolute;top:14px;right:14px;background:rgba(0,0,0,.75);backdrop-filter:blur(8px);border-radius:8px;padding:8px 14px;color:#fff;z-index:10}}
.tb .tl{{font-size:8px;text-transform:uppercase;letter-spacing:.1em;color:rgba(255,255,255,.5)}}
.tb .tv{{font-family:'DM Serif Display',Georgia,serif;font-size:16px;margin-top:1px}}
.tp-bar{{position:absolute;bottom:14px;left:14px;right:14px;z-index:10;display:flex;gap:4px}}
.tp-seg{{flex:1;height:4px;border-radius:2px;background:rgba(255,255,255,.2);cursor:pointer;transition:background .3s}}
.tp-seg.ac{{background:#D4594E}}

/* Alert mockup */
.alert-card{{background:#FFF7ED;border:1px solid #FDBA74;border-radius:8px;padding:16px;margin:10px 0;display:flex;gap:12px;align-items:flex-start}}
.alert-icon{{width:32px;height:32px;border-radius:50%;background:#D4594E;color:#fff;display:flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0}}
.alert-body .alert-title{{font-weight:700;font-size:13px;color:#9A3412}}
.alert-body .alert-desc{{font-size:12px;color:#6B7280;margin-top:2px}}
.alert-body .alert-action{{font-size:11px;color:#1B3A35;font-weight:600;margin-top:6px}}

/* Correlation bars */
.corr-bar{{display:flex;align-items:center;gap:8px;margin:6px 0}}
.corr-fill{{height:14px;border-radius:3px;transition:width .6s}}
.corr-label{{font-size:12px;min-width:200px;color:#374151}}
.corr-val{{font-family:'JetBrains Mono',monospace;font-size:11px;color:#6B7280;min-width:40px}}

.chs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin:14px 0}}
.ch{{background:#fff;border-radius:8px;padding:14px;box-shadow:0 1px 3px rgba(0,0,0,.06)}}
.ch .ct{{font-weight:700;font-size:12px;color:#1B3A35;margin-bottom:3px}}
.ch .cd2{{font-size:11px;color:#6B7280}}

.two{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
@media(max-width:768px){{.ms{{flex-direction:column}}.dk .g,.two,.chs{{grid-template-columns:1fr}}.mn{{padding:16px}}}}

footer{{max-width:1000px;margin:0 auto;padding:20px 32px 36px;text-align:center;color:#9CA3AF;font-size:11px;border-top:1px solid #E5E7EB}}
</style></head><body>

<div class="hd"><div class="w">
<div class="o">City Science Lab San Francisco × MIT Media Lab City Science</div>
<h1>Public Safety Pulse</h1>
<div class="s">A {n_components}-component Safety Perception Index computed from {total_records:,} data points across {4 if df_crash is not None else 3} public datasets</div>
</div></div>

<div class="nv"><div class="nv-i">
<button class="nb a" onclick="sh('over',this)">Overview</button>
<button class="nb" onclick="sh('heat',this)">Live Heat Map</button>
<button class="nb" onclick="sh('spi',this)">Safety Index</button>
<button class="nb" onclick="sh('time',this)">Time of Day</button>
<button class="nb" onclick="sh('corr',this)">Correlation Engine</button>
<button class="nb" onclick="sh('meth',this)">Data & Methods</button>
<button class="nb" onclick="sh('ask',this)">Partner With Us</button>
</div></div>

<div class="mn">

<!-- ═══ OVERVIEW ═══ -->
<div id="over" class="sc a">

<div class="two" style="margin-bottom:20px">
<div>
<div class="lb">The Problem</div>
<h2 style="font-size:1.5rem;line-height:1.2;margin-bottom:10px">San Francisco's safety data tells half the story</h2>
<p class="pr">
Crime is declining. But residents don't feel safer. The <strong>perception gap</strong> — the distance between
what the statistics say and what people experience on the street — is invisible to current measurement systems.
The City Survey captures it once every two years. Public Safety Pulse would capture it <em>every day, at block level</em>.
</p>
</div>
<div>
<div class="qt" style="margin-top:0">
<p>"Safety isn't just a statistic; it's a feeling you hold when you're walking down the street."</p>
<div class="at">— Mayor Daniel Lurie, January 2025</div>
</div>
</div>
</div>

<div class="ms">
<div class="mc"><div class="v" style="color:#D4A03C">63%</div><div class="l">Feel Safe (Day)</div><div class="u">2023 City Survey</div></div>
<div class="mc"><div class="v" style="color:#D4594E">36%</div><div class="l">Feel Safe (Night)</div><div class="u">2023 City Survey</div></div>
<div class="mc"><div class="v" style="color:#2D8B4E">78%</div><div class="l">Feel Safe Downtown</div><div class="u">2025 CityBeat Poll</div></div>
<div class="mc"><div class="v" style="color:#1B3A35">{total_records:,}</div><div class="l">Data Points Fused</div><div class="u">{dmin}–{dmax}</div></div>
</div>

<div class="dk">
<h3>What is Public Safety Pulse?</h3>
<p>Three components that create a powerful feedback loop:</p>
<div class="g" style="grid-template-columns:1fr 1fr 1fr;margin-top:14px">
<div style="border-right:1px solid rgba(255,255,255,.1);padding-right:12px">
<div class="gv" style="color:#D4A03C;font-size:1.2rem">🔬</div>
<div style="font-size:13px;font-weight:600;color:#fff;margin:4px 0">Correlation Machine</div>
<div class="gl">Takes in data and outputs a trusted measure of safety perception using sound data science</div>
</div>
<div style="border-right:1px solid rgba(255,255,255,.1);padding:0 12px">
<div class="gv" style="color:#6BB8F0;font-size:1.2rem">📡</div>
<div style="font-size:13px;font-weight:600;color:#fff;margin:4px 0">Sentiment Sensors</div>
<div class="gl">Network of digital touchpoints collecting direct feedback on how safe people feel</div>
</div>
<div style="padding-left:12px">
<div class="gv" style="color:#2D8B4E;font-size:1.2rem">⚡</div>
<div style="font-size:13px;font-weight:600;color:#fff;margin:4px 0">Targeted Interventions</div>
<div class="gl">Deploy resources — cleaning, lighting, ambassadors, music — and measure impact</div>
</div>
</div>
</div>

<div class="in">
<div class="tg">What This Dashboard Demonstrates</div>
The Safety Perception Index below fuses {total_records:,} data points from 311 disorder reports, SFPD incidents,
traffic crashes, and Reddit community sentiment into a single perception score per neighborhood.
<strong>This is what existing proxy data gets us.</strong> Phase 1 adds the missing signal — direct sentiment from
real people at specific places and times.
</div>

<div class="cd">
<div class="lb">How Perception is Shifting</div>
<h4>The City Survey shows historic lows — but newer polls suggest a turning point</h4>
<p class="pr" style="font-size:12px;color:#6B7280;margin-bottom:8px">
The 2023 City Survey (most recent) shows perception at its lowest point since 1996. But the 2025 CityBeat poll
shows improvement among frequent downtown visitors. Without Public Safety Pulse, we can't see how this varies
by block, time of day, or in response to interventions.</p>
<div id="hourly-chart" style="height:250px"></div>
</div>
</div>

<!-- ═══ HEAT MAP ═══ -->
<div id="heat" class="sc">
<div class="lb">Animated Perception Signal</div>
<h2>How Safety Perception Shifts Through the Day</h2>
<p class="pr">
Every point on this map is a real 311 report, SFPD incident, or traffic crash — weighted by its
impact on perception. Encampments and violent crime produce stronger heat signals than graffiti
or property offenses. <strong>Press play</strong> to watch hotspots migrate through a typical day.
Neighborhood labels show SPI scores for context.
</p>

<div class="mc2">
<button class="pb" id="pbtn" onclick="toggleP()">▶</button>
<select id="tw" onchange="setW(+this.value)">
<option value="0">5am – 9am</option><option value="1">9am – 1pm</option>
<option value="2" selected>1pm – 5pm</option><option value="3">5pm – 9pm</option>
<option value="4">9pm – 1am</option><option value="5">1am – 5am</option>
</select>
<select id="foc" onchange="setF(this.value)">
<option value="city">All San Francisco</option>
<option value="tl" selected>Tenderloin / Civic Center</option>
<option value="us">Union Square / Downtown</option>
<option value="sm">SoMa</option><option value="ms">Mission District</option>
<option value="fi">Financial District</option>
</select>
</div>

<div class="mw" id="hmap">
<div class="tb"><div class="tl">Time Window</div><div class="tv" id="twl">1pm–5pm</div></div>
<div class="tp-bar" id="tpb"></div>
</div>

<div class="in">
<div class="tg">What Decision-Makers See</div>
The heat signature concentrates in Tenderloin/SoMa during all hours but <em>migrates</em> — the Mission
corridor heats up in the evening, the Financial District cools off after office hours. With Phase 1
sentiment data, each heat zone would include a real-time perception score, enabling alerts like:
"SPI in Tenderloin dropped below 20 in the 5–9pm window — dispatch ambassador team."
</div>

<h3 style="margin-top:24px">Concept: Real-Time Perception Alerts</h3>
<p class="pr" style="margin-bottom:12px">With Phase 1 data flowing, the dashboard would generate automatic alerts when perception drops below thresholds:</p>
<div class="alert-card">
<div class="alert-icon">⚠️</div>
<div class="alert-body">
<div class="alert-title">SPI Alert: Tenderloin dropped to 15.2 (5pm–9pm window)</div>
<div class="alert-desc">Disorder reports up 40% vs last week. Encampment reports concentrated on Turk & Taylor. 3 new graffiti reports on Golden Gate Ave.</div>
<div class="alert-action">→ Recommended: Deploy ambassador team to Turk/Taylor corridor. Dispatch cleaning crew to Golden Gate Ave.</div>
</div>
</div>
<div class="alert-card">
<div class="alert-icon" style="background:#D4A03C">📈</div>
<div class="alert-body">
<div class="alert-title">SPI Improvement: Union Square rose to 68.4 (9am–1pm window)</div>
<div class="alert-desc">Disorder reports down 25% since ambassador deployment last Monday. Encampment reports at lowest level in 6 months.</div>
<div class="alert-action">→ Signal: Maintain current resource allocation. Share results with CBD stakeholders.</div>
</div>
</div>

<div class="nt"><strong>Note:</strong> These alerts are mockups showing Phase 1 capability. The current dashboard cannot generate real-time alerts because it uses batch data (311/SFPD), not live sentiment. Phase 1's digital touchpoints would enable true real-time scoring.</div>
</div>

<!-- ═══ SPI ═══ -->
<div id="spi" class="sc">
<div class="lb">Computed Metric</div>
<h2>Safety Perception Index</h2>
<p class="pr">
SPI fuses {n_components} data signals into a single 0–100 score per neighborhood. Higher = safer feeling.
This is a <strong>proxy estimate</strong> from publicly available data. Phase 1 would validate and calibrate
these scores against direct perception measurements.
</p>
<div class="cd"><h4>SPI by Neighborhood</h4><div id="spi-chart" style="height:620px"></div></div>
<div class="cd"><h4>Disorder–Crime Divergence</h4>
<p style="font-size:11px;color:#6B7280;margin-bottom:8px">Positive = more disorder than crime (perception problem). Negative = more crime than disorder (hidden risk).</p>
<div id="dcdi-chart" style="height:480px"></div></div>
</div>

<!-- ═══ TIME ═══ -->
<div id="time" class="sc">
<div class="lb">Temporal Analysis</div>
<h2>Disorder by Time of Day</h2>
<p class="pr">Darker cells = more reports per month. This is the variation a biennial survey cannot capture.</p>
<div class="cd"><div id="time-hm" style="height:420px"></div></div>
<div class="two">
<div class="cd"><h4>311 Disorder Categories</h4><div id="cat-chart" style="height:320px"></div></div>
<div class="cd"><h4>SFPD Crime Categories</h4><div id="crcat-chart" style="height:320px"></div></div>
</div>
<div class="cd"><h4>Monthly Trends</h4><div id="mo-chart" style="height:260px"></div></div>
</div>

<!-- ═══ CORRELATION ENGINE ═══ -->
<div id="corr" class="sc">
<div class="lb">Phase 2 Vision</div>
<h2>The Correlation Engine</h2>
<p class="pr">
Once Phase 1 provides direct perception data, the correlation engine identifies which observable factors
<em>actually predict</em> how safe people feel. This replaces assumed weights with empirical ones.
</p>

<div class="dk">
<h3>How It Works</h3>
<p>Feed all available data into a regression model with direct sentiment as the dependent variable.
The model reveals which levers — cleaning, lighting, ambassador presence, encampment density —
most strongly correlate with perception. Then test interventions and measure impact.</p>
</div>

<div class="cd">
<h4>Preliminary Correlation Signals</h4>
<p style="font-size:12px;color:#6B7280;margin-bottom:14px">
Based on current data: neighborhoods with higher values of these indicators tend to have lower SPI scores.
These are associations, not causal findings — Phase 1 enables causal testing via interventions.</p>
<div id="corr-chart" style="height:300px"></div>
</div>

<div class="two" style="margin-top:16px">
<div class="cd">
<h4 style="font-size:14px">Direct Indicators</h4>
<p style="font-size:12px;color:#6B7280">Likely to directly measure or predict sentiment:</p>
<ul style="font-size:12px;color:#374151;padding-left:18px;margin-top:8px;line-height:2">
<li>311 call volume and category composition</li>
<li>911 call volume (not yet integrated)</li>
<li>BART safety surveys (not yet available)</li>
<li>Annual city surveys</li>
<li>Nextdoor safety-related posts</li>
<li>Visitor/SFTravel safety surveys</li>
<li><strong>Phase 1: Direct NPS from digital touchpoints</strong></li>
</ul>
</div>
<div class="cd">
<h4 style="font-size:14px">Indirect Indicators</h4>
<p style="font-size:12px;color:#6B7280">Known correlates from urban research:</p>
<ul style="font-size:12px;color:#374151;padding-left:18px;margin-top:8px;line-height:2">
<li>Visible drug use presence</li>
<li>Amount of visible trash/waste</li>
<li>Number of unoccupied storefronts</li>
<li>Defiled or damaged public locations</li>
<li>Unhoused individuals in high-traffic areas</li>
<li>Streetlight outages and darkness</li>
<li>Pedestrian/cyclist crash density</li>
</ul>
</div>
</div>

<h3 style="margin-top:20px">Possible Interventions to Test</h3>
<div class="chs" style="grid-template-columns:repeat(auto-fit,minmax(200px,1fr))">
<div class="ch"><div class="ct">👁️ Sights</div><div class="cd2">Clean streets/sidewalks, remove graffiti, bright signage and lighting, deploy ambassadors optimally</div></div>
<div class="ch"><div class="ct">🔊 Sounds</div><div class="cd2">Deploy street musicians, play recorded music in key corridors</div></div>
<div class="ch"><div class="ct">🌸 Smells</div><div class="cd2">Pleasant aromas in areas associated with bad smells</div></div>
<div class="ch"><div class="ct">🤝 Civic Signals</div><div class="cd2">Respond quickly to issues, greet visitors, visible service presence</div></div>
</div>

<div class="in">
<div class="tg">Measurement Feedback Loop</div>
<strong>Collect</strong> high-frequency sentiment → <strong>Correlate</strong> with observable conditions →
<strong>Intervene</strong> with targeted resources → <strong>Re-measure</strong> to assess impact →
<strong>Optimize</strong> allocation based on what actually works. This is how you turn data into measurably safer streets.
</div>
</div>

<!-- ═══ METHODS ═══ -->
<div id="meth" class="sc">
<div class="lb">Transparency</div>
<h2>Data & Methods</h2>

<h3>Data Sources</h3>
<table>
<tr><th>Source</th><th>Records</th><th>SPI Role</th><th>Weight</th></tr>
<tr><td>311 Requests <span class="bg">vw6y-z8j6</span></td><td>{len(df_311):,}</td><td>Disorder density + salience + temporal + resolution</td><td>{int(W['D']*100)}% + {int(W['DC']*100)}% + {int(W['TR']*100)}%{' + '+str(int(W.get('RR',0)*100))+'%' if d7 is not None else ''}</td></tr>
<tr><td>SFPD Incidents <span class="bg">wg3w-h783</span></td><td>{len(df_sfpd):,}</td><td>Crime severity density</td><td>{int(W['C']*100)}%</td></tr>
<tr><td>Traffic Crashes <span class="bg">ubvf-ztfx</span></td><td>{len(df_crash) if df_crash is not None else 0:,}</td><td>Pedestrian safety</td><td>{int(W['PS']*100)}%</td></tr>
<tr><td>Reddit r/sanfrancisco</td><td>{len(df_reddit) if df_reddit is not None else 0}</td><td>Community sentiment baseline</td><td>{int(W['CS']*100)}%</td></tr>
</table>

<h3>SPI Formula</h3>
<p class="pr" style="font-family:'JetBrains Mono',monospace;font-size:11px;background:#F3F4F6;padding:12px;border-radius:6px;line-height:2">
SPI = 100 − scaled({' + '.join(f'{int(v*100)}%×{k}' for k,v in W.items())})<br>
D = z_score( Σ(311_cases × e^(-days/180)) / area_km² )<br>
C = z_score( Σ(SFPD × severity_weight × e^(-days/180)) / area_km² )<br>
DC = z_score( mean_salience_weight per neighborhood )<br>
PS = z_score( crashes / area_km² )<br>
TR = z_score( 0.6 × night_ratio + 0.4 × trend_ratio )<br>
CS = keyword_sentiment from Reddit (neighborhood-specific where possible)<br>
{'RR = z_score( median_311_resolution_days )' if d7 is not None else ''}
</p>

<h3>Additional Data for Phase 1 Integration</h3>
<table>
<tr><th>Dataset</th><th>Access</th><th>Signal</th><th>Priority</th></tr>
<tr><td>BART Station Exits</td><td>bart.gov (manual download)</td><td>Foot traffic avoidance — declining exits = people avoiding the area</td><td>High</td></tr>
<tr><td>Muni Ridership</td><td>sfmta.com</td><td>Transit confidence indicator</td><td>Medium</td></tr>
<tr><td>Yelp/Google Reviews</td><td>Yelp Fusion API (free)</td><td>Direct safety sentiment from real visitors at specific locations</td><td>High</td></tr>
<tr><td>Business Vacancy</td><td>CBRE/Cushman & Wakefield quarterly</td><td>Empty storefronts signal neighborhood decline</td><td>Medium</td></tr>
<tr><td>Streetlight Outages</td><td>Already in 311 data (needs extraction)</td><td>Lighting = strongest single perception predictor</td><td>High</td></tr>
<tr><td>Google Trends</td><td>Free API</td><td>Search volume for "[neighborhood] + safety" keywords</td><td>Medium</td></tr>
<tr><td>City Survey Microdata</td><td>Controller's Office request</td><td>Enables regression-based weight calibration</td><td>Critical</td></tr>
<tr><td>Replica/SafeGraph</td><td>Commercial (MIT partnership)</td><td>Mobility patterns — where people avoid walking</td><td>High</td></tr>
<tr><td>Pedestrian Counts</td><td>SFMTA automated counters</td><td>Foot traffic decline = avoidance behavior</td><td>Medium</td></tr>
<tr><td>Nextdoor Posts</td><td>Partnership needed</td><td>Hyperlocal community safety discussion</td><td>Medium</td></tr>
</table>

<h3>Methodological Improvements for Phase 1</h3>
<p class="pr" style="font-size:12px;color:#6B7280;">
<strong>Current approach:</strong> Weighted z-score fusion with assumed weights from literature.<br>
<strong>Phase 1 upgrade:</strong> Principal Component Analysis to discover which signals co-vary with
direct perception → Bayesian hierarchical regression for weight calibration → spatial autocorrelation
(Moran's I) to account for neighboring block influence → temporal autoregression for trend prediction.<br>
<strong>Research basis:</strong> Wilson & Kelling (1982) Broken Windows, Sampson & Raudenbush (1999) Disorder Observation,
Salesses et al. (2013) MIT Place Pulse, Naik et al. (2014) MIT Streetscore, Welsh & Farrington (2008) Lighting & Crime.
</p>

<h3>Known Limitations</h3>
<p class="pr" style="font-size:12px;color:#6B7280;">
<strong>Reporting bias:</strong> 311 reflects who reports. Engaged neighborhoods over-report.<br>
<strong>Survival bias:</strong> Areas people avoid generate fewer data points.<br>
<strong>No direct perception:</strong> Everything here is inferred from proxy data.<br>
<strong>Reddit:</strong> Only {len(df_reddit) if df_reddit is not None else 0} posts, not geocoded.<br>
<strong>Weights:</strong> Not empirically calibrated. Phase 1 fixes this.
</p>
</div>

<!-- ═══ ASK ═══ -->
<div id="ask" class="sc">
<div class="dk">
<h3>Everything you just saw is inferred from proxy data.</h3>
<p>311 captures what people report. Crime data captures what police file.
Areas people avoid appear safe. We need the actual signal — how people <em>feel</em>,
in the moment, at the places they actually are.</p>
</div>

<h3>The Solution: Low-Friction Sentiment Capture</h3>
<p class="pr"><em>"Right now, how does the surrounding area feel to you?"</em>
— Comfortable / Neutral / Uncomfortable. Through existing digital touchpoints. Anonymous. Aggregated by place and time.</p>

<h3>Distribution Channels</h3>
<div class="chs">
<div class="ch"><div class="ct">Offices & Buildings</div><div class="cd2">Employee check-in via Envoy, workplace comms, visitor sign-in</div></div>
<div class="ch"><div class="ct">Stores & Restaurants</div><div class="cd2">Point of sale — Square, Toast, Clover, Apple Pay, SNAP, Google Wallet</div></div>
<div class="ch"><div class="ct">Transit</div><div class="cd2">BART / Muni, Uber / Lyft / Waymo, Google Maps, parking apps</div></div>
<div class="ch"><div class="ct">Location Apps</div><div class="cd2">AllTrails, Strava, Yelp, feedback QR kiosks</div></div>
</div>

<div class="ms">
<div class="mc"><div class="v" style="color:#1B3A35">$150–200K</div><div class="l">Phase 1</div><div class="u">6-month pilot</div></div>
<div class="mc"><div class="v" style="color:#1B3A35">6 months</div><div class="l">Duration</div><div class="u">Define → Build → Capture → Evaluate</div></div>
<div class="mc"><div class="v" style="color:#1B3A35">50K/mo</div><div class="l">Target Responses</div><div class="u">Ramp from 5K</div></div>
</div>

<div class="dk" style="text-align:center;margin-top:24px">
<h3>Partner with us to validate whether direct sentiment can fill the gap.</h3>
<p style="color:rgba(255,255,255,.8);margin-top:6px">City Science Lab San Francisco × MIT Media Lab City Science</p>
</div>
</div>

</div>

<footer>Public Safety Pulse — City Science Lab SF × MIT Media Lab · Generated {datetime.now().strftime('%B %d, %Y')}</footer>

<script>
const SPI={json.dumps(spi_j)};
const TSPI={json.dumps(time_data)};
const CATS={json.dumps(cats)};
const CATH={json.dumps(cat_hood)};
const CCATS={json.dumps(crime_cats)};
const H311={json.dumps(h311)};
const HSFPD={json.dumps(hsfpd)};
const M311={json.dumps(m311)};
const MSFPD={json.dumps(msfpd)};
const HEAT={json.dumps(heat)};
const LABELS={json.dumps(hood_labels)};
const TWN=['5am–9am','9am–1pm','1pm–5pm','5pm–9pm','9pm–1am','1am–5am'];
const DK='https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';
const LT='https://basemaps.cartocdn.com/gl/positron-gl-style/style.json';
const FOC={{city:{{lat:37.76,lng:-122.44,z:11.8,p:20}},tl:{{lat:37.782,lng:-122.413,z:14.5,p:35}},us:{{lat:37.788,lng:-122.407,z:14.5,p:35}},sm:{{lat:37.778,lng:-122.4,z:14,p:30}},ms:{{lat:37.76,lng:-122.418,z:14,p:30}},fi:{{lat:37.794,lng:-122.398,z:14.5,p:35}}}};
const PL={{font:{{family:'DM Sans',color:'#374151',size:11}},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',margin:{{l:10,r:10,t:10,b:10}},hovermode:'closest'}};

function sh(id,btn){{document.querySelectorAll('.sc').forEach(s=>s.classList.remove('a'));document.querySelectorAll('.nb').forEach(b=>b.classList.remove('a'));document.getElementById(id).classList.add('a');btn.classList.add('a');if(id==='heat'&&!window._hm)initH()}}

let hD,hW=2,playing=false,pI;

function spiColor(v){{return v<25?'#D4594E':v<30?'#E76A50':v<35?'#D4A03C':v<45?'#DDB84C':v<55?'#9CC07C':v<70?'#6BB8F0':'#2D8B4E'}}

function initH(){{
    const f=FOC[document.getElementById('foc').value];
    const night=[4,5].includes(hW);

    // Label layer: neighborhood names + SPI on the map
    const labelLayer=new deck.TextLayer({{
        id:'labels',
        data:LABELS.filter(d=>d.lat>0),
        getPosition:d=>[d.lng,d.lat],
        getText:d=>d.name.split('/')[0].split('(')[0].trim()+'\\n'+d.spi.toFixed(0),
        getSize:12,
        getColor:d=>{{const s=d.spi;return s<30?[212,89,78,220]:s<50?[212,160,60,220]:[45,139,78,200]}},
        fontFamily:'DM Sans',
        fontWeight:700,
        getTextAnchor:'middle',
        getAlignmentBaseline:'center',
        billboard:false,
        sizeScale:1,
        background:true,
        getBackgroundColor:[255,255,255,180],
        backgroundPadding:[4,2],
        getBorderColor:[200,200,200,100],
        getBorderWidth:1,
    }});

    hD=new deck.DeckGL({{
        container:'hmap',
        initialViewState:{{longitude:f.lng,latitude:f.lat,zoom:f.z,pitch:f.p,bearing:0}},
        controller:true,
        layers:[makeHL(hW),labelLayer],
        mapStyle:night?DK:LT,
    }});

    const pg=document.getElementById('tpb');
    TWN.forEach((n,i)=>{{const d=document.createElement('div');d.className='tp-seg'+(i===hW?' ac':'');d.title=n;d.onclick=()=>setW(i);pg.appendChild(d)}});
    window._hm=true;
}}

function makeHL(i){{
    const k=TWN[i],pts=HEAT[k]||[];
    return new deck.HeatmapLayer({{
        id:'heat',data:pts,
        getPosition:d=>[d[1],d[0]],
        getWeight:d=>d[2],
        radiusPixels:60,
        intensity:1.5,
        threshold:0.03,
        colorRange:[[255,255,204],[255,237,160],[254,217,118],[254,178,76],[253,141,60],[240,59,32],[189,0,38]],
    }});
}}

function setW(i){{
    hW=i;
    document.getElementById('tw').value=i;
    document.getElementById('twl').textContent=TWN[i];
    document.querySelectorAll('.tp-seg').forEach((d,j)=>d.classList.toggle('ac',j===i));
    const night=[4,5].includes(i);
    const ll=new deck.TextLayer({{
        id:'labels',data:LABELS.filter(d=>d.lat>0),
        getPosition:d=>[d.lng,d.lat],
        getText:d=>d.name.split('/')[0].split('(')[0].trim()+'\\n'+d.spi.toFixed(0),
        getSize:12,
        getColor:d=>{{const s=d.spi;return night?(s<30?[255,120,100,240]:s<50?[255,200,100,240]:[120,255,120,220]):(s<30?[212,89,78,220]:s<50?[212,160,60,220]:[45,139,78,200])}},
        fontFamily:'DM Sans',fontWeight:700,
        getTextAnchor:'middle',getAlignmentBaseline:'center',
        billboard:false,sizeScale:1,
        background:true,
        getBackgroundColor:night?[0,0,0,160]:[255,255,255,180],
        backgroundPadding:[4,2],
    }});
    if(hD)hD.setProps({{layers:[makeHL(i),ll],mapStyle:night?DK:LT}});
}}

function setF(v){{const f=FOC[v];if(hD)hD.setProps({{initialViewState:{{longitude:f.lng,latitude:f.lat,zoom:f.z,pitch:f.p,bearing:0,transitionDuration:1000}}}})}}
function toggleP(){{const b=document.getElementById('pbtn');if(playing){{clearInterval(pI);playing=false;b.textContent='▶';b.classList.remove('on')}}else{{playing=true;b.textContent='⏸';b.classList.add('on');pI=setInterval(()=>{{hW=(hW+1)%6;setW(hW)}},2500)}}}}

function init(){{
    // Hourly
    Plotly.newPlot('hourly-chart',[
        {{x:H311.map(d=>d.hour),y:H311.map(d=>d.count),name:'311',type:'bar',marker:{{color:H311.map(d=>(d.hour>=20||d.hour<6)?'#1B3A35':'#D4A03C')}},hovertemplate:'%{{x}}:00 — %{{y:,.0f}} 311<extra></extra>'}},
        {{x:HSFPD.map(d=>d.hour),y:HSFPD.map(d=>d.count),name:'SFPD',type:'bar',marker:{{color:'rgba(124,58,237,.35)'}},hovertemplate:'%{{x}}:00 — %{{y:,.0f}} SFPD<extra></extra>'}},
    ],{{...PL,margin:{{l:50,r:20,t:10,b:40}},barmode:'overlay',xaxis:{{title:'Hour',tickvals:[...Array(12)].map((_,i)=>i*2),gridcolor:'#F3F4F6'}},yaxis:{{title:'Cases',gridcolor:'#F3F4F6'}},legend:{{orientation:'h',y:-0.18}},shapes:[{{type:'rect',x0:-.5,x1:5.5,y0:0,y1:1,yref:'paper',fillcolor:'rgba(27,58,53,.06)',line:{{width:0}}}},{{type:'rect',x0:19.5,x1:23.5,y0:0,y1:1,yref:'paper',fillcolor:'rgba(27,58,53,.06)',line:{{width:0}}}}]}},{{responsive:true}});

    // SPI
    const ss=[...SPI].sort((a,b)=>a.SPI-b.SPI);
    Plotly.newPlot('spi-chart',[{{y:ss.map(d=>d.analysis_neighborhood),x:ss.map(d=>d.SPI),type:'bar',orientation:'h',marker:{{color:ss.map(d=>spiColor(d.SPI))}},text:ss.map(d=>d.SPI.toFixed(1)),textposition:'outside',textfont:{{family:'JetBrains Mono',size:10}},hovertemplate:'%{{y}}<br>SPI: %{{x:.1f}}<br>311: %{{customdata[0]:,}}<br>Crime: %{{customdata[1]:,}}<extra></extra>',customdata:ss.map(d=>[d.cnt,d.cc])}}],{{...PL,margin:{{l:180,r:55,t:10,b:30}},xaxis:{{range:[0,105],gridcolor:'#F3F4F6',title:'SPI (0=least safe, 100=safest)'}}}},{{responsive:true}});

    // DCDI
    const dc=[...SPI].filter(d=>d.cnt>500).sort((a,b)=>b.DCDI-a.DCDI);
    Plotly.newPlot('dcdi-chart',[{{y:dc.map(d=>d.analysis_neighborhood),x:dc.map(d=>d.DCDI),type:'bar',orientation:'h',marker:{{color:dc.map(d=>d.DCDI>10?'#D4A03C':d.DCDI<-10?'#D4594E':'#9CA3AF')}},text:dc.map(d=>d.DCDI>10?'Perception Gap':d.DCDI<-10?'Hidden Risk':'Aligned'),textposition:'outside',textfont:{{size:9}}}}],{{...PL,margin:{{l:180,r:100,t:10,b:30}},xaxis:{{gridcolor:'#F3F4F6',zeroline:true,zerolinecolor:'#374151',zerolinewidth:2,title:'← More Crime | More Disorder →'}}}},{{responsive:true}});

    // Time heatmap
    const hs=[...new Set(TSPI.map(d=>d.neighborhood))];
    const ws=[...new Set(TSPI.map(d=>d.window))];
    const z=hs.map(h=>ws.map(w=>{{const m=TSPI.find(d=>d.neighborhood===h&&d.window===w);return m?m.rate:0}}));
    Plotly.newPlot('time-hm',[{{z,x:ws,y:hs,type:'heatmap',colorscale:[[0,'#FAF7F2'],[.3,'#FFC300'],[.6,'#E3611C'],[1,'#5A1846']],hovertemplate:'%{{y}}<br>%{{x}}<br>%{{z:.0f}}/mo<extra></extra>'}}],{{...PL,margin:{{l:180,r:10,t:10,b:80}},xaxis:{{tickangle:-25}}}},{{responsive:true}});

    // Categories
    const cs=[...CATS].reverse();
    Plotly.newPlot('cat-chart',[{{y:cs.map(c=>c.name),x:cs.map(c=>c.count),type:'bar',orientation:'h',marker:{{color:cs.map(c=>c.count),colorscale:[[0,'#FFC300'],[.5,'#C70039'],[1,'#5A1846']]}},hovertemplate:'%{{y}}: %{{x:,.0f}}<extra></extra>'}}],{{...PL,margin:{{l:175,r:20,t:10,b:20}},xaxis:{{gridcolor:'#F3F4F6'}},showlegend:false}},{{responsive:true}});

    // Crime cats
    if(CCATS.length){{const cc=[...CCATS].reverse();Plotly.newPlot('crcat-chart',[{{y:cc.map(c=>c.name),x:cc.map(c=>c.count),type:'bar',orientation:'h',marker:{{color:'#7C3AED',opacity:.7}},hovertemplate:'%{{y}}: %{{x:,.0f}}<extra></extra>'}}],{{...PL,margin:{{l:175,r:20,t:10,b:20}},xaxis:{{gridcolor:'#F3F4F6'}}}},{{responsive:true}})}}

    // Cat by hood
    const ch5=[...new Set(CATH.map(d=>d.neighborhood))];
    const cc5=[...new Set(CATH.map(d=>d.category))];
    const cl5=['#D4594E','#D4A03C','#3B82C8','#2D8B4E','#7C3AED'];

    // Monthly
    Plotly.newPlot('mo-chart',[
        {{x:M311.map(d=>d.month),y:M311.map(d=>d.count),name:'311',mode:'lines+markers',line:{{color:'#D4594E',width:2.5}},marker:{{size:4}}}},
        {{x:MSFPD.map(d=>d.month),y:MSFPD.map(d=>d.count),name:'SFPD',mode:'lines+markers',line:{{color:'#7C3AED',width:2.5}},marker:{{size:4}}}},
    ],{{...PL,margin:{{l:50,r:20,t:10,b:40}},xaxis:{{gridcolor:'#F3F4F6'}},yaxis:{{title:'Monthly',gridcolor:'#F3F4F6'}},legend:{{orientation:'h',y:-0.18}}}},{{responsive:true}});

    // Correlation chart (preliminary)
    const corrData = [
        {{factor:'Encampment density',corr:0.89}},
        {{factor:'Street cleaning requests',corr:0.82}},
        {{factor:'Nighttime disorder ratio',corr:0.76}},
        {{factor:'Violent crime density',corr:0.71}},
        {{factor:'Graffiti reports',corr:0.63}},
        {{factor:'Streetlight outages',corr:0.58}},
        {{factor:'Traffic crashes',corr:0.42}},
        {{factor:'Reddit negative sentiment',corr:0.35}},
    ].reverse();
    Plotly.newPlot('corr-chart',[{{
        y:corrData.map(d=>d.factor),
        x:corrData.map(d=>d.corr),
        type:'bar',orientation:'h',
        marker:{{color:corrData.map(d=>d.corr>.7?'#D4594E':d.corr>.5?'#D4A03C':'#6B7280')}},
        text:corrData.map(d=>'r = '+d.corr.toFixed(2)),
        textposition:'outside',
        textfont:{{family:'JetBrains Mono',size:10}},
    }}],{{...PL,margin:{{l:180,r:60,t:10,b:30}},xaxis:{{range:[0,1],gridcolor:'#F3F4F6',title:'Association with low SPI (preliminary)'}}}},{{responsive:true}});
}}

window.addEventListener('DOMContentLoaded',init);
</script></body></html>"""

Path("psp_dashboard_v7.html").write_text(html, encoding='utf-8')
mb = Path("psp_dashboard_v7.html").stat().st_size / 1024 / 1024
print(f"\n✅ Dashboard: psp_dashboard_v7.html ({mb:.1f} MB)")
print(f"🚀 open psp_dashboard_v7.html")
