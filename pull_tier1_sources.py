#!/usr/bin/env python3
"""
pull_tier1_sources.py — Pull 8 Tier 1 (free, public, high-impact) data sources
for PSP's enhanced 6-layer model.

Sources:
  L2: Streetlight inventory (~25K fixtures)
  L4: Weather (Open-Meteo), Daylight (astronomical calc), AQI (Open-Meteo)
  L5: Building permits, Event permits, Outdoor dining permits, Bay Wheels trips

All sources are free APIs or public downloads. Robust fallbacks generate
synthetic data from SF climate normals when APIs are unavailable.

Usage:
    python pull_tier1_sources.py              # Pull all sources
    python pull_tier1_sources.py --source weather  # Pull one source
    python pull_tier1_sources.py --demo       # Generate all synthetic data
    python pull_tier1_sources.py --list       # Show available sources

Output: parquet files in data/raw/layer{2,4,5}/ directories
"""

import os
import sys
import json
import time
import argparse
import math
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

try:
    import requests
except ImportError:
    print("ERROR: requests library required. Run: pip install requests")
    sys.exit(1)

# ─────────────────────────── Constants ───────────────────────────

SF_LAT, SF_LON = 37.7749, -122.4194
SOCRATA_BASE = "https://data.sfgov.org/resource"
SOCRATA_BATCH = 10000
SOCRATA_DELAY = 1.0  # seconds between batches


def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in ['data/raw/layer2', 'data/raw/layer3', 'data/raw/layer4', 'data/raw/layer5']:
        os.makedirs(d, exist_ok=True)


def socrata_pull(dataset_id, params=None, max_records=100000):
    """Pull data from DataSF Socrata API with pagination."""
    url = f"{SOCRATA_BASE}/{dataset_id}.json"
    all_records = []
    offset = 0
    base_params = params or {}
    
    while offset < max_records:
        p = {**base_params, "$limit": SOCRATA_BATCH, "$offset": offset}
        try:
            resp = requests.get(url, params=p, timeout=60)
            if resp.status_code != 200:
                print(f"  API returned {resp.status_code} at offset {offset}")
                break
            batch = resp.json()
            if not batch:
                break
            all_records.extend(batch)
            print(f"  Pulled {len(all_records)} records...", end='\r')
            offset += len(batch)
            if len(batch) < SOCRATA_BATCH:
                break
            time.sleep(SOCRATA_DELAY)
        except requests.exceptions.RequestException as e:
            print(f"  Request error: {e}")
            break
    
    print(f"  Total: {len(all_records)} records" + " " * 20)
    return all_records


# ─────────────────────────── Source 1: Weather (Open-Meteo) ───────────────────────────

def pull_weather(demo=False):
    """Pull hourly weather data from Open-Meteo API (free, no key needed)."""
    output = "data/raw/layer4/weather_hourly.parquet"
    print("\n[1/8] Weather (Open-Meteo)...")
    
    if demo:
        return generate_synthetic_weather(output)
    
    # Pull last 90 days of historical data
    end = datetime.now()
    start = end - timedelta(days=90)
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": SF_LAT,
        "longitude": SF_LON,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,rain,wind_speed_10m,visibility,cloud_cover,weather_code",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "past_days": 90,
        "forecast_days": 1,
        "timezone": "America/Los_Angeles"
    }
    
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        hourly = data.get('hourly', {})
        if not hourly or 'time' not in hourly:
            print("  No hourly data returned, using synthetic fallback")
            return generate_synthetic_weather(output)
        
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(hourly['time']),
            'temp_f': hourly.get('temperature_2m', []),
            'humidity_pct': hourly.get('relative_humidity_2m', []),
            'precipitation_in': hourly.get('precipitation', []),
            'rain_in': hourly.get('rain', []),
            'wind_mph': hourly.get('wind_speed_10m', []),
            'visibility_ft': [v * 3280.84 if v else None for v in hourly.get('visibility', [])],  # m to ft
            'cloud_cover_pct': hourly.get('cloud_cover', []),
            'weather_code': hourly.get('weather_code', []),
        })
        
        # Compute weather discomfort index (0=pleasant, 100=miserable)
        df['weather_discomfort'] = compute_weather_discomfort(df)
        
        # Fog indicator (SF-specific: visibility < 1 mile = 5280 ft)
        df['is_fog'] = (df['visibility_ft'] < 5280).astype(int) if 'visibility_ft' in df else 0
        
        df.to_parquet(output, index=False)
        print(f"  Saved {len(df)} hourly records to {output}")
        return df
        
    except Exception as e:
        print(f"  Open-Meteo error: {e}")
        print("  Using synthetic fallback...")
        return generate_synthetic_weather(output)


def compute_weather_discomfort(df):
    """Compute 0-100 discomfort index from weather variables."""
    score = np.zeros(len(df))
    
    # Temperature discomfort: comfortable = 55-70°F
    if 'temp_f' in df:
        temp = pd.to_numeric(df['temp_f'], errors='coerce').fillna(60)
        score += np.where(temp < 45, (45 - temp) * 2, 0)  # Cold
        score += np.where(temp > 80, (temp - 80) * 2, 0)  # Hot
    
    # Rain discomfort
    if 'rain_in' in df:
        rain = pd.to_numeric(df['rain_in'], errors='coerce').fillna(0)
        score += np.minimum(rain * 50, 30)
    
    # Wind discomfort
    if 'wind_mph' in df:
        wind = pd.to_numeric(df['wind_mph'], errors='coerce').fillna(5)
        score += np.where(wind > 15, (wind - 15) * 1.5, 0)
    
    # Low visibility discomfort
    if 'visibility_ft' in df:
        vis = pd.to_numeric(df['visibility_ft'], errors='coerce').fillna(30000)
        score += np.where(vis < 5280, (5280 - vis) / 5280 * 20, 0)
    
    return np.clip(score, 0, 100).round(1)


def generate_synthetic_weather(output):
    """Generate realistic SF weather data from climate normals."""
    print("  Generating synthetic weather from SF climate normals...")
    hours = pd.date_range(end=datetime.now(), periods=90*24, freq='h', tz='America/Los_Angeles')
    
    # SF climate normals: mild, foggy summers, rainy winters
    month = hours.month
    hour = hours.hour
    
    # Temperature: 50-65°F range, cooler in summer fog, warmer in fall
    base_temp = 55 + 5 * np.sin(2 * np.pi * (month - 4) / 12)
    diurnal = 5 * np.sin(2 * np.pi * (hour - 6) / 24)
    temp = base_temp + diurnal + np.random.normal(0, 3, len(hours))
    
    # Rain: mostly Nov-Mar
    rain_prob = np.where((month >= 11) | (month <= 3), 0.15, 0.02)
    rain = np.where(np.random.random(len(hours)) < rain_prob, 
                    np.random.exponential(0.05, len(hours)), 0)
    
    # Fog: mostly Jun-Sep mornings
    fog_prob = np.where((month >= 6) & (month <= 9) & (hour < 11), 0.4, 0.05)
    vis = np.where(np.random.random(len(hours)) < fog_prob,
                   np.random.uniform(500, 5000, len(hours)),
                   np.random.uniform(15000, 50000, len(hours)))
    
    df = pd.DataFrame({
        'timestamp': hours.tz_localize(None),
        'temp_f': temp.round(1),
        'humidity_pct': np.clip(70 + np.random.normal(0, 15, len(hours)), 30, 100).round(0),
        'precipitation_in': rain.round(3),
        'rain_in': rain.round(3),
        'wind_mph': np.clip(8 + np.random.normal(0, 5, len(hours)), 0, 40).round(1),
        'visibility_ft': vis.round(0),
        'cloud_cover_pct': np.clip(40 + np.random.normal(0, 25, len(hours)), 0, 100).round(0),
        'weather_code': 0,
        'is_fog': (vis < 5280).astype(int),
    })
    
    df['weather_discomfort'] = compute_weather_discomfort(df)
    df.to_parquet(output, index=False)
    print(f"  Saved {len(df)} synthetic hourly records to {output}")
    return df


# ─────────────────────────── Source 2: Daylight ───────────────────────────

def pull_daylight(demo=False):
    """Compute daylight quality per hour using astronomical formulas."""
    output = "data/raw/layer4/daylight_hourly.parquet"
    print("\n[2/8] Daylight (astronomical calculation)...")
    
    hours = pd.date_range(end=datetime.now(), periods=365*24, freq='h', tz='America/Los_Angeles')
    
    records = []
    for ts in hours:
        doy = ts.timetuple().tm_yday
        hour = ts.hour + ts.minute / 60
        
        # Solar declination
        decl = 23.45 * math.sin(math.radians(360 / 365 * (doy - 81)))
        
        # Hour angle at sunrise/sunset
        lat_rad = math.radians(SF_LAT)
        decl_rad = math.radians(decl)
        cos_ha = -math.tan(lat_rad) * math.tan(decl_rad)
        cos_ha = max(-1, min(1, cos_ha))
        ha = math.degrees(math.acos(cos_ha))
        
        # Sunrise/sunset in hours (approximate, local solar time)
        sunrise = 12 - ha / 15 + 0.5  # +0.5 rough timezone/equation of time offset
        sunset = 12 + ha / 15 + 0.5
        daylight_hours = 2 * ha / 15
        
        # Daylight quality for this hour (0-100)
        if hour < sunrise - 0.5 or hour > sunset + 0.5:
            quality = 0  # Full dark
        elif hour < sunrise + 0.5:
            quality = (hour - (sunrise - 0.5)) * 100  # Dawn transition
        elif hour > sunset - 0.5:
            quality = (sunset + 0.5 - hour) * 100  # Dusk transition
        else:
            quality = 100  # Full daylight
        
        quality = max(0, min(100, quality))
        
        records.append({
            'timestamp': ts.tz_localize(None),
            'sunrise_hour': round(sunrise, 2),
            'sunset_hour': round(sunset, 2),
            'daylight_hours': round(daylight_hours, 2),
            'daylight_quality': round(quality, 1),
            'is_dark': 1 if quality < 20 else 0,
        })
    
    df = pd.DataFrame(records)
    df.to_parquet(output, index=False)
    print(f"  Saved {len(df)} hourly records to {output}")
    return df


# ─────────────────────────── Source 3: Air Quality ───────────────────────────

def pull_aqi(demo=False):
    """Pull AQI data from Open-Meteo Air Quality API."""
    output = "data/raw/layer4/aqi_hourly.parquet"
    print("\n[3/8] Air Quality (Open-Meteo)...")
    
    if demo:
        return generate_synthetic_aqi(output)
    
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": SF_LAT,
        "longitude": SF_LON,
        "hourly": "us_aqi,pm2_5,pm10,ozone,nitrogen_dioxide",
        "past_days": 90,
        "forecast_days": 1,
        "timezone": "America/Los_Angeles"
    }
    
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        hourly = data.get('hourly', {})
        if not hourly:
            return generate_synthetic_aqi(output)
        
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(hourly['time']),
            'us_aqi': hourly.get('us_aqi', []),
            'pm25': hourly.get('pm2_5', []),
            'pm10': hourly.get('pm10', []),
            'ozone': hourly.get('ozone', []),
            'no2': hourly.get('nitrogen_dioxide', []),
        })
        
        # AQI discomfort (0 = good air, 100 = hazardous)
        aqi = pd.to_numeric(df['us_aqi'], errors='coerce').fillna(30)
        df['aqi_discomfort'] = np.clip(aqi / 3, 0, 100).round(1)
        df['is_smoke_day'] = (aqi > 100).astype(int)
        
        df.to_parquet(output, index=False)
        print(f"  Saved {len(df)} hourly records to {output}")
        return df
        
    except Exception as e:
        print(f"  AQI API error: {e}")
        return generate_synthetic_aqi(output)


def generate_synthetic_aqi(output):
    """Generate realistic SF AQI data."""
    print("  Generating synthetic AQI data...")
    hours = pd.date_range(end=datetime.now(), periods=90*24, freq='h')
    
    # SF typically has good AQI (20-50) except during fire season (Aug-Oct)
    month = hours.month
    base_aqi = np.where((month >= 8) & (month <= 10), 55, 30)
    aqi = base_aqi + np.random.normal(0, 15, len(hours))
    
    # Occasional smoke days
    smoke = np.random.random(len(hours)) < 0.02
    aqi = np.where(smoke, aqi + np.random.uniform(50, 150, len(hours)), aqi)
    aqi = np.clip(aqi, 5, 300)
    
    df = pd.DataFrame({
        'timestamp': hours,
        'us_aqi': aqi.round(0),
        'pm25': (aqi * 0.4 + np.random.normal(0, 3, len(hours))).clip(0).round(1),
        'pm10': (aqi * 0.6 + np.random.normal(0, 5, len(hours))).clip(0).round(1),
        'ozone': np.clip(30 + np.random.normal(0, 10, len(hours)), 0, 150).round(1),
        'no2': np.clip(15 + np.random.normal(0, 8, len(hours)), 0, 100).round(1),
        'aqi_discomfort': np.clip(aqi / 3, 0, 100).round(1),
        'is_smoke_day': (aqi > 100).astype(int),
    })
    
    df.to_parquet(output, index=False)
    print(f"  Saved {len(df)} synthetic hourly records to {output}")
    return df


# ─────────────────────────── Source 4: Streetlight Inventory ───────────────────────────

def pull_streetlights(demo=False):
    """Pull full streetlight fixture inventory from DataSF."""
    output = "data/raw/layer2/streetlight_inventory.parquet"
    print("\n[4/8] Streetlight Inventory (DataSF)...")
    
    if demo:
        print("  Using demo mode — skipping API pull")
        return None
    
    # Dataset: Street Light List
    # Try several known dataset IDs for streetlight data
    dataset_ids = ['fbux-9tqf', 'vw6y-z8j6']  # streetlight-specific, fallback to 311
    
    # First try the streetlight-specific dataset
    records = socrata_pull('fbux-9tqf', max_records=50000)
    
    if records:
        df = pd.DataFrame(records)
        df.to_parquet(output, index=False)
        print(f"  Saved {len(df)} streetlight records to {output}")
        return df
    
    # Fallback: extract streetlight info from 311
    print("  Streetlight dataset not found, extracting from 311 data...")
    try:
        cases = pd.read_parquet("data/raw/layer2/311_cases_all.parquet")
        light_kw = ['streetlight', 'street light', 'light out', 'lighting', 'lamp']
        if 'category' in cases.columns:
            mask = cases['category'].str.lower().str.contains('|'.join(light_kw), na=False)
        elif 'request_type' in cases.columns:
            mask = cases['request_type'].str.lower().str.contains('|'.join(light_kw), na=False)
        else:
            mask = pd.Series([False] * len(cases))
        
        lights = cases[mask].copy()
        if len(lights) > 0:
            lights.to_parquet(output, index=False)
            print(f"  Extracted {len(lights)} streetlight-related 311 records to {output}")
            return lights
    except FileNotFoundError:
        pass
    
    print("  No streetlight data available")
    return None


# ─────────────────────────── Source 5: Building Permits ───────────────────────────

def pull_building_permits(demo=False):
    """Pull building permits from DataSF (investment signal)."""
    output = "data/raw/layer5/building_permits.parquet"
    print("\n[5/8] Building Permits (DataSF)...")
    
    if demo:
        print("  Using demo mode — skipping API pull")
        return None
    
    # Dataset: Building Permits (DBI)
    # Field: neighborhoods_analysis_boundaries
    two_years_ago = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%dT00:00:00')
    
    records = socrata_pull(
        'i98e-djp9',
        params={
            "$where": f"filed_date > '{two_years_ago}'",
            "$order": "filed_date DESC"
        },
        max_records=50000
    )
    
    if records:
        df = pd.DataFrame(records)
        df.to_parquet(output, index=False)
        print(f"  Saved {len(df)} building permit records to {output}")
        return df
    
    print("  Building permits API returned no data")
    return None


# ─────────────────────────── Source 6: Event Permits ───────────────────────────

def pull_event_permits(demo=False):
    """Pull event permits from DataSF (activation signal)."""
    output = "data/raw/layer5/event_permits.parquet"
    print("\n[6/8] Event Permits (DataSF)...")
    
    if demo:
        print("  Using demo mode — skipping API pull")
        return None
    
    # Entertainment Commission permits
    records = socrata_pull(
        'vz4q-5t2a',
        params={"$order": ":id"},
        max_records=20000
    )
    
    if not records:
        # Try special events permits
        records = socrata_pull(
            'aibk-2r77',
            params={"$order": ":id"},
            max_records=20000
        )
    
    if records:
        df = pd.DataFrame(records)
        df.to_parquet(output, index=False)
        print(f"  Saved {len(df)} event permit records to {output}")
        return df
    
    print("  Event permits API returned no data")
    return None


# ─────────────────────────── Source 7: Outdoor Dining ───────────────────────────

def pull_outdoor_dining(demo=False):
    """Pull outdoor dining/Shared Spaces permits from DataSF."""
    output = "data/raw/layer5/outdoor_dining_permits.parquet"
    print("\n[7/8] Outdoor Dining Permits (DataSF)...")
    
    if demo:
        print("  Using demo mode — skipping API pull")
        return None
    
    # Shared Spaces permits (post-COVID outdoor dining program)
    records = socrata_pull(
        'j4sj-j2nf',
        max_records=10000
    )
    
    if records:
        df = pd.DataFrame(records)
        df.to_parquet(output, index=False)
        print(f"  Saved {len(df)} outdoor dining records to {output}")
        return df
    
    print("  Outdoor dining API returned no data")
    return None


# ─────────────────────────── Source 8: Bay Wheels ───────────────────────────

def pull_baywheels(demo=False):
    """
    Pull Bay Wheels (Lyft) bike share trip data.
    Note: Bay Wheels publishes monthly CSV exports, not a real-time API.
    Data available at: https://www.lyft.com/bikes/bay-wheels/system-data
    """
    output = "data/raw/layer3/baywheels_trips.parquet"
    print("\n[8/8] Bay Wheels Trips...")
    
    if demo:
        return generate_synthetic_baywheels(output)
    
    # Try to download recent months from Lyft's public S3 bucket
    # Format: https://s3.amazonaws.com/baywheels-data/YYYYMM-baywheels-tripdata.csv.zip
    base_url = "https://s3.amazonaws.com/baywheels-data"
    
    all_dfs = []
    now = datetime.now()
    
    for months_ago in range(1, 7):  # Try last 6 months
        target = now - timedelta(days=30 * months_ago)
        filename = f"{target.strftime('%Y%m')}-baywheels-tripdata.csv.zip"
        url = f"{base_url}/{filename}"
        
        try:
            resp = requests.get(url, timeout=30, stream=True)
            if resp.status_code == 200:
                # Save temporarily and read
                import tempfile, zipfile, io
                content = io.BytesIO(resp.content)
                with zipfile.ZipFile(content) as zf:
                    csv_names = [n for n in zf.namelist() if n.endswith('.csv')]
                    if csv_names:
                        with zf.open(csv_names[0]) as f:
                            df = pd.read_csv(f)
                            all_dfs.append(df)
                            print(f"  Downloaded {filename}: {len(df)} trips")
            else:
                print(f"  {filename}: not available (HTTP {resp.status_code})")
        except Exception as e:
            print(f"  {filename}: error - {e}")
    
    if all_dfs:
        df = pd.concat(all_dfs, ignore_index=True)
        
        # Filter to SF only (approximate bounding box)
        if 'start_lat' in df.columns:
            sf_mask = (
                (df['start_lat'].between(37.7, 37.82)) &
                (df['start_lng'].between(-122.52, -122.35))
            )
            df = df[sf_mask].copy()
        
        df.to_parquet(output, index=False)
        print(f"  Saved {len(df)} SF trips to {output}")
        return df
    
    print("  No Bay Wheels data downloaded, generating synthetic...")
    return generate_synthetic_baywheels(output)


def generate_synthetic_baywheels(output):
    """Generate synthetic Bay Wheels trip data."""
    print("  Generating synthetic Bay Wheels data...")
    
    # Major Bay Wheels stations in SF with approximate coords
    stations = {
        'Market St at 10th': (37.7764, -122.4174, 'South of Market'),
        'Powell St BART': (37.7844, -122.4080, 'Tenderloin'),
        'Embarcadero at Sansome': (37.8009, -122.4032, 'Financial District/South Beach'),
        'Townsend at 4th': (37.7809, -122.3924, 'South of Market'),
        'Valencia at 16th': (37.7650, -122.4218, 'Mission'),
        'Howard at 2nd': (37.7866, -122.3977, 'South of Market'),
        'Steuart at Market': (37.7949, -122.3945, 'Financial District/South Beach'),
        'Golden Gate at Polk': (37.7814, -122.4189, 'Tenderloin'),
        'Berry at 4th': (37.7759, -122.3907, 'South of Market'),
        'Fell at Divisadero': (37.7726, -122.4383, 'Western Addition'),
    }
    
    n_trips = 50000  # 50K synthetic trips
    station_names = list(stations.keys())
    
    # Generate trips with realistic patterns
    start_stations = np.random.choice(station_names, n_trips, 
                                       p=[0.15, 0.08, 0.15, 0.12, 0.10, 0.10, 0.10, 0.05, 0.08, 0.07])
    
    records = []
    base_date = datetime.now() - timedelta(days=180)
    
    for i in range(n_trips):
        start = start_stations[i]
        end = np.random.choice(station_names)
        s_lat, s_lng, s_nhood = stations[start]
        e_lat, e_lng, e_nhood = stations[end]
        
        # Time: peak at 8-9am and 5-6pm on weekdays
        day_offset = np.random.randint(0, 180)
        hour = np.random.choice(range(24), p=[
            0.01, 0.005, 0.005, 0.005, 0.005, 0.01, 0.02, 0.06, 0.10, 0.08,
            0.06, 0.05, 0.06, 0.05, 0.04, 0.04, 0.06, 0.10, 0.08, 0.05,
            0.03, 0.02, 0.015, 0.01
        ])
        
        ts = base_date + timedelta(days=day_offset, hours=hour, minutes=np.random.randint(0, 60))
        duration = np.random.lognormal(np.log(10), 0.6)  # ~10 min median
        
        records.append({
            'started_at': ts.isoformat(),
            'ended_at': (ts + timedelta(minutes=duration)).isoformat(),
            'start_station_name': start,
            'end_station_name': end,
            'start_lat': s_lat + np.random.normal(0, 0.001),
            'start_lng': s_lng + np.random.normal(0, 0.001),
            'end_lat': e_lat + np.random.normal(0, 0.001),
            'end_lng': e_lng + np.random.normal(0, 0.001),
            'start_neighborhood': s_nhood,
            'end_neighborhood': e_nhood,
            'duration_min': round(duration, 1),
            'member_casual': np.random.choice(['member', 'casual'], p=[0.7, 0.3]),
        })
    
    df = pd.DataFrame(records)
    df.to_parquet(output, index=False)
    print(f"  Saved {len(df)} synthetic trips to {output}")
    return df


# ─────────────────────────── Main ───────────────────────────

SOURCES = {
    'weather': pull_weather,
    'daylight': pull_daylight,
    'aqi': pull_aqi,
    'streetlights': pull_streetlights,
    'building_permits': pull_building_permits,
    'event_permits': pull_event_permits,
    'outdoor_dining': pull_outdoor_dining,
    'baywheels': pull_baywheels,
}


def main():
    parser = argparse.ArgumentParser(description='Pull Tier 1 data sources for PSP')
    parser.add_argument('--source', choices=list(SOURCES.keys()),
                        help='Pull a specific source only')
    parser.add_argument('--demo', action='store_true',
                        help='Generate synthetic data (no API calls)')
    parser.add_argument('--list', action='store_true',
                        help='List available sources')
    args = parser.parse_args()

    if args.list:
        print("Available Tier 1 sources:")
        for name in SOURCES:
            print(f"  {name}")
        return

    ensure_dirs()
    
    if args.source:
        SOURCES[args.source](demo=args.demo)
    else:
        print("=" * 60)
        print("PSP Tier 1 Data Source Pull")
        print(f"Mode: {'DEMO (synthetic data)' if args.demo else 'LIVE (API calls)'}")
        print("=" * 60)
        
        for name, func in SOURCES.items():
            try:
                func(demo=args.demo)
            except Exception as e:
                print(f"  ERROR pulling {name}: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("Tier 1 pull complete. Check data/raw/layer{2,4,5}/ for output files.")
        print("=" * 60)


if __name__ == '__main__':
    main()
