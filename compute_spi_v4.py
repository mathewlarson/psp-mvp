#!/usr/bin/env python3
"""
Public Safety Pulse — SPI v4 Computation Engine
================================================
Computes a 15-component Safety Perception Index per SF neighborhood
per 4-hour time window. Outputs enriched GeoJSON for dashboard rendering.

Usage:
    cd ~/Desktop/psp-mvp
    source venv/bin/activate
    python compute_spi_v4.py

Output:
    data/spi_v4_output.json  — GeoJSON with SPI scores in feature properties
    data/spi_v4_summary.json — Summary stats for dashboard metadata

Dependencies: pandas, numpy, geopandas (optional), json, pathlib
"""

import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

warnings.filterwarnings("ignore")

# Try pandas import
try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed. Run: pip install pandas")
    exit(1)

# ─── CONFIGURATION ──────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
GEOJSON_PATH = DATA_DIR / "sf_neighborhoods.geojson"
OUTPUT_PATH = DATA_DIR / "spi_v4_output.json"
SUMMARY_PATH = DATA_DIR / "spi_v4_summary.json"

# Time windows (6 x 4-hour blocks)
TIME_WINDOWS = {
    "early_morning": (4, 8),    # 4am-8am
    "morning":       (8, 12),   # 8am-12pm
    "afternoon":     (12, 16),  # 12pm-4pm
    "evening":       (16, 20),  # 4pm-8pm
    "night":         (20, 24),  # 8pm-12am
    "late_night":    (0, 4),    # 12am-4am
}

# SPI v4 component weights (from v4 guide)
WEIGHTS = {
    "disorder_density":       0.18,
    "crime_severity":         0.14,
    "disorder_salience":      0.08,
    "pedestrian_safety":      0.05,
    "temporal_risk":          0.08,
    "resolution_responsive":  0.10,
    "review_sentiment":       0.08,
    "transit_confidence":     0.06,
    "lighting_risk":          0.05,
    "business_vitality":      0.04,
    "emergency_density":      0.04,
    "environmental_quality":  0.04,
    "digital_concern":        0.02,
    "noise_disorder":         0.02,
    "waste_odor":             0.02,
}

# Crime severity weights by category
CRIME_SEVERITY = {
    "homicide":              10.0,
    "rape":                   9.0,
    "robbery":                8.0,
    "assault":                7.0,
    "aggravated assault":     7.0,
    "arson":                  6.0,
    "burglary":               5.0,
    "motor vehicle theft":    4.0,
    "larceny theft":          3.0,
    "drug offense":           3.0,
    "drug/narcotic":          3.0,
    "vandalism":              2.0,
    "warrant":                1.5,
    "other offenses":         1.0,
    "non-criminal":           0.5,
    "suspicious occ":         1.0,
    "malicious mischief":     2.0,
    "weapons offense":        6.0,
    "weapon laws":            6.0,
    "sex offense":            7.0,
    "kidnapping":             9.0,
    "disorderly conduct":     1.5,
    "trespass":               1.0,
    "fraud":                  2.0,
    "forgery/counterfeiting": 2.0,
    "stolen property":        3.0,
    "prostitution":           2.0,
    "embezzlement":           2.0,
    "recovered vehicle":      0.5,
    "missing person":         3.0,
    "traffic violation arrest":1.0,
    "other miscellaneous":    1.0,
    "miscellaneous investigation":1.0,
}

# 311 disorder salience weights (how much each category affects perception)
DISORDER_SALIENCE = {
    "encampment":                    1.0,
    "homeless concerns":             0.9,
    "human/animal waste":            0.9,
    "street and sidewalk cleaning":  0.7,
    "needles":                       0.95,
    "graffiti public":               0.5,
    "graffiti private":              0.4,
    "noise":                         0.6,
    "blocked street and sidewalk":   0.3,
    "damaged property":              0.5,
    "damage property":               0.5,
    "streetlights":                  0.7,
    "illegal postings":              0.2,
    "general request":               0.3,
    "sewer":                         0.4,
    "rpd general":                   0.2,
    "litter receptacle maintenance": 0.5,
}

# Approximate neighborhood areas in km² (for density normalization)
NEIGHBORHOOD_AREAS_KM2 = {
    "Bayview Hunters Point": 6.8,
    "Bernal Heights": 2.4,
    "Castro/Upper Market": 1.5,
    "Chinatown": 0.5,
    "Excelsior": 3.2,
    "Financial District/South Beach": 2.1,
    "Glen Park": 2.0,
    "Golden Gate Park": 4.1,
    "Haight Ashbury": 1.2,
    "Hayes Valley": 0.8,
    "Inner Richmond": 2.3,
    "Inner Sunset": 2.5,
    "Japantown": 0.4,
    "Lakeshore": 5.5,
    "Lincoln Park": 1.8,
    "Lone Mountain/USF": 1.0,
    "Marina": 1.5,
    "McLaren Park": 2.8,
    "Mission": 2.5,
    "Mission Bay": 1.2,
    "Nob Hill": 0.8,
    "Noe Valley": 1.8,
    "North Beach": 0.9,
    "Oceanview/Merced/Ingleside": 3.5,
    "Outer Mission": 2.2,
    "Outer Richmond": 3.8,
    "Outer Sunset": 5.5,
    "Pacific Heights": 1.5,
    "Portola": 2.0,
    "Potrero Hill": 2.2,
    "Presidio": 4.8,
    "Presidio Heights": 0.8,
    "Russian Hill": 0.7,
    "Seacliff": 0.6,
    "South of Market": 3.0,
    "Sunset/Parkside": 5.0,
    "Tenderloin": 0.6,
    "Treasure Island": 2.2,
    "Twin Peaks": 1.5,
    "Visitacion Valley": 2.5,
    "West of Twin Peaks": 3.5,
}


def load_parquet_safe(path, name=""):
    """Load parquet with error handling."""
    p = Path(path)
    if p.exists():
        try:
            df = pd.read_parquet(p)
            print(f"  ✅ {name}: {len(df):,} records")
            return df
        except Exception as e:
            print(f"  ❌ {name}: Error loading — {e}")
            return pd.DataFrame()
    else:
        print(f"  ⚠️  {name}: File not found at {p}")
        return pd.DataFrame()


def z_score_cap(series, cap=3.0):
    """Z-score normalize with soft cap at ±3σ."""
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    z = (series - mean) / std
    return z.clip(-cap, cap)


def get_neighborhood_field(df):
    """Auto-detect neighborhood column name."""
    for col in ["analysis_neighborhood", "neighborhoods_analysis_boundaries", "neighborhood", "nhood"]:
        if col in df.columns:
            return col
    return None


def extract_hour(df):
    """Extract hour from datetime columns."""
    for col in ["incident_datetime", "requested_datetime", "received_dttm", "call_date", "datetime"]:
        if col in df.columns:
            try:
                df["_hour"] = pd.to_datetime(df[col], errors="coerce").dt.hour
                return df
            except:
                continue
    df["_hour"] = np.nan
    return df


def assign_time_window(hour):
    """Map hour to time window name."""
    if pd.isna(hour):
        return "all_day"
    h = int(hour)
    if 4 <= h < 8: return "early_morning"
    if 8 <= h < 12: return "morning"
    if 12 <= h < 16: return "afternoon"
    if 16 <= h < 20: return "evening"
    if 20 <= h < 24: return "night"
    return "late_night"


# ─── COMPONENT COMPUTATION FUNCTIONS ────────────────────────────────────────

def compute_disorder_density(df_311, hoods):
    """Component 1: 311 disorder case density per km²."""
    nhood_col = get_neighborhood_field(df_311)
    if nhood_col is None or df_311.empty:
        return pd.Series(0.0, index=hoods)
    counts = df_311.groupby(nhood_col).size()
    density = pd.Series(0.0, index=hoods)
    for hood in hoods:
        area = NEIGHBORHOOD_AREAS_KM2.get(hood, 2.0)
        density[hood] = counts.get(hood, 0) / area
    return density


def compute_disorder_density_by_window(df_311, hoods):
    """Component 1 by time window."""
    nhood_col = get_neighborhood_field(df_311)
    result = {}
    if nhood_col is None or df_311.empty:
        for tw in TIME_WINDOWS:
            result[tw] = pd.Series(0.0, index=hoods)
        return result
    df_311 = extract_hour(df_311)
    df_311["_tw"] = df_311["_hour"].apply(assign_time_window)
    for tw in TIME_WINDOWS:
        subset = df_311[df_311["_tw"] == tw]
        counts = subset.groupby(nhood_col).size()
        density = pd.Series(0.0, index=hoods)
        for hood in hoods:
            area = NEIGHBORHOOD_AREAS_KM2.get(hood, 2.0)
            density[hood] = counts.get(hood, 0) / area
        result[tw] = density
    return result


def compute_crime_severity(df_sfpd, hoods):
    """Component 2: Severity-weighted crime density per km²."""
    nhood_col = get_neighborhood_field(df_sfpd)
    if nhood_col is None or df_sfpd.empty:
        return pd.Series(0.0, index=hoods)

    cat_col = None
    for c in ["incident_category", "category", "incident_subcategory"]:
        if c in df_sfpd.columns:
            cat_col = c
            break

    severity = pd.Series(0.0, index=hoods)
    if cat_col:
        df_sfpd["_severity"] = df_sfpd[cat_col].str.lower().map(
            lambda x: max((v for k, v in CRIME_SEVERITY.items() if k in str(x)), default=1.0)
        )
    else:
        df_sfpd["_severity"] = 1.0

    weighted = df_sfpd.groupby(nhood_col)["_severity"].sum()
    for hood in hoods:
        area = NEIGHBORHOOD_AREAS_KM2.get(hood, 2.0)
        severity[hood] = weighted.get(hood, 0) / area
    return severity


def compute_crime_severity_by_window(df_sfpd, hoods):
    """Component 2 by time window."""
    nhood_col = get_neighborhood_field(df_sfpd)
    result = {}
    if nhood_col is None or df_sfpd.empty:
        for tw in TIME_WINDOWS:
            result[tw] = pd.Series(0.0, index=hoods)
        return result

    cat_col = None
    for c in ["incident_category", "category", "incident_subcategory"]:
        if c in df_sfpd.columns:
            cat_col = c
            break
    if cat_col:
        df_sfpd["_severity"] = df_sfpd[cat_col].str.lower().map(
            lambda x: max((v for k, v in CRIME_SEVERITY.items() if k in str(x)), default=1.0)
        )
    else:
        df_sfpd["_severity"] = 1.0

    df_sfpd = extract_hour(df_sfpd)
    df_sfpd["_tw"] = df_sfpd["_hour"].apply(assign_time_window)
    for tw in TIME_WINDOWS:
        subset = df_sfpd[df_sfpd["_tw"] == tw]
        weighted = subset.groupby(nhood_col)["_severity"].sum()
        sev = pd.Series(0.0, index=hoods)
        for hood in hoods:
            area = NEIGHBORHOOD_AREAS_KM2.get(hood, 2.0)
            sev[hood] = weighted.get(hood, 0) / area
        result[tw] = sev
    return result


def compute_disorder_salience(df_311, hoods):
    """Component 3: Mean salience weight of 311 categories per neighborhood."""
    nhood_col = get_neighborhood_field(df_311)
    if nhood_col is None or df_311.empty:
        return pd.Series(0.5, index=hoods)

    svc_col = None
    for c in ["service_name", "category", "service_subtype", "request_type"]:
        if c in df_311.columns:
            svc_col = c
            break

    if svc_col is None:
        return pd.Series(0.5, index=hoods)

    df_311["_salience"] = df_311[svc_col].str.lower().map(
        lambda x: max((v for k, v in DISORDER_SALIENCE.items() if k in str(x)), default=0.3)
    )
    mean_sal = df_311.groupby(nhood_col)["_salience"].mean()
    salience = pd.Series(0.5, index=hoods)
    for hood in hoods:
        if hood in mean_sal.index:
            salience[hood] = mean_sal[hood]
    return salience


def compute_pedestrian_safety(df_crashes, hoods):
    """Component 4: Traffic crash density per km²."""
    nhood_col = get_neighborhood_field(df_crashes)
    if nhood_col is None or df_crashes.empty:
        return pd.Series(0.0, index=hoods)
    counts = df_crashes.groupby(nhood_col).size()
    density = pd.Series(0.0, index=hoods)
    for hood in hoods:
        area = NEIGHBORHOOD_AREAS_KM2.get(hood, 2.0)
        density[hood] = counts.get(hood, 0) / area
    return density


def compute_temporal_risk(df_311, df_sfpd, hoods):
    """Component 5: Night concentration ratio + 3-month trend."""
    risk = pd.Series(0.5, index=hoods)

    # Night ratio from 311
    nhood_col_311 = get_neighborhood_field(df_311)
    if nhood_col_311 and not df_311.empty:
        df_311 = extract_hour(df_311)
        for hood in hoods:
            hood_data = df_311[df_311[nhood_col_311] == hood]
            if len(hood_data) > 0:
                night = hood_data[hood_data["_hour"].between(20, 23) | (hood_data["_hour"] < 4)]
                night_ratio = len(night) / len(hood_data) if len(hood_data) > 0 else 0.33
                risk[hood] = 0.6 * night_ratio + 0.4 * 0.5  # trend placeholder
    return risk


def compute_resolution_responsiveness(hoods):
    """Component 6: Median 311 resolution days per neighborhood."""
    # Try loading pre-computed resolution times
    res_path = RAW_DIR / "layer2" / "resolution_times.parquet"
    if res_path.exists():
        try:
            df = pd.read_parquet(res_path)
            nhood_col = get_neighborhood_field(df)
            if nhood_col:
                resp = pd.Series(7.0, index=hoods)  # default 7 days
                for _, row in df.iterrows():
                    hood = row.get(nhood_col, row.get("neighborhood", ""))
                    for col in ["median_resolution_days", "median_days", "resolution_days"]:
                        if col in df.columns:
                            resp[hood] = row[col]
                            break
                return resp
        except:
            pass
    return pd.Series(7.0, index=hoods)


def compute_review_sentiment(hoods):
    """Component 7: Yelp average rating per neighborhood."""
    yelp_path = RAW_DIR / "layer3" / "yelp_neighborhood_quality.parquet"
    sentiment = pd.Series(3.5, index=hoods)  # neutral default
    if yelp_path.exists():
        try:
            df = pd.read_parquet(yelp_path)
            nhood_col = get_neighborhood_field(df)
            if nhood_col is None:
                for c in df.columns:
                    if "neigh" in c.lower() or "hood" in c.lower() or "name" in c.lower():
                        nhood_col = c
                        break
            if nhood_col:
                for _, row in df.iterrows():
                    hood = row[nhood_col]
                    for col in ["avg_rating", "rating", "mean_rating", "average_rating"]:
                        if col in df.columns:
                            if hood in hoods:
                                sentiment[hood] = row[col]
                            break
        except:
            pass
    return sentiment


def compute_transit_confidence(hoods):
    """Component 8: BART exit year-over-year change (placeholder — needs BART parsing)."""
    # Map BART stations to neighborhoods
    BART_STATION_MAP = {
        "Embarcadero": "Financial District/South Beach",
        "Montgomery": "Financial District/South Beach",
        "Powell": "Tenderloin",
        "Civic Center": "Tenderloin",
        "16th St Mission": "Mission",
        "24th St Mission": "Mission",
        "Glen Park": "Glen Park",
        "Balboa Park": "Excelsior",
    }
    # Default neutral — will be overwritten if BART data parsed
    conf = pd.Series(0.0, index=hoods)

    bart_dir = RAW_DIR / "layer3" / "bart"
    if bart_dir.exists() and any(bart_dir.glob("*.xlsx")):
        print("  ℹ️  BART Excel files found but OD matrix parsing not yet implemented.")
        print("     Run Priority 2 (BART parsing) to populate this component.")
    return conf


def compute_lighting_risk(hoods):
    """Component 9: Streetlight outage density per km²."""
    sl_path = RAW_DIR / "layer2" / "streetlight_outages.parquet"
    risk = pd.Series(0.0, index=hoods)
    if sl_path.exists():
        try:
            df = pd.read_parquet(sl_path)
            nhood_col = get_neighborhood_field(df)
            if nhood_col:
                counts = df.groupby(nhood_col).size()
                for hood in hoods:
                    area = NEIGHBORHOOD_AREAS_KM2.get(hood, 2.0)
                    risk[hood] = counts.get(hood, 0) / area
        except:
            pass
    return risk


def compute_business_vitality(hoods):
    """Component 10: Net business openings vs closures."""
    bv_path = RAW_DIR / "layer3" / "business_vitality.parquet"
    vitality = pd.Series(0.0, index=hoods)
    if bv_path.exists():
        try:
            df = pd.read_parquet(bv_path)
            nhood_col = get_neighborhood_field(df)
            if nhood_col is None:
                for c in df.columns:
                    if "neigh" in c.lower() or "hood" in c.lower():
                        nhood_col = c
                        break
            if nhood_col:
                for _, row in df.iterrows():
                    hood = row[nhood_col]
                    for col in ["net_change", "vitality", "net_openings", "business_change"]:
                        if col in df.columns and hood in hoods:
                            vitality[hood] = row[col]
                            break
        except:
            pass
    return vitality


def compute_emergency_density(df_fire, hoods):
    """Component 11: Fire/EMS medical call density per km²."""
    nhood_col = get_neighborhood_field(df_fire)
    if nhood_col is None or df_fire.empty:
        return pd.Series(0.0, index=hoods)

    # Filter to medical calls if possible
    type_col = None
    for c in ["call_type", "call_type_group", "type"]:
        if c in df_fire.columns:
            type_col = c
            break

    if type_col:
        medical = df_fire[df_fire[type_col].str.lower().str.contains("medical|overdose|drug|unconscious", na=False)]
        if len(medical) == 0:
            medical = df_fire  # Use all if filter is too strict
    else:
        medical = df_fire

    counts = medical.groupby(nhood_col).size()
    density = pd.Series(0.0, index=hoods)
    for hood in hoods:
        area = NEIGHBORHOOD_AREAS_KM2.get(hood, 2.0)
        density[hood] = counts.get(hood, 0) / area
    return density


def compute_environmental_quality(hoods):
    """Component 12: Tree density + food/drink establishment density (positive signal)."""
    eq = pd.Series(0.5, index=hoods)

    # Trees
    tree_path = RAW_DIR / "layer2" / "street_tree_list.parquet"
    if tree_path.exists():
        try:
            df = pd.read_parquet(tree_path)
            nhood_col = get_neighborhood_field(df)
            if nhood_col:
                counts = df.groupby(nhood_col).size()
                for hood in hoods:
                    area = NEIGHBORHOOD_AREAS_KM2.get(hood, 2.0)
                    eq[hood] = counts.get(hood, 0) / area / 100  # normalize
        except:
            pass

    # Food/drink density as positive amenity signal
    food_path = RAW_DIR / "layer2" / "food_drink_density.parquet"
    if food_path.exists():
        try:
            df = pd.read_parquet(food_path)
            nhood_col = get_neighborhood_field(df)
            if nhood_col:
                counts = df.groupby(nhood_col).size()
                for hood in hoods:
                    area = NEIGHBORHOOD_AREAS_KM2.get(hood, 2.0)
                    eq[hood] += counts.get(hood, 0) / area / 50
        except:
            pass
    return eq


def compute_digital_concern(hoods):
    """Component 13: Google Trends safety search volume."""
    gt_path = RAW_DIR / "layer3" / "google_trends_safety.parquet"
    concern = pd.Series(0.0, index=hoods)
    if gt_path.exists():
        try:
            df = pd.read_parquet(gt_path)
            nhood_col = get_neighborhood_field(df)
            if nhood_col is None:
                for c in df.columns:
                    if "neigh" in c.lower() or "hood" in c.lower() or "name" in c.lower():
                        nhood_col = c
                        break
            if nhood_col:
                for _, row in df.iterrows():
                    hood = row[nhood_col]
                    for col in ["concern_score", "search_volume", "trend_score", "interest"]:
                        if col in df.columns and hood in hoods:
                            concern[hood] = row[col]
                            break
        except:
            pass
    return concern


def compute_noise_disorder(hoods):
    """Component 14: Noise complaint density per km²."""
    noise_path = RAW_DIR / "layer2" / "noise_complaints.parquet"
    density = pd.Series(0.0, index=hoods)
    if noise_path.exists():
        try:
            df = pd.read_parquet(noise_path)
            nhood_col = get_neighborhood_field(df)
            if nhood_col:
                counts = df.groupby(nhood_col).size()
                for hood in hoods:
                    area = NEIGHBORHOOD_AREAS_KM2.get(hood, 2.0)
                    density[hood] = counts.get(hood, 0) / area
        except:
            pass
    return density


def compute_waste_odor(hoods):
    """Component 15: Waste/odor report density per km²."""
    waste_path = RAW_DIR / "layer2" / "waste_odor_reports.parquet"
    density = pd.Series(0.0, index=hoods)
    if waste_path.exists():
        try:
            df = pd.read_parquet(waste_path)
            nhood_col = get_neighborhood_field(df)
            if nhood_col:
                counts = df.groupby(nhood_col).size()
                for hood in hoods:
                    area = NEIGHBORHOOD_AREAS_KM2.get(hood, 2.0)
                    density[hood] = counts.get(hood, 0) / area
        except:
            pass
    return density


# ─── MAIN SPI COMPUTATION ───────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PUBLIC SAFETY PULSE — SPI v4 Computation Engine")
    print("=" * 60)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Load GeoJSON
    print("\n📍 Loading GeoJSON...")
    if not GEOJSON_PATH.exists():
        print(f"  ❌ GeoJSON not found at {GEOJSON_PATH}")
        print("  Run: curl -o data/sf_neighborhoods.geojson 'https://raw.githubusercontent.com/sfchronicle/sf-shapefiles/main/SF%20neighborhoods/sf-neighborhoods-analysis.json'")
        return
    with open(GEOJSON_PATH, "r") as f:
        geojson = json.load(f)
    hoods = [feat["properties"]["nhood"] for feat in geojson["features"]]
    print(f"  ✅ {len(hoods)} neighborhoods loaded")

    # 2. Load datasets
    print("\n📊 Loading datasets...")
    df_311 = load_parquet_safe(RAW_DIR / "layer2" / "311_cases_safety.parquet", "311 Safety Cases")
    df_sfpd = load_parquet_safe(RAW_DIR / "layer1" / "sfpd_incidents.parquet", "SFPD Incidents")
    df_crashes = load_parquet_safe(RAW_DIR / "layer1" / "traffic_crashes.parquet", "Traffic Crashes")
    df_fire = load_parquet_safe(RAW_DIR / "layer1" / "fire_calls.parquet", "Fire/EMS Calls")

    # 3. Compute all-day components
    print("\n🔧 Computing SPI v4 components (all-day)...")

    components = {}

    print("  [1/15] Disorder Density...")
    components["disorder_density"] = compute_disorder_density(df_311, hoods)

    print("  [2/15] Crime Severity...")
    components["crime_severity"] = compute_crime_severity(df_sfpd, hoods)

    print("  [3/15] Disorder Salience...")
    components["disorder_salience"] = compute_disorder_salience(df_311, hoods)

    print("  [4/15] Pedestrian Safety...")
    components["pedestrian_safety"] = compute_pedestrian_safety(df_crashes, hoods)

    print("  [5/15] Temporal Risk...")
    components["temporal_risk"] = compute_temporal_risk(df_311, df_sfpd, hoods)

    print("  [6/15] Resolution Responsiveness...")
    components["resolution_responsive"] = compute_resolution_responsiveness(hoods)

    print("  [7/15] Review Sentiment...")
    components["review_sentiment"] = compute_review_sentiment(hoods)

    print("  [8/15] Transit Confidence...")
    components["transit_confidence"] = compute_transit_confidence(hoods)

    print("  [9/15] Lighting Risk...")
    components["lighting_risk"] = compute_lighting_risk(hoods)

    print("  [10/15] Business Vitality...")
    components["business_vitality"] = compute_business_vitality(hoods)

    print("  [11/15] Emergency Density...")
    components["emergency_density"] = compute_emergency_density(df_fire, hoods)

    print("  [12/15] Environmental Quality...")
    components["environmental_quality"] = compute_environmental_quality(hoods)

    print("  [13/15] Digital Concern...")
    components["digital_concern"] = compute_digital_concern(hoods)

    print("  [14/15] Noise Disorder...")
    components["noise_disorder"] = compute_noise_disorder(hoods)

    print("  [15/15] Waste/Odor...")
    components["waste_odor"] = compute_waste_odor(hoods)

    # 4. Compute all-day SPI
    print("\n📈 Computing all-day SPI scores...")

    # Z-score normalize all components
    z_components = {}
    for name, values in components.items():
        z_components[name] = z_score_cap(values)

    # Invert positive signals (higher = better perception)
    POSITIVE_SIGNALS = {"review_sentiment", "transit_confidence", "business_vitality", "environmental_quality"}
    for name in POSITIVE_SIGNALS:
        z_components[name] = -z_components[name]  # flip so higher raw = lower z = better SPI

    # Weighted sum
    weighted_sum = pd.Series(0.0, index=hoods)
    for name, z in z_components.items():
        weighted_sum += WEIGHTS[name] * z

    # SPI = 100 - scaled(weighted_sum) → higher is better
    # Scale to 0-100 range
    ws_min = weighted_sum.min()
    ws_max = weighted_sum.max()
    if ws_max > ws_min:
        spi_all_day = 100 - ((weighted_sum - ws_min) / (ws_max - ws_min)) * 100
    else:
        spi_all_day = pd.Series(50.0, index=hoods)

    spi_all_day = spi_all_day.clip(0, 100).round(1)

    # 5. Compute time-window SPI
    print("\n⏰ Computing time-window SPI scores...")

    # Get time-windowed components for 311 and SFPD
    disorder_by_tw = compute_disorder_density_by_window(df_311, hoods)
    crime_by_tw = compute_crime_severity_by_window(df_sfpd, hoods)

    spi_by_window = {}
    for tw in TIME_WINDOWS:
        # Use time-specific for disorder and crime, all-day for others
        tw_components = dict(components)  # copy
        tw_components["disorder_density"] = disorder_by_tw[tw]
        tw_components["crime_severity"] = crime_by_tw[tw]

        # Apply night penalty
        if tw in ("night", "late_night"):
            tw_components["temporal_risk"] = components["temporal_risk"] * 1.5
            tw_components["lighting_risk"] = components["lighting_risk"] * 2.0

        # Z-score
        tw_z = {}
        for name, values in tw_components.items():
            tw_z[name] = z_score_cap(values)
        for name in POSITIVE_SIGNALS:
            tw_z[name] = -tw_z[name]

        # Weighted sum
        tw_ws = pd.Series(0.0, index=hoods)
        for name, z in tw_z.items():
            tw_ws += WEIGHTS[name] * z

        # Scale
        tw_min = tw_ws.min()
        tw_max = tw_ws.max()
        if tw_max > tw_min:
            tw_spi = 100 - ((tw_ws - tw_min) / (tw_max - tw_min)) * 100
        else:
            tw_spi = pd.Series(50.0, index=hoods)

        spi_by_window[tw] = tw_spi.clip(0, 100).round(1)

    # 6. Find top contributing factors per neighborhood
    print("\n🔍 Identifying top factors per neighborhood...")

    top_factors = {}
    factor_labels = {
        "disorder_density": "311 Disorder Reports",
        "crime_severity": "Crime Severity",
        "disorder_salience": "Disorder Visibility",
        "pedestrian_safety": "Traffic Crashes",
        "temporal_risk": "Nighttime Risk",
        "resolution_responsive": "Slow City Response",
        "review_sentiment": "Yelp Ratings",
        "transit_confidence": "Transit Ridership",
        "lighting_risk": "Streetlight Outages",
        "business_vitality": "Business Closures",
        "emergency_density": "Emergency Calls",
        "environmental_quality": "Environmental Quality",
        "digital_concern": "Online Safety Concern",
        "noise_disorder": "Noise Complaints",
        "waste_odor": "Waste/Odor Reports",
    }

    for hood in hoods:
        contributions = {}
        for name, z in z_components.items():
            contributions[name] = abs(WEIGHTS[name] * z[hood])
        sorted_c = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        top_factors[hood] = [
            {"factor": factor_labels.get(name, name), "impact": round(val, 3)}
            for name, val in sorted_c[:5]
        ]

    # 7. Compute category breakdowns per neighborhood (for detail panels)
    print("\n📋 Computing category breakdowns...")
    category_breakdown = {}
    nhood_311 = get_neighborhood_field(df_311)
    if nhood_311 and not df_311.empty:
        svc_col = None
        for c in ["service_name", "category", "request_type"]:
            if c in df_311.columns:
                svc_col = c
                break
        if svc_col:
            for hood in hoods:
                hood_data = df_311[df_311[nhood_311] == hood]
                if len(hood_data) > 0:
                    cats = hood_data[svc_col].value_counts().head(5)
                    category_breakdown[hood] = {k: int(v) for k, v in cats.items()}
                else:
                    category_breakdown[hood] = {}
        else:
            for hood in hoods:
                category_breakdown[hood] = {}
    else:
        for hood in hoods:
            category_breakdown[hood] = {}

    # 8. Inject into GeoJSON
    print("\n🗺️  Injecting SPI data into GeoJSON...")

    total_records = len(df_311) + len(df_sfpd) + len(df_crashes) + len(df_fire)

    for feat in geojson["features"]:
        hood = feat["properties"]["nhood"]
        props = feat["properties"]

        # All-day SPI
        props["spi"] = float(spi_all_day.get(hood, 50.0))

        # Time-window SPI
        for tw, scores in spi_by_window.items():
            props[f"spi_{tw}"] = float(scores.get(hood, 50.0))

        # Raw component z-scores (for tooltips)
        for name, z in z_components.items():
            props[f"z_{name}"] = round(float(z.get(hood, 0.0)), 3)

        # Top factors
        props["top_factors"] = top_factors.get(hood, [])

        # Category breakdown
        props["categories"] = category_breakdown.get(hood, {})

        # Record counts
        if nhood_311 and not df_311.empty:
            props["count_311"] = int(df_311[df_311[nhood_311] == hood].shape[0])
        else:
            props["count_311"] = 0
        nhood_sfpd = get_neighborhood_field(df_sfpd)
        if nhood_sfpd and not df_sfpd.empty:
            props["count_crime"] = int(df_sfpd[df_sfpd[nhood_sfpd] == hood].shape[0])
        else:
            props["count_crime"] = 0

        # Area
        props["area_km2"] = NEIGHBORHOOD_AREAS_KM2.get(hood, 2.0)

    # 9. Write output
    print(f"\n💾 Writing output to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(geojson, f)
    print(f"  ✅ GeoJSON with SPI data: {OUTPUT_PATH.stat().st_size / 1024:.0f} KB")

    # 10. Write summary
    summary = {
        "generated": datetime.now().isoformat(),
        "version": "SPI v4",
        "num_neighborhoods": len(hoods),
        "num_components": 15,
        "total_records": total_records,
        "dataset_counts": {
            "311_safety": len(df_311),
            "sfpd_incidents": len(df_sfpd),
            "traffic_crashes": len(df_crashes),
            "fire_calls": len(df_fire),
        },
        "spi_stats": {
            "mean": round(float(spi_all_day.mean()), 1),
            "median": round(float(spi_all_day.median()), 1),
            "min": round(float(spi_all_day.min()), 1),
            "max": round(float(spi_all_day.max()), 1),
            "std": round(float(spi_all_day.std()), 1),
        },
        "bottom_5": spi_all_day.nsmallest(5).to_dict(),
        "top_5": spi_all_day.nlargest(5).to_dict(),
        "weights": WEIGHTS,
        "time_windows": list(TIME_WINDOWS.keys()),
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✅ Summary: {SUMMARY_PATH}")

    # 11. Print results
    print("\n" + "=" * 60)
    print("SPI v4 RESULTS")
    print("=" * 60)
    print(f"\nTotal records analyzed: {total_records:,}")
    print(f"SPI range: {spi_all_day.min():.1f} – {spi_all_day.max():.1f}")
    print(f"SPI mean:  {spi_all_day.mean():.1f} (std: {spi_all_day.std():.1f})")

    print("\n🔴 Bottom 5 (lowest perceived safety):")
    for hood, score in spi_all_day.nsmallest(5).items():
        print(f"   {hood:40s}  {score:5.1f}")

    print("\n🟢 Top 5 (highest perceived safety):")
    for hood, score in spi_all_day.nlargest(5).items():
        print(f"   {hood:40s}  {score:5.1f}")

    print(f"\n✅ Done. Open psp_dashboard_v8.html to view the choropleth.")
    print(f"   Make sure data/spi_v4_output.json is in the same directory.")


if __name__ == "__main__":
    main()
