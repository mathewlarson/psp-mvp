#!/usr/bin/env python3
"""
Public Safety Pulse — Safety Perception Index Computation
==========================================================
Computes a 15-component Safety Perception Index per SF neighborhood
per 4-hour time window. Outputs enriched GeoJSON for dashboard rendering.

Usage:
    cd ~/Desktop/psp-mvp
    source venv/bin/activate
    python compute_spi.py

Output:
    data/spi_v4_output.json  — GeoJSON with SPI scores (name kept for compatibility)
    data/spi_summary.json    — Summary stats
"""

import json
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
warnings.filterwarnings("ignore")

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed. Run: pip install pandas")
    exit(1)

# ─── CONFIGURATION ──────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

# Try multiple GeoJSON locations
GEOJSON_PATHS = [
    DATA_DIR / "sf_neighborhoods.geojson",
    BASE_DIR / "sf_neighborhoods.geojson",
    DATA_DIR / "raw" / "sf_neighborhoods.geojson",
]

OUTPUT_PATH = DATA_DIR / "spi_v4_output.json"  # kept for dashboard compatibility
SUMMARY_PATH = DATA_DIR / "spi_summary.json"

TIME_WINDOWS = {
    "early_morning": (4, 8),
    "morning":       (8, 12),
    "afternoon":     (12, 16),
    "evening":       (16, 20),
    "night":         (20, 24),
    "late_night":    (0, 4),
}

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

CRIME_SEVERITY = {
    "homicide":10,"rape":9,"robbery":8,"assault":7,"aggravated assault":7,
    "arson":6,"burglary":5,"motor vehicle theft":4,"larceny theft":3,
    "drug offense":3,"drug/narcotic":3,"vandalism":2,"warrant":1.5,
    "weapons offense":6,"weapon laws":6,"sex offense":7,"kidnapping":9,
    "malicious mischief":2,"suspicious occ":1,"disorderly conduct":1.5,
    "trespass":1,"fraud":2,"stolen property":3,"missing person":3,
    "other offenses":1,"non-criminal":0.5,
}

DISORDER_SALIENCE = {
    "encampment":1.0,"homeless concerns":0.9,"human/animal waste":0.9,
    "street and sidewalk cleaning":0.7,"needles":0.95,"graffiti public":0.5,
    "graffiti private":0.4,"noise":0.6,"blocked street and sidewalk":0.3,
    "damaged property":0.5,"damage property":0.5,"streetlights":0.7,
    "illegal postings":0.2,"general request":0.3,"sewer":0.4,
}

NEIGHBORHOOD_AREAS_KM2 = {
    "Bayview Hunters Point":6.8,"Bernal Heights":2.4,"Castro/Upper Market":1.5,
    "Chinatown":0.5,"Excelsior":3.2,"Financial District/South Beach":2.1,
    "Glen Park":2.0,"Golden Gate Park":4.1,"Haight Ashbury":1.2,"Hayes Valley":0.8,
    "Inner Richmond":2.3,"Inner Sunset":2.5,"Japantown":0.4,"Lakeshore":5.5,
    "Lincoln Park":1.8,"Lone Mountain/USF":1.0,"Marina":1.5,"McLaren Park":2.8,
    "Mission":2.5,"Mission Bay":1.2,"Nob Hill":0.8,"Noe Valley":1.8,
    "North Beach":0.9,"Oceanview/Merced/Ingleside":3.5,"Outer Mission":2.2,
    "Outer Richmond":3.8,"Outer Sunset":5.5,"Pacific Heights":1.5,"Portola":2.0,
    "Potrero Hill":2.2,"Presidio":4.8,"Presidio Heights":0.8,"Russian Hill":0.7,
    "Seacliff":0.6,"South of Market":3.0,"Sunset/Parkside":5.0,"Tenderloin":0.6,
    "Treasure Island":2.2,"Twin Peaks":1.5,"Visitacion Valley":2.5,
    "West of Twin Peaks":3.5,
}

POSITIVE_SIGNALS = {"review_sentiment","transit_confidence","business_vitality","environmental_quality"}

# ─── HELPERS ──────────────────────────────────────────────────────────────

def load_safe(path, name=""):
    p = Path(path)
    if p.exists():
        try:
            df = pd.read_parquet(p)
            print(f"  ✅ {name}: {len(df):,} records")
            return df
        except Exception as e:
            print(f"  ❌ {name}: {e}")
    else:
        print(f"  ⚠️  {name}: not found")
    return pd.DataFrame()

def z_cap(s, cap=3.0):
    m, sd = s.mean(), s.std()
    if sd == 0 or pd.isna(sd): return pd.Series(0.0, index=s.index)
    return ((s - m) / sd).clip(-cap, cap)

def get_hood_col(df):
    for c in ["analysis_neighborhood","neighborhoods_analysis_boundaries","neighborhood","nhood"]:
        if c in df.columns: return c
    return None

def get_hour(df):
    for c in ["incident_datetime","requested_datetime","received_dttm","call_date"]:
        if c in df.columns:
            try:
                df["_hour"] = pd.to_datetime(df[c], errors="coerce").dt.hour
                return df
            except: continue
    df["_hour"] = np.nan
    return df

def to_tw(h):
    if pd.isna(h): return "all_day"
    h = int(h)
    if 4<=h<8: return "early_morning"
    if 8<=h<12: return "morning"
    if 12<=h<16: return "afternoon"
    if 16<=h<20: return "evening"
    if 20<=h<24: return "night"
    return "late_night"

def density(df, hoods, hood_col=None):
    if hood_col is None: hood_col = get_hood_col(df)
    if hood_col is None or df.empty: return pd.Series(0.0, index=hoods)
    counts = df.groupby(hood_col).size()
    return pd.Series({h: counts.get(h,0)/NEIGHBORHOOD_AREAS_KM2.get(h,2.0) for h in hoods})

def density_by_tw(df, hoods):
    hood_col = get_hood_col(df)
    result = {}
    if hood_col is None or df.empty:
        for tw in TIME_WINDOWS: result[tw] = pd.Series(0.0, index=hoods)
        return result
    df = get_hour(df)
    df["_tw"] = df["_hour"].apply(to_tw)
    for tw in TIME_WINDOWS:
        sub = df[df["_tw"]==tw]
        result[tw] = density(sub, hoods, hood_col)
    return result

# ─── MAIN ──────────────────────────────────────────────────────────────

def main():
    print("=" * 56)
    print("PUBLIC SAFETY PULSE — Safety Perception Index")
    print("=" * 56)

    # Find GeoJSON
    geojson_path = None
    for p in GEOJSON_PATHS:
        if p.exists():
            geojson_path = p
            break
    if geojson_path is None:
        print("❌ GeoJSON not found. Checked:")
        for p in GEOJSON_PATHS: print(f"   {p}")
        print("Download: curl -o data/sf_neighborhoods.geojson 'https://raw.githubusercontent.com/sfchronicle/sf-shapefiles/main/SF%20neighborhoods/sf-neighborhoods-analysis.json'")
        return

    with open(geojson_path) as f:
        geojson = json.load(f)
    hoods = [feat["properties"]["nhood"] for feat in geojson["features"]]
    print(f"\n📍 {len(hoods)} neighborhoods from {geojson_path.name}")

    # Load datasets
    print("\n📊 Loading datasets...")
    df_311 = load_safe(RAW_DIR/"layer2"/"311_cases_safety.parquet", "311 Safety")
    df_sfpd = load_safe(RAW_DIR/"layer1"/"sfpd_incidents.parquet", "SFPD")
    df_crash = load_safe(RAW_DIR/"layer1"/"traffic_crashes.parquet", "Crashes")
    df_fire = load_safe(RAW_DIR/"layer1"/"fire_calls.parquet", "Fire/EMS")

    # Compute all-day components
    print("\n🔧 Computing 15 components...")
    C = {}

    # 1. Disorder density
    C["disorder_density"] = density(df_311, hoods)

    # 2. Crime severity
    hc = get_hood_col(df_sfpd)
    if hc and not df_sfpd.empty:
        cc = next((c for c in ["incident_category","category"] if c in df_sfpd.columns), None)
        if cc:
            df_sfpd["_sev"] = df_sfpd[cc].str.lower().map(
                lambda x: max((v for k,v in CRIME_SEVERITY.items() if k in str(x)), default=1.0))
        else:
            df_sfpd["_sev"] = 1.0
        ws = df_sfpd.groupby(hc)["_sev"].sum()
        C["crime_severity"] = pd.Series({h: ws.get(h,0)/NEIGHBORHOOD_AREAS_KM2.get(h,2.0) for h in hoods})
    else:
        C["crime_severity"] = pd.Series(0.0, index=hoods)

    # 3. Disorder salience
    hc311 = get_hood_col(df_311)
    sc = next((c for c in ["service_name","category","request_type"] if c in df_311.columns), None) if not df_311.empty else None
    if sc and hc311:
        df_311["_sal"] = df_311[sc].str.lower().map(
            lambda x: max((v for k,v in DISORDER_SALIENCE.items() if k in str(x)), default=0.3))
        ms = df_311.groupby(hc311)["_sal"].mean()
        C["disorder_salience"] = pd.Series({h: ms.get(h,0.5) for h in hoods})
    else:
        C["disorder_salience"] = pd.Series(0.5, index=hoods)

    # 4. Pedestrian safety
    C["pedestrian_safety"] = density(df_crash, hoods)

    # 5. Temporal risk
    tr = pd.Series(0.5, index=hoods)
    if hc311 and not df_311.empty:
        df_311 = get_hour(df_311)
        for h in hoods:
            hd = df_311[df_311[hc311]==h]
            if len(hd)>0:
                night = hd[hd["_hour"].between(20,23) | (hd["_hour"]<4)]
                tr[h] = 0.6*(len(night)/len(hd)) + 0.4*0.5
    C["temporal_risk"] = tr

    # 6-15: Load from auxiliary files with fallbacks
    def load_hood_metric(path, default=0.0):
        s = pd.Series(default, index=hoods)
        if Path(path).exists():
            try:
                df = pd.read_parquet(path)
                hc = get_hood_col(df)
                if hc is None:
                    for c in df.columns:
                        if "neigh" in c.lower() or "hood" in c.lower() or "name" in c.lower():
                            hc = c; break
                if hc:
                    for _, row in df.iterrows():
                        h = row[hc]
                        if h in hoods:
                            for vc in df.columns:
                                if vc != hc:
                                    try: s[h] = float(row[vc]); break
                                    except: continue
            except: pass
        return s

    C["resolution_responsive"] = load_hood_metric(RAW_DIR/"layer2"/"resolution_times.parquet", 7.0)
    C["review_sentiment"] = load_hood_metric(RAW_DIR/"layer3"/"yelp_neighborhood_quality.parquet", 3.5)
    C["transit_confidence"] = pd.Series(0.0, index=hoods)  # needs BART parsing
    C["lighting_risk"] = density(load_safe(RAW_DIR/"layer2"/"streetlight_outages.parquet","Streetlights"), hoods) if (RAW_DIR/"layer2"/"streetlight_outages.parquet").exists() else pd.Series(0.0,index=hoods)
    C["business_vitality"] = load_hood_metric(RAW_DIR/"layer3"/"business_vitality.parquet", 0.0)

    # Emergency density (fire/EMS)
    if not df_fire.empty:
        tc = next((c for c in ["call_type","call_type_group"] if c in df_fire.columns), None)
        if tc:
            med = df_fire[df_fire[tc].str.lower().str.contains("medical|overdose|drug|unconscious",na=False)]
            C["emergency_density"] = density(med if len(med)>0 else df_fire, hoods)
        else:
            C["emergency_density"] = density(df_fire, hoods)
    else:
        C["emergency_density"] = pd.Series(0.0, index=hoods)

    C["environmental_quality"] = pd.Series(0.5, index=hoods)
    if (RAW_DIR/"layer2"/"street_tree_list.parquet").exists():
        trees = load_safe(RAW_DIR/"layer2"/"street_tree_list.parquet","Trees")
        thc = get_hood_col(trees)
        if thc:
            tc = trees.groupby(thc).size()
            for h in hoods:
                C["environmental_quality"][h] = tc.get(h,0)/NEIGHBORHOOD_AREAS_KM2.get(h,2.0)/100

    C["digital_concern"] = load_hood_metric(RAW_DIR/"layer3"/"google_trends_safety.parquet", 0.0)

    for key, path in [("noise_disorder","noise_complaints.parquet"),("waste_odor","waste_odor_reports.parquet")]:
        fp = RAW_DIR/"layer2"/path
        if fp.exists():
            df_sub = load_safe(fp, key)
            C[key] = density(df_sub, hoods)
        else:
            C[key] = pd.Series(0.0, index=hoods)

    # Z-score and compute SPI
    print("\n📈 Computing Safety Perception Index...")
    Z = {}
    for name, vals in C.items():
        Z[name] = z_cap(vals)
    for name in POSITIVE_SIGNALS:
        Z[name] = -Z[name]

    ws = pd.Series(0.0, index=hoods)
    for name, z in Z.items():
        ws += WEIGHTS[name] * z

    wmin, wmax = ws.min(), ws.max()
    if wmax > wmin:
        spi_all = 100 - ((ws - wmin) / (wmax - wmin)) * 100
    else:
        spi_all = pd.Series(50.0, index=hoods)
    spi_all = spi_all.clip(0, 100).round(1)

    # Time-window SPI
    print("⏰ Computing time-window scores...")
    dd_tw = density_by_tw(df_311, hoods)

    # Simplified crime by time window
    cs_tw = {}
    hc_sfpd = get_hood_col(df_sfpd)
    if hc_sfpd and not df_sfpd.empty:
        df_sfpd = get_hour(df_sfpd)
        df_sfpd["_tw"] = df_sfpd["_hour"].apply(to_tw)
        for tw in TIME_WINDOWS:
            sub = df_sfpd[df_sfpd["_tw"]==tw]
            if "_sev" in sub.columns:
                ww = sub.groupby(hc_sfpd)["_sev"].sum()
                cs_tw[tw] = pd.Series({h: ww.get(h,0)/NEIGHBORHOOD_AREAS_KM2.get(h,2.0) for h in hoods})
            else:
                cs_tw[tw] = density(sub, hoods, hc_sfpd)
    else:
        for tw in TIME_WINDOWS:
            cs_tw[tw] = pd.Series(0.0, index=hoods)

    spi_tw = {}
    for tw in TIME_WINDOWS:
        twC = dict(C)
        twC["disorder_density"] = dd_tw[tw]
        twC["crime_severity"] = cs_tw[tw]
        if tw in ("night","late_night"):
            twC["temporal_risk"] = C["temporal_risk"]*1.5
            twC["lighting_risk"] = C["lighting_risk"]*2.0

        twZ = {n: z_cap(v) for n,v in twC.items()}
        for n in POSITIVE_SIGNALS: twZ[n] = -twZ[n]
        tw_ws = sum(WEIGHTS[n]*z for n,z in twZ.items())
        mn, mx = tw_ws.min(), tw_ws.max()
        if mx > mn:
            spi_tw[tw] = (100 - ((tw_ws-mn)/(mx-mn))*100).clip(0,100).round(1)
        else:
            spi_tw[tw] = pd.Series(50.0, index=hoods)

    # Top factors
    factor_labels = {
        "disorder_density":"311 Disorder","crime_severity":"Crime Severity",
        "disorder_salience":"Disorder Visibility","pedestrian_safety":"Traffic Crashes",
        "temporal_risk":"Nighttime Risk","resolution_responsive":"Slow Response",
        "review_sentiment":"Yelp Ratings","transit_confidence":"Transit",
        "lighting_risk":"Streetlights","business_vitality":"Business Health",
        "emergency_density":"Emergency Calls","environmental_quality":"Environment",
        "digital_concern":"Online Concern","noise_disorder":"Noise","waste_odor":"Waste/Odor",
    }
    top_factors = {}
    for h in hoods:
        contrib = {n: abs(WEIGHTS[n]*Z[n][h]) for n in Z}
        top_factors[h] = [{"factor":factor_labels.get(n,n),"impact":round(v,3)}
                          for n,v in sorted(contrib.items(),key=lambda x:x[1],reverse=True)[:5]]

    # Inject into GeoJSON
    print("\n🗺️  Writing enriched GeoJSON...")
    total = len(df_311)+len(df_sfpd)+len(df_crash)+len(df_fire)

    for feat in geojson["features"]:
        h = feat["properties"]["nhood"]
        p = feat["properties"]
        p["spi"] = float(spi_all.get(h,50))
        for tw, scores in spi_tw.items():
            p[f"spi_{tw}"] = float(scores.get(h,50))
        p["top_factors"] = top_factors.get(h,[])
        p["count_311"] = int(df_311[df_311[hc311]==h].shape[0]) if hc311 and not df_311.empty else 0
        hcs = get_hood_col(df_sfpd)
        p["count_crime"] = int(df_sfpd[df_sfpd[hcs]==h].shape[0]) if hcs and not df_sfpd.empty else 0
        p["area_km2"] = NEIGHBORHOOD_AREAS_KM2.get(h,2.0)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH,"w") as f:
        json.dump(geojson, f)
    print(f"  ✅ {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size/1024:.0f} KB)")

    # Summary
    summary = {
        "generated": datetime.now().isoformat(),
        "neighborhoods": len(hoods),
        "total_records": total,
        "spi_mean": round(float(spi_all.mean()),1),
        "spi_std": round(float(spi_all.std()),1),
        "bottom_5": spi_all.nsmallest(5).to_dict(),
        "top_5": spi_all.nlargest(5).to_dict(),
    }
    with open(SUMMARY_PATH,"w") as f:
        json.dump(summary, f, indent=2)

    # Print results
    print("\n" + "=" * 56)
    print("SAFETY PERCEPTION INDEX — RESULTS")
    print("=" * 56)
    print(f"\nTotal records analyzed: {total:,}")
    print(f"SPI range: {spi_all.min():.1f} – {spi_all.max():.1f}")
    print(f"SPI mean:  {spi_all.mean():.1f} (std: {spi_all.std():.1f})")
    print("\n🔴 Bottom 5 (lowest perceived safety):")
    for h,s in spi_all.nsmallest(5).items():
        print(f"   {h:40s}  {s:5.1f}")
    print("\n🟢 Top 5 (highest perceived safety):")
    for h,s in spi_all.nlargest(5).items():
        print(f"   {h:40s}  {s:5.1f}")
    print(f"\n✅ Done. Open psp_dashboard_v9.html to view.")

if __name__ == "__main__":
    main()
