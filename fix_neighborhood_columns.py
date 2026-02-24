#!/usr/bin/env python3
"""
fix_neighborhood_columns.py — Fix datasets where neighborhood is the index
or needs spatial assignment, then recompute SPI with corrected data.
"""
import os, json
import pandas as pd
import numpy as np

def fix_yelp():
    path = 'data/raw/layer3/yelp_neighborhood_quality.parquet'
    df = pd.read_parquet(path)
    print(f"\nYelp: {len(df)} rows, columns={list(df.columns)}")
    print(f"  Index name: {df.index.name}")
    print(f"  Index values: {list(df.index[:5])}")
    
    if 'neighborhood' not in df.columns and 'analysis_neighborhood' not in df.columns:
        # Neighborhood is the index
        df = df.reset_index()
        # Rename the index column
        idx_col = df.columns[0]
        if idx_col not in ['neighborhood', 'analysis_neighborhood']:
            df = df.rename(columns={idx_col: 'neighborhood'})
        df.to_parquet(path, index=False)
        print(f"  FIXED: Added neighborhood column from index")
        print(f"  New columns: {list(df.columns)}")
        print(f"  Sample neighborhoods: {list(df['neighborhood'][:5])}")
    else:
        print(f"  OK: Already has neighborhood column")
    return df

def fix_business():
    path = 'data/raw/layer3/business_vitality.parquet'
    df = pd.read_parquet(path)
    print(f"\nBusiness: {len(df)} rows, columns={list(df.columns)}")
    print(f"  Index name: {df.index.name}")
    print(f"  Index values: {list(df.index[:5])}")
    
    if 'neighborhood' not in df.columns and 'analysis_neighborhood' not in df.columns:
        df = df.reset_index()
        idx_col = df.columns[0]
        if idx_col not in ['neighborhood', 'analysis_neighborhood']:
            df = df.rename(columns={idx_col: 'neighborhood'})
        df.to_parquet(path, index=False)
        print(f"  FIXED: Added neighborhood column from index")
        print(f"  New columns: {list(df.columns)}")
        print(f"  Sample neighborhoods: {list(df['neighborhood'][:5])}")
    else:
        print(f"  OK: Already has neighborhood column")
    return df

def fix_trees():
    path = 'data/raw/layer2/street_tree_list.parquet'
    df = pd.read_parquet(path)
    print(f"\nTrees: {len(df)} rows, columns={list(df.columns)}")
    
    if 'neighborhood' in df.columns or 'analysis_neighborhood' in df.columns:
        print(f"  OK: Already has neighborhood column")
        return df
    
    # Check for lat/lon
    has_coords = 'latitude' in df.columns and 'longitude' in df.columns
    if not has_coords:
        has_coords = 'ycoord' in df.columns and 'xcoord' in df.columns
    
    if not has_coords:
        print(f"  No coords found, skipping spatial assignment")
        return df
    
    # Load GeoJSON for spatial assignment
    gj_path = 'data/sf_neighborhoods.geojson'
    if not os.path.exists(gj_path):
        print(f"  No GeoJSON found at {gj_path}, skipping")
        return df
    
    print(f"  Assigning neighborhoods from coordinates...")
    
    with open(gj_path) as f:
        gj = json.load(f)
    
    # Build simple bounding boxes per neighborhood for fast assignment
    nhood_bounds = {}
    for feat in gj['features']:
        props = feat['properties']
        name = props.get('nhood') or props.get('name') or props.get('neighborhood')
        if not name: continue
        coords = feat['geometry']['coordinates']
        try:
            if feat['geometry']['type'] == 'MultiPolygon':
                flat = np.array([c for poly in coords for ring in poly for c in ring])
            else:
                flat = np.array([c for ring in coords for c in ring])
            nhood_bounds[name] = {
                'lon_min': flat[:,0].min(), 'lon_max': flat[:,0].max(),
                'lat_min': flat[:,1].min(), 'lat_max': flat[:,1].max(),
                'lon_mid': flat[:,0].mean(), 'lat_mid': flat[:,1].mean(),
            }
        except:
            continue
    
    # Use latitude/longitude or ycoord/xcoord
    if 'latitude' in df.columns:
        lats = pd.to_numeric(df['latitude'], errors='coerce')
        lons = pd.to_numeric(df['longitude'], errors='coerce')
    else:
        lats = pd.to_numeric(df['ycoord'], errors='coerce')
        lons = pd.to_numeric(df['xcoord'], errors='coerce')
    
    # Assign each tree to nearest neighborhood centroid (fast approximate method)
    neighborhoods = []
    nhood_names = list(nhood_bounds.keys())
    nhood_mids = np.array([[nhood_bounds[n]['lon_mid'], nhood_bounds[n]['lat_mid']] 
                           for n in nhood_names])
    
    # Process in chunks for speed
    chunk_size = 10000
    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        chunk_lats = lats.iloc[start:end].values
        chunk_lons = lons.iloc[start:end].values
        
        for i in range(len(chunk_lats)):
            lat, lon = chunk_lats[i], chunk_lons[i]
            if np.isnan(lat) or np.isnan(lon):
                neighborhoods.append(None)
                continue
            
            # Find nearest centroid
            dists = (nhood_mids[:,0] - lon)**2 + (nhood_mids[:,1] - lat)**2
            neighborhoods.append(nhood_names[np.argmin(dists)])
        
        if start % 50000 == 0 and start > 0:
            print(f"    Assigned {start}/{len(df)} trees...")
    
    df['neighborhood'] = neighborhoods
    assigned = df['neighborhood'].notna().sum()
    print(f"  Assigned {assigned}/{len(df)} trees to neighborhoods")
    
    # Save
    df.to_parquet(path, index=False)
    print(f"  FIXED: Saved with neighborhood column")
    print(f"  Trees per neighborhood (top 5):")
    print(df['neighborhood'].value_counts().head().to_string())
    return df

def check_311():
    """Check if 311 data has neighborhood column."""
    for path in ['data/raw/layer2/311_cases_safety.parquet', 
                 'data/raw/layer2/311_cases_all.parquet']:
        if os.path.exists(path):
            df = pd.read_parquet(path)
            print(f"\n311 ({os.path.basename(path)}): {len(df)} rows")
            nhood_cols = [c for c in df.columns if 'neighborhood' in c.lower() or 'nhood' in c.lower()]
            print(f"  Neighborhood columns: {nhood_cols}")
            if nhood_cols:
                print(f"  Unique neighborhoods: {df[nhood_cols[0]].nunique()}")
                print(f"  Sample: {list(df[nhood_cols[0]].dropna().unique()[:5])}")
            return

if __name__ == '__main__':
    print("=" * 60)
    print("Fixing neighborhood column issues")
    print("=" * 60)
    
    fix_yelp()
    fix_business()
    fix_trees()
    check_311()
    
    print("\n" + "=" * 60)
    print("Done. Now re-run: python compute_spi_enhanced.py --validate")
    print("=" * 60)
