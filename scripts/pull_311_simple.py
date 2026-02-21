#!/usr/bin/env python3
import requests, pandas as pd, time
from pathlib import Path
from datetime import datetime, timedelta

Path("data/raw/layer2").mkdir(parents=True, exist_ok=True)
url = "https://data.sfgov.org/resource/vw6y-z8j6.json"
cutoff = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%dT00:00:00")

print("PULLING: 311 Cases (smaller batches + retries)")
all_records = []
offset = 0
BATCH = 10000

while True:
    print(f"  Fetching from offset {offset}...")
    for attempt in range(3):
        try:
            resp = requests.get(url, params={
                "$where": f"requested_datetime > '{cutoff}'",
                "$limit": BATCH,
                "$offset": offset,
                "$order": ":id",
            }, timeout=180)
            resp.raise_for_status()
            records = resp.json()
            break
        except Exception as e:
            print(f"    Retry {attempt+1}/3: {e}")
            time.sleep(5)
            records = []
    if not records:
        break
    all_records.extend(records)
    print(f"  Total so far: {len(all_records):,}")
    if len(records) < BATCH:
        break
    offset += BATCH
    time.sleep(1)

df = pd.DataFrame(all_records)
print(f"\nTotal records: {len(df):,}")
df.to_parquet("data/raw/layer2/311_cases_all.parquet", index=False)

safety_cats = ["Encampments","Homeless Concerns","Street and Sidewalk Cleaning",
    "Graffiti","Streetlights","Abandoned Vehicle","Damaged Property",
    "SFPD Requests","General Requests","Noise Report","Sewer Issues"]
safety = df[df["service_name"].isin(safety_cats)]
safety.to_parquet("data/raw/layer2/311_cases_safety.parquet", index=False)
print(f"Safety-relevant: {len(safety):,}")
print(safety["service_name"].value_counts().head(10))
print("\nDone!")
