# Public Safety Pulse — MVP Data Pipeline & Dashboard

**City Science Lab San Francisco × MIT Media Lab**

> Build a working interactive dashboard that fuses existing public data to demonstrate 
> the type of insight PSP would provide. This is your roadshow asset.

---

## Quick Start (5 minutes)

```bash
# 1. Clone and set up
cd psp-mvp
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Pull the two critical data sources
python scripts/pull_all_data.py --source 311      # ~5 min, free, no auth
python scripts/pull_all_data.py --source sfpd     # ~5 min, free, no auth

# 3. Process into composite index
python scripts/pull_all_data.py --process

# 4. Launch the dashboard
cd dashboard
streamlit run dashboard_app.py
```

Dashboard will open at `http://localhost:8501`

---

## Project Structure

```
psp-mvp/
├── scripts/
│   ├── pull_all_data.py          # Master data pipeline (all 19 sources)
│   └── sentiment_analysis.py     # NLP pipeline for Yelp/Reddit safety sentiment
├── dashboard/
│   └── dashboard_app.py          # Streamlit interactive dashboard
├── data/
│   ├── raw/
│   │   ├── layer1/               # SFPD incidents, fire calls, crashes
│   │   ├── layer2/               # 311 cases, tent counts, ambassadors
│   │   └── layer3/               # BART, Yelp, Reddit, mobility
│   ├── processed/                # Composite indices, scored sentiment
│   └── dashboard_export/         # JSON/GeoJSON for web dashboard
├── requirements.txt
└── README.md
```

---

## Data Sources: Complete Inventory (19 Sources)

### Layer 1 — Hard Incident Data ("What Happened")

| # | Dataset | Granularity | Freq | Access | Priority | Auto-Pull? |
|---|---------|-------------|------|--------|----------|------------|
| 1 | SFPD Incidents | Individual/geocoded | Daily | Free API | **HIGH** | ✅ Yes |
| 2 | CompStat Reports | District/monthly | Monthly | Free PDF | LOW | ❌ Manual |
| 3 | Crime Dashboard | District/type | Ongoing | Free | MEDIUM | ❌ Manual |
| 4 | Fire Calls | Individual/geocoded | Ongoing | Free API | MEDIUM | ✅ Yes |
| 5 | Traffic Crashes | Individual/geocoded | Quarterly | Free API | MEDIUM | ✅ Yes |
| 6 | SFPD Community Survey | Citywide | Annual | Free PDF | LOW | ❌ Manual |

### Layer 2 — Condition & Environment ("What It Looks/Feels Like")

| # | Dataset | Granularity | Freq | Access | Priority | Auto-Pull? |
|---|---------|-------------|------|--------|----------|------------|
| 7 | **311 Cases** | Individual/geocoded | Nightly | Free API | **CRITICAL** | ✅ Yes |
| 8 | Tent/Structure Counts | District/quarterly | Quarterly | Free | HIGH | ❌ Manual |
| 9 | PIT Homeless Count | District/biennial | Biennial | Free PDF | LOW | ❌ Manual |
| 10 | Ambassador Data | Neighborhood | Quarterly | Free | MEDIUM | ❌ Manual |
| 11 | Cleaning Response | Citywide | Ongoing | Free | LOW | ❌ Manual |
| 12 | Pit Stop Locations | Location | Static | Free | LOW | ❌ Manual |

### Layer 3 — Behavioral & Sentiment Proxies ("How People Respond")

| # | Dataset | Granularity | Freq | Access | Priority | Auto-Pull? |
|---|---------|-------------|------|--------|----------|------------|
| 13 | **City Survey** | Neighborhood | Biennial | Free | **CRITICAL** | ❌ Manual |
| 14 | BART Ridership | Station/daily | Monthly | Free | HIGH | ✅ Yes |
| 15 | Muni Ridership | Route/monthly | Monthly | Free | MEDIUM | ❌ Manual |
| 16 | Yelp/Google Reviews | Business/geocoded | Real-time | Free API* | HIGH | ✅ w/key |
| 17 | Reddit Sentiment | Post-level | Real-time | Free API | MEDIUM | ✅ Yes |
| 18 | Replica/SafeGraph | Block-level | Varies | **Paid** | HIGH | ❌ License |
| 19 | SFCTA Travel Study | Downtown | One-time | Free PDF | LOW | ❌ Manual |

*Yelp requires free API key registration

---

## Pipeline Commands

### Pull data by layer
```bash
python scripts/pull_all_data.py --layer 1    # Hard incident data
python scripts/pull_all_data.py --layer 2    # Condition/environment
python scripts/pull_all_data.py --layer 3    # Behavioral/sentiment
```

### Pull individual sources
```bash
python scripts/pull_all_data.py --source sfpd
python scripts/pull_all_data.py --source 311
python scripts/pull_all_data.py --source fire
python scripts/pull_all_data.py --source traffic
python scripts/pull_all_data.py --source bart
python scripts/pull_all_data.py --source reddit
python scripts/pull_all_data.py --source yelp --yelp-key YOUR_API_KEY
```

### Pull everything
```bash
python scripts/pull_all_data.py --all --months 12
```

### Run NLP sentiment analysis
```bash
python scripts/sentiment_analysis.py --all
```

### Show manual download instructions
```bash
python scripts/pull_all_data.py --manual
```

---

## 6-Week Build Sprint

### Week 1: Data Foundation
**Data Science Jr:**
- [ ] Run `pull_all_data.py --source 311` — pull all safety-relevant 311 cases (12 months)
- [ ] Run `pull_all_data.py --source sfpd` — pull SFPD incidents (12 months)
- [ ] Register for Yelp Fusion API at yelp.com/developers
- [ ] Run `pull_all_data.py --source yelp --yelp-key YOUR_KEY`
- [ ] Verify data quality: check for gaps, geocoding issues, date ranges

**SF Partnership Sr:**
- [ ] Identify 3–5 target CBD areas for MVP focus
- [ ] Begin conversations with CBD directors

### Week 2: Index Construction
**Data Science Sr:**
- [ ] Run `pull_all_data.py --process` — builds disorder density + crime overlay
- [ ] Download City Survey data manually from sf.gov
- [ ] Calibrate composite index weights against City Survey ground truth
- [ ] Validate: does the composite index match known "unsafe" and "safe" areas?

**Data Science Jr:**
- [ ] Run `sentiment_analysis.py --all` — NLP scoring on Yelp + Reddit
- [ ] Download BART ridership data manually
- [ ] Build BART station exit volume overlay

**SF Partnership Sr:**
- [ ] Identify 3–5 target CBD areas for MVP focus

### Week 3: Dashboard v0
**Data Science Jr:**
- [ ] Launch `streamlit run dashboard/dashboard_app.py`
- [ ] Customize map views for target CBD areas
- [ ] Add time-of-day toggle (4-hour windows)
- [ ] Build the "divergence view" (311 vs crime hotspots)

**SF Partnership Jr:**
- [ ] Begin CBD outreach using MVP wireframes/screenshots
- [ ] Schedule merchant feedback sessions

### Week 4: Refinement
**Data Science Team:**
- [ ] Add BART/Muni ridership overlay to dashboard
- [ ] Add Yelp sentiment layer to map
- [ ] Build the "killer roadshow slide" — side-by-side comparison
- [ ] Stress test: do the patterns hold when you drill into blocks?

**SF Partnership Sr:**
- [ ] Draft roadshow deck incorporating MVP screenshots
- [ ] Refine based on CBD feedback

### Week 5: MVP v1 Complete
**Full Team:**
- [ ] MVP v1 locked — all layers integrated
- [ ] Deploy to Vercel/GitHub Pages for live demo access
- [ ] Begin funder presentations
- [ ] Prepare 3 demo scenarios: "walk me through Union Square at 8pm"

### Week 6: Iterate & Formalize
**Full Team:**
- [ ] Iterate based on funder feedback
- [ ] Formalize Phase 1 partnerships
- [ ] Document methodology for MIT collaboration

---

## Technical Architecture

### Composite Safety Perception Score (Illustrative)

```
Block Score = w1 × (inverse 311 disorder density)
            + w2 × (inverse crime density)
            + w3 × (foot traffic volume ÷ baseline)
            + w4 × (Yelp safety sentiment score)
            + w5 × (City Survey neighborhood score)
```

Weights w1–w5 calibrated against City Survey ground truth using regression.

### Spatial & Temporal Units
- **Spatial:** H3 hex grid, resolution 9 (≈ 0.1 km², roughly one city block)
- **Temporal:** 4-hour windows (late night / morning / midday / afternoon / evening / night)
- **Normalization:** Divide counts by foot traffic to avoid high-traffic area bias

### Tech Stack
- **Ingestion:** Python, pandas, requests, sodapy (Socrata API client)
- **NLP:** TextBlob (quick), spaCy (better), HuggingFace transformers (best)
- **Geospatial:** GeoPandas, H3, Shapely
- **Visualization:** Streamlit + Plotly + PyDeck (prototype); Mapbox GL JS / Deck.gl (production)
- **Hosting:** GitHub Pages or Vercel (free)

### API Access Notes
- **DataSF Socrata:** No auth needed for read-only. Generous rate limits. Use `$where` for filtering.
- **BART:** Excel downloads from bart.gov. No API.
- **Yelp Fusion:** Free tier = 5,000 calls/day. Register at yelp.com/developers.
- **Reddit (PRAW):** Free with app registration. Rate limited but sufficient.
- **Replica/SafeGraph:** Paid or academic partnership. MIT affiliation may help.

---

## The Killer Roadshow Slide

| What We Have Today (MVP) | What Phase 1 Unlocks |
|--------------------------|---------------------|
| 311 complaints (lagging, reporter bias) | Direct, in-the-moment perception |
| Crime incidents (lagging, only reported) | Real-time safety sentiment |
| Biennial survey (2-year lag, neighborhood) | Daily signal, block-level |
| Review text mining (business-adjacent only) | Universal coverage via existing touchpoints |
| Foot traffic proxy (infers avoidance) | Directly asks "how does this feel?" |

**"The MVP shows you what we can approximate with existing data. Phase 1 validates whether we can replace proxies with the real thing."**

---

## Target Funders

| Category | Targets |
|----------|---------|
| **City Government** | Mayor's Office, Controller's Office, SFPD, DPH |
| **Community Benefit Districts** | Downtown, Union Square, Yerba Buena, SoMa West, Castro, Japantown, Tenderloin |
| **Foundations** | SF Foundation, Hellman Foundation, Tipping Point Community, Salesforce Foundation |
| **Tech Sponsors** | Salesforce, Stripe, Airbnb, Square |
| **MIT Network** | City Science consortium members, Media Lab sponsors |

---

## Phase 1 Investment: $150,000–$200,000 (6 months)

**Team:**
- SF Partnership Sr (25%) — Advisory, stakeholder management
- SF Partnership Jr (50%) — Channel partner sign-ups
- Data Science Sr (25%) — Methodology, modeling
- Data Science Jr (50%) — Pipeline, dashboard, analysis

**Budget:** $25K/month × 6 months = $150K core + up to $50K ancillary (datasets, legal, marketing)
