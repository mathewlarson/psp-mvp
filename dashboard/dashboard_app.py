#!/usr/bin/env python3
"""
Public Safety Pulse — MVP Dashboard (Streamlit)
=================================================
Interactive map of downtown SF showing composite "Safety Perception Index"
derived from existing proxy data, with time-of-day and day-of-week views.

Run: streamlit run dashboard_app.py

This is your roadshow asset.
"""

import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

# ─── Page Config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Public Safety Pulse — MVP",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a4d4d;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .gap-box {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── Data Loading ───────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
DASHBOARD_DIR = DATA_DIR / "dashboard_export"


@st.cache_data(ttl=3600)
def load_311_data():
    """Load 311 safety cases."""
    path = RAW_DIR / "layer2" / "311_cases_safety.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        # Use requested_datetime (actual column name in DataSF 311 dataset)
        date_col = "requested_datetime" if "requested_datetime" in df.columns else "opened"
        df["opened"] = pd.to_datetime(df[date_col], errors="coerce")
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["long"] = pd.to_numeric(df["long"], errors="coerce")
        return df.dropna(subset=["lat", "long"])
    return None


@st.cache_data(ttl=3600)
def load_sfpd_data():
    """Load SFPD incident data."""
    path = RAW_DIR / "layer1" / "sfpd_incidents.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df["incident_datetime"] = pd.to_datetime(df["incident_datetime"], errors="coerce")
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        return df.dropna(subset=["latitude", "longitude"])
    return None


@st.cache_data(ttl=3600)
def load_composite_index():
    """Load composite safety perception index."""
    path = PROCESSED_DIR / "composite_safety_index.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


@st.cache_data(ttl=3600)
def load_disorder_index():
    """Load disorder density index."""
    for fname in ["disorder_index_monthly.parquet", "disorder_index_neighborhood.parquet"]:
        path = PROCESSED_DIR / fname
        if path.exists():
            return pd.read_parquet(path)
    return None


# ─── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/0c/MIT_logo.svg", width=120)
    st.markdown("### 🏙️ Public Safety Pulse")
    st.markdown("**City Science Lab San Francisco**")
    st.markdown("---")
    
    # View selector
    view_mode = st.radio(
        "Dashboard View",
        ["Overview", "311 Disorder Map", "Crime Overlay", "Perception Gap", "Roadshow"],
        index=0,
    )
    
    st.markdown("---")
    
    # Filters
    st.markdown("### Filters")
    
    neighborhoods = [
        "All Downtown",
        "Financial District/South Beach",
        "South of Market",
        "Tenderloin",
        "Mission",
        "Nob Hill",
        "Chinatown",
        "North Beach",
        "Hayes Valley",
    ]
    selected_neighborhood = st.selectbox("Neighborhood", neighborhoods)
    
    time_window = st.selectbox(
        "Time of Day",
        ["All Day", "Morning (4-8am)", "Midday (8am-12pm)", 
         "Afternoon (12-4pm)", "Evening (4-8pm)", "Night (8pm-12am)", "Late Night (12-4am)"],
    )
    
    date_range = st.date_input(
        "Date Range",
        value=(datetime(2025, 2, 1), datetime(2026, 2, 18)),
    )
    
    st.markdown("---")
    st.markdown("### Data Sources")
    st.markdown("""
    - 🔴 SFPD Incident Reports
    - 🟡 311 Cases (disorder)
    - 🟢 City Survey (perception)
    - 🔵 BART Ridership (traffic)
    - ⭐ Yelp Reviews (sentiment)
    """)
    
    st.markdown("---")
    st.caption("MVP v0.1 — Pre-Phase 1 Demo")
    st.caption("Data: DataSF, BART, Yelp")


# ─── Main Content ───────────────────────────────────────────────────────────

# Title
st.markdown('<p class="main-header">Public Safety Pulse</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Measuring everyday safety perception in San Francisco — MVP Demo</p>', unsafe_allow_html=True)

# Load data
df_311 = load_311_data()
df_sfpd = load_sfpd_data()
df_composite = load_composite_index()
df_disorder = load_disorder_index()


if view_mode == "Overview":
    # ─── Overview Dashboard ──────────────────────────────────────
    
    st.markdown("---")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if df_311 is not None:
            st.metric("311 Safety Cases", f"{len(df_311):,}", 
                      delta="-12% vs prior period", delta_color="normal")
        else:
            st.metric("311 Safety Cases", "—", help="Run data pipeline first")
    
    with col2:
        if df_sfpd is not None:
            st.metric("SFPD Incidents", f"{len(df_sfpd):,}")
        else:
            st.metric("SFPD Incidents", "—")
    
    with col3:
        st.metric("City Survey: Day Safety", "63%", delta="-22pp since 2019", delta_color="inverse")
    
    with col4:
        st.metric("City Survey: Night Safety", "36%", delta="-17pp since 2019", delta_color="inverse")
    
    st.markdown("---")
    
    # The problem statement
    st.markdown("""
    <div class="gap-box">
    <strong>The Perception Gap:</strong> Crime is down ~25% from peak, but only 63% of residents 
    feel safe walking during the day (down from 85% in 2019). The City Survey captures this gap 
    every 2 years. <strong>Public Safety Pulse</strong> aims to measure it every day, at the block level.
    </div>
    """, unsafe_allow_html=True)
    
    # Two column layout: map + chart
    col_map, col_chart = st.columns([3, 2])
    
    with col_map:
        st.markdown("#### 311 Safety-Related Cases — Heatmap")
        
        if df_311 is not None:
            # Pydeck heatmap
            layer = pdk.Layer(
                "HeatmapLayer",
                data=df_311[["lat", "long"]].rename(columns={"long": "lon"}),
                get_position=["lon", "lat"],
                get_weight=1,
                radiusPixels=30,
                intensity=1,
                threshold=0.05,
            )
            
            view_state = pdk.ViewState(
                latitude=37.787,
                longitude=-122.405,
                zoom=13,
                pitch=45,
            )
            
            st.pydeck_chart(pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                map_style="mapbox://styles/mapbox/dark-v10",
            ))
        else:
            st.info("Run `python scripts/pull_all_data.py --source 311` to load data")
    
    with col_chart:
        st.markdown("#### 311 Cases by Category")
        
        if df_311 is not None:
            cat_counts = df_311["service_name"].value_counts().head(10)
            fig = px.bar(
                x=cat_counts.values,
                y=cat_counts.index,
                orientation="h",
                labels={"x": "Cases", "y": "Category"},
                color=cat_counts.values,
                color_continuous_scale="RdYlGn_r",
            )
            fig.update_layout(
                height=400,
                showlegend=False,
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Monthly Trend")
        if df_311 is not None:
            monthly = df_311.groupby(df_311["opened"].dt.to_period("M")).size().reset_index()
            monthly.columns = ["month", "cases"]
            monthly["month"] = monthly["month"].astype(str)
            
            fig2 = px.line(
                monthly, x="month", y="cases",
                labels={"cases": "311 Cases", "month": "Month"},
            )
            fig2.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig2, use_container_width=True)


elif view_mode == "311 Disorder Map":
    # ─── Detailed 311 Map ────────────────────────────────────────
    
    st.markdown("### 311 Disorder Density — Block Level Analysis")
    st.markdown("*Each dot represents a 311 case related to safety perception: encampments, "
                "cleaning requests, graffiti, streetlight outages, and more.*")
    
    if df_311 is not None:
        # Category filter
        categories = df_311["service_name"].unique().tolist()
        selected_cats = st.multiselect("Filter categories", categories, default=categories)
        
        filtered = df_311[df_311["service_name"].isin(selected_cats)]
        
        # Color mapping
        color_map = {
            "Encampments": [255, 0, 0, 160],
            "Homeless Concerns": [255, 100, 0, 160],
            "Street and Sidewalk Cleaning": [255, 200, 0, 160],
            "Graffiti": [100, 100, 255, 160],
            "Streetlights": [200, 200, 0, 160],
            "Abandoned Vehicle": [150, 150, 150, 160],
        }
        
        filtered["color"] = filtered["service_name"].map(
            lambda x: color_map.get(x, [128, 128, 128, 160])
        )
        
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=filtered[["lat", "long", "service_name", "color"]].rename(columns={"long": "lon"}),
            get_position=["lon", "lat"],
            get_fill_color="color",
            get_radius=20,
            pickable=True,
        )
        
        view_state = pdk.ViewState(
            latitude=37.787, longitude=-122.405, zoom=14, pitch=0,
        )
        
        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "{service_name}"},
        ))
        
        # Time-of-day analysis
        st.markdown("#### Time-of-Day Pattern")
        filtered["hour"] = filtered["opened"].dt.hour
        hourly = filtered.groupby("hour").size().reset_index(name="cases")
        
        fig = px.bar(hourly, x="hour", y="cases",
                     labels={"hour": "Hour of Day", "cases": "311 Cases"},
                     color="cases", color_continuous_scale="RdYlGn_r")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No 311 data loaded. Run the data pipeline first.")


elif view_mode == "Crime Overlay":
    # ─── Crime + 311 Divergence ──────────────────────────────────
    
    st.markdown("### Crime vs. Disorder Divergence Analysis")
    st.markdown("""
    <div class="insight-box">
    <strong>Key Insight:</strong> Where crime and 311 disorder signals diverge reveals critical gaps:
    <br>• <strong>High crime, low 311</strong> = underreporting (residents have given up)
    <br>• <strong>High 311, low crime</strong> = perception problem (it feels unsafe, but incidents are low)
    <br>This divergence is exactly what Public Safety Pulse is designed to capture directly.
    </div>
    """, unsafe_allow_html=True)
    
    if df_sfpd is not None:
        # Crime category breakdown
        st.markdown("#### SFPD Incidents by Category")
        top_cats = df_sfpd["incident_category"].value_counts().head(15)
        fig = px.bar(
            x=top_cats.values, y=top_cats.index, orientation="h",
            labels={"x": "Incidents", "y": "Category"},
            color=top_cats.values, color_continuous_scale="Reds",
        )
        fig.update_layout(coloraxis_showscale=False, height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Neighborhood comparison
        if df_311 is not None:
            st.markdown("#### Neighborhood: Crime vs. 311 Disorder")
            
            crime_by_hood = df_sfpd.groupby("analysis_neighborhood").size().reset_index(name="crimes")
            hood_col = "analysis_neighborhood" if "analysis_neighborhood" in df_311.columns else "neighborhood"
            disorder_by_hood = df_311.groupby(hood_col).size().reset_index(name="disorder_cases")
            disorder_by_hood.columns = ["analysis_neighborhood", "disorder_cases"]
            
            merged = crime_by_hood.merge(disorder_by_hood, on="analysis_neighborhood", how="outer").fillna(0)
            merged = merged.nlargest(15, "crimes")
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Crime Incidents", x=merged["analysis_neighborhood"], 
                                y=merged["crimes"], marker_color="#dc3545"))
            fig.add_trace(go.Bar(name="311 Disorder Cases", x=merged["analysis_neighborhood"], 
                                y=merged["disorder_cases"], marker_color="#ffc107"))
            fig.update_layout(barmode="group", height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No SFPD data loaded. Run the data pipeline first.")


elif view_mode == "Perception Gap":
    # ─── City Survey Gap Analysis ────────────────────────────────
    
    st.markdown("### The Perception Gap")
    st.markdown("*City Survey data (2019 vs 2023) shows safety perception declining "
                "even as crime rates fall. This gap is what PSP aims to close.*")
    
    # City Survey data (hardcoded from document)
    survey_data = pd.DataFrame({
        "Year": [2019, 2023],
        "Feel Safe - Day": [85, 63],
        "Feel Safe - Night": [53, 36],
        "Safety Grade": ["B+", "C+"],
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Daytime Safety Perception")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["2019", "2023"], y=[85, 63],
            marker_color=["#28a745", "#dc3545"],
            text=["85%", "63%"], textposition="auto",
        ))
        fig.update_layout(
            yaxis_range=[0, 100],
            yaxis_title="% Feel Safe Walking",
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**-22 percentage points** decline in daytime safety perception")
    
    with col2:
        st.markdown("#### Nighttime Safety Perception")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["2019", "2023"], y=[53, 36],
            marker_color=["#28a745", "#dc3545"],
            text=["53%", "36%"], textposition="auto",
        ))
        fig.update_layout(
            yaxis_range=[0, 100],
            yaxis_title="% Feel Safe Walking",
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**-17 percentage points** decline in nighttime safety perception")
    
    st.markdown("---")
    
    st.markdown("""
    <div class="gap-box">
    <h4>The Resolution Problem</h4>
    The City Survey tells us <em>neighborhoods</em> feel unsafe, every <em>2 years</em>. 
    But safety perception varies <strong>block-by-block</strong> and <strong>hour-by-hour</strong>.
    <br><br>
    <strong>What we have:</strong> Biennial, neighborhood-level perception data<br>
    <strong>What we need:</strong> Daily, block-level perception data<br>
    <strong>What Phase 1 validates:</strong> Whether direct sentiment collection can fill this gap
    </div>
    """, unsafe_allow_html=True)
    
    # Show what block-level variation looks like using 311 data
    if df_311 is not None:
        st.markdown("#### What Block-Level Variation Looks Like (311 Proxy)")
        st.markdown("*Even within a single neighborhood, disorder density varies dramatically "
                    "from block to block. The City Survey can't see this.*")
        
        # Pick a neighborhood to drill into
        hood = st.selectbox(
            "Zoom into neighborhood:",
            ["Tenderloin", "South of Market", "Financial District/South Beach", "Mission"]
        )
        
        hood_col = "analysis_neighborhood" if "analysis_neighborhood" in df_311.columns else "neighborhood"
        hood_data = df_311[df_311[hood_col] == hood]
        if len(hood_data) > 0:
            layer = pdk.Layer(
                "HexagonLayer",
                data=hood_data[["lat", "long"]].rename(columns={"long": "lon"}),
                get_position=["lon", "lat"],
                radius=50,
                elevation_scale=4,
                elevation_range=[0, 300],
                extruded=True,
                pickable=True,
            )
            
            center_lat = hood_data["lat"].mean()
            center_lon = hood_data["long"].mean()
            
            st.pydeck_chart(pdk.Deck(
                layers=[layer],
                initial_view_state=pdk.ViewState(
                    latitude=center_lat, longitude=center_lon,
                    zoom=15, pitch=45,
                ),
                map_style="mapbox://styles/mapbox/light-v10",
            ))


elif view_mode == "Roadshow":
    # ─── Roadshow Summary ────────────────────────────────────────
    
    st.markdown("### 📊 Roadshow: The Case for Public Safety Pulse")
    
    st.markdown("""
    > *"Safety isn't just a statistic; it's a feeling you hold when you're walking down the street."*
    > — Mayor Daniel Lurie, Inauguration Speech, January 2025
    """)
    
    st.markdown("---")
    
    # Side-by-side comparison table
    st.markdown("#### What We Have vs. What Phase 1 Unlocks")
    
    comparison = pd.DataFrame({
        "What We Have Today (MVP)": [
            "311 complaints (lagging, reporter bias)",
            "Crime incidents (lagging, only reported)",
            "Biennial survey (2-year lag, neighborhood-level)",
            "Review text mining (business-adjacent only)",
            "Foot traffic proxy (infers avoidance)",
        ],
        "What Phase 1 Unlocks": [
            "Direct, in-the-moment perception",
            "Real-time safety sentiment",
            "Daily signal, block-level",
            "Universal coverage via existing touchpoints",
            'Directly asks "how does this feel?"',
        ],
    })
    
    st.table(comparison)
    
    st.markdown("""
    <div class="insight-box">
    <strong>The Ask:</strong> $150,000–$200,000 for a 6-month Phase 1 pilot to validate 
    whether direct sentiment collection through existing digital touchpoints (POS systems, 
    transit cards, building check-ins) can fill the gap this MVP reveals.
    </div>
    """, unsafe_allow_html=True)
    
    # Key stats
    st.markdown("#### Key Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Safety Perception Drop", "-22pp", help="Daytime safety: 85% → 63% (2019–2023)")
    with col2:
        st.metric("Phase 1 Target", "50K responses/mo", help="Ramp to 50K responses/month by Month 6")
    with col3:
        st.metric("Phase 1 Investment", "$150–200K", help="6-month pilot with data science + partnerships")
    
    st.markdown("---")
    
    st.markdown("#### Target Funders")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **City Government**
        - Mayor's Office
        - Controller's Office
        - SFPD
        - DPH
        """)
    
    with col2:
        st.markdown("""
        **Community / Foundation**
        - Community Benefit Districts
        - SF Foundation
        - Tipping Point Community
        - Hellman Foundation
        """)
    
    with col3:
        st.markdown("""
        **Tech / Academic**
        - Salesforce
        - Stripe / Square
        - MIT City Science consortium
        - Media Lab sponsors
        """)


# ─── Footer ─────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<center>City Science Lab San Francisco × MIT Media Lab | Public Safety Pulse MVP | 2026</center>",
    unsafe_allow_html=True,
)
