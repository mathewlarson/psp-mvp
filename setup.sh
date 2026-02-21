#!/bin/bash
# Public Safety Pulse MVP — Quick Setup
# Run: bash setup.sh

set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Public Safety Pulse — MVP Setup                        ║"
echo "║  City Science Lab San Francisco                         ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "→ Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "→ Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "→ Installing dependencies..."
pip install -q -r requirements.txt

# Create data directories
echo "→ Creating data directories..."
mkdir -p data/{raw/{layer1,layer2,layer3},processed,dashboard_export}

# Pull critical data sources
echo ""
echo "═══════════════════════════════════════════════"
echo "PHASE 1: Pulling 311 Cases (MVP Workhorse)..."
echo "═══════════════════════════════════════════════"
python scripts/pull_all_data.py --source 311 --months 12

echo ""
echo "═══════════════════════════════════════════════"
echo "PHASE 2: Pulling SFPD Incident Reports..."
echo "═══════════════════════════════════════════════"
python scripts/pull_all_data.py --source sfpd --months 12

echo ""
echo "═══════════════════════════════════════════════"
echo "PHASE 3: Processing into Composite Index..."
echo "═══════════════════════════════════════════════"
python scripts/pull_all_data.py --process

echo ""
echo "═══════════════════════════════════════════════"
echo "PHASE 4: Exporting for Dashboard..."
echo "═══════════════════════════════════════════════"
python scripts/pull_all_data.py --export

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✓ Setup Complete!                                      ║"
echo "║                                                         ║"
echo "║  Launch the dashboard:                                  ║"
echo "║    cd dashboard && streamlit run dashboard_app.py       ║"
echo "║                                                         ║"
echo "║  Pull additional data:                                  ║"
echo "║    python scripts/pull_all_data.py --all                ║"
echo "║                                                         ║"
echo "║  Run NLP analysis (after setting YELP_API_KEY):         ║"
echo "║    python scripts/sentiment_analysis.py --all           ║"
echo "║                                                         ║"
echo "║  See manual download instructions:                      ║"
echo "║    python scripts/pull_all_data.py --manual             ║"
echo "╚══════════════════════════════════════════════════════════╝"
