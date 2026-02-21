#!/usr/bin/env python3
"""
Quick test: verify all free API endpoints are accessible.
Run this first to confirm connectivity before the full pipeline.

Usage: python test_api_access.py
"""

import requests
import sys

TESTS = [
    {
        "name": "DataSF: 311 Cases (Socrata API)",
        "url": "https://data.sfgov.org/resource/vw6y-z8j6.json",
        "params": {"$limit": 1},
        "expect": "list with 1 record",
    },
    {
        "name": "DataSF: SFPD Incidents (Socrata API)",
        "url": "https://data.sfgov.org/resource/wg3w-h783.json",
        "params": {"$limit": 1},
        "expect": "list with 1 record",
    },
    {
        "name": "DataSF: Fire Calls (Socrata API)",
        "url": "https://data.sfgov.org/resource/nuek-vuh3.json",
        "params": {"$limit": 1},
        "expect": "list with 1 record",
    },
    {
        "name": "DataSF: Traffic Crashes (Socrata API)",
        "url": "https://data.sfgov.org/resource/ubvf-ztfx.json",
        "params": {"$limit": 1},
        "expect": "list with 1 record",
    },
]

def main():
    print("Public Safety Pulse — API Connectivity Test")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test in TESTS:
        try:
            resp = requests.get(test["url"], params=test.get("params"), timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    print(f"  ✅ {test['name']}")
                    passed += 1
                else:
                    print(f"  ⚠️  {test['name']} — empty response")
                    failed += 1
            else:
                print(f"  ❌ {test['name']} — HTTP {resp.status_code}")
                failed += 1
        except Exception as e:
            print(f"  ❌ {test['name']} — {e}")
            failed += 1
    
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(TESTS)} tests")
    
    if failed == 0:
        print("\n✅ All APIs accessible! Ready to run the full pipeline.")
        print("   → python scripts/pull_all_data.py --all")
    else:
        print("\n⚠️  Some APIs failed. Check network connectivity.")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
