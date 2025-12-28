"""
Download Indonesian trade data from public sources:
1. UN Comtrade API (official trade statistics)
2. Atlas of Economic Complexity (pre-processed data)
"""

import requests
import pandas as pd
import time
import json
from pathlib import Path

# Create data directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

print("="*70)
print("DOWNLOADING INDONESIAN TRADE DATA")
print("="*70)

# Indonesia country code
INDONESIA_CODE = "360"  # ISO 3166-1 numeric code
INDONESIA_ISO = "IDN"   # ISO 3-letter code

# ============================================================================
# 1. DOWNLOAD FROM ATLAS OF ECONOMIC COMPLEXITY
# ============================================================================
print("\n[1/2] Downloading from Atlas of Economic Complexity...")

# The Atlas provides pre-processed data on GitHub
atlas_base_url = "https://intl-atlas-downloads.s3.amazonaws.com"

try:
    # Download country-product export data (SITC 4-digit)
    print("  → Fetching HS92 country-product-year data...")

    # Atlas provides data in Parquet format for recent years
    # We'll try to get HS data which is most commonly used
    atlas_url = f"{atlas_base_url}/hs92/country_partner_hsproduct4digit_year.csv"

    print(f"  → Downloading from: {atlas_url[:80]}...")
    response = requests.get(atlas_url, timeout=300)

    if response.status_code == 200:
        # Save full dataset
        atlas_file = DATA_DIR / "atlas_hs92_full.csv"
        with open(atlas_file, 'wb') as f:
            f.write(response.content)
        print(f"  ✓ Downloaded full dataset: {atlas_file}")

        # Load and filter for Indonesia
        print("  → Filtering for Indonesia...")
        df = pd.read_csv(atlas_file)

        # Filter for Indonesia exports
        df_idn = df[df['location_code'] == INDONESIA_ISO].copy()

        # Save Indonesia-specific data
        idn_file = DATA_DIR / "indonesia_exports_atlas.csv"
        df_idn.to_csv(idn_file, index=False)
        print(f"  ✓ Saved Indonesia data: {idn_file}")
        print(f"    - Years: {df_idn['year'].min()} to {df_idn['year'].max()}")
        print(f"    - Products: {df_idn['hs_product_code'].nunique()} unique HS codes")
        print(f"    - Records: {len(df_idn):,} rows")
    else:
        print(f"  ✗ Atlas download failed (status {response.status_code})")
        print("  → Trying alternative Atlas endpoint...")

        # Try alternative: get country-level complexity data
        atlas_eci_url = f"{atlas_base_url}/eci_rankings/eci_rankings.csv"
        response = requests.get(atlas_eci_url, timeout=60)

        if response.status_code == 200:
            eci_file = DATA_DIR / "atlas_eci_rankings.csv"
            with open(eci_file, 'wb') as f:
                f.write(response.content)
            print(f"  ✓ Downloaded ECI rankings: {eci_file}")
        else:
            print(f"  ✗ Alternative download also failed")

except Exception as e:
    print(f"  ✗ Error downloading from Atlas: {e}")

# ============================================================================
# 2. DOWNLOAD FROM UN COMTRADE API
# ============================================================================
print("\n[2/2] Downloading from UN Comtrade API...")

def fetch_comtrade_data(reporter_code, year, freq="A", classification="HS"):
    """
    Fetch trade data from UN Comtrade API.

    Parameters:
    - reporter_code: Country code (e.g., "360" for Indonesia)
    - year: Year to fetch (e.g., "2022")
    - freq: Frequency ("A" for annual, "M" for monthly)
    - classification: Trade classification ("HS" for Harmonized System)
    """
    base_url = "https://comtradeapi.un.org/public/v1/preview"

    params = {
        'reporterCode': reporter_code,
        'period': year,
        'flowCode': 'X',  # X = Exports, M = Imports
        'partnerCode': '0',  # 0 = World (all partners)
        'classificationCode': classification,
        'freqCode': freq
    }

    try:
        response = requests.get(base_url + "/C/A/HS", params=params, timeout=60)

        if response.status_code == 200:
            data = response.json()
            return data.get('data', [])
        else:
            print(f"    ✗ Failed for year {year}: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"    ✗ Error for year {year}: {e}")
        return None

# Download data for recent years (2018-2023)
years = [2023, 2022, 2021, 2020, 2019, 2018]
all_comtrade_data = []

print(f"  → Fetching Indonesian export data for {len(years)} years...")

for year in years:
    print(f"    Downloading year {year}...", end=" ")
    data = fetch_comtrade_data(INDONESIA_CODE, str(year))

    if data:
        print(f"✓ ({len(data):,} records)")
        all_comtrade_data.extend(data)

        # Save individual year file
        year_file = DATA_DIR / f"indonesia_exports_{year}.json"
        with open(year_file, 'w') as f:
            json.dump(data, f, indent=2)

        # Be nice to the API
        time.sleep(2)
    else:
        print("✗")

if all_comtrade_data:
    # Save combined data
    combined_file = DATA_DIR / "indonesia_exports_comtrade.json"
    with open(combined_file, 'w') as f:
        json.dump(all_comtrade_data, f, indent=2)
    print(f"  ✓ Saved combined Comtrade data: {combined_file}")
    print(f"    - Total records: {len(all_comtrade_data):,}")

    # Convert to CSV for easier analysis
    df_comtrade = pd.DataFrame(all_comtrade_data)
    csv_file = DATA_DIR / "indonesia_exports_comtrade.csv"
    df_comtrade.to_csv(csv_file, index=False)
    print(f"  ✓ Saved as CSV: {csv_file}")
else:
    print("  ✗ No Comtrade data downloaded")

# ============================================================================
# 3. CREATE SAMPLE DATASET (fallback)
# ============================================================================
print("\n[3/3] Creating sample dataset for testing...")

# Create a sample dataset in case downloads failed
sample_data = {
    'year': [],
    'country_code': [],
    'country_name': [],
    'product_code': [],
    'product_name': [],
    'export_value': []
}

# Sample HS codes for Indonesia's key exports
sample_products = {
    '2709': 'Petroleum oils, crude',
    '1511': 'Palm oil',
    '2701': 'Coal',
    '4011': 'New pneumatic tires (rubber)',
    '8517': 'Telephone sets, smartphones',
    '3004': 'Medicaments',
    '8703': 'Motor cars',
    '8471': 'Automatic data processing machines',
    '7108': 'Gold',
    '0801': 'Coconuts, Brazil nuts and cashew nuts'
}

for year in range(2018, 2024):
    for hs_code, product_name in sample_products.items():
        sample_data['year'].append(year)
        sample_data['country_code'].append(INDONESIA_ISO)
        sample_data['country_name'].append('Indonesia')
        sample_data['product_code'].append(hs_code)
        sample_data['product_name'].append(product_name)
        # Random export value (just for testing)
        import random
        sample_data['export_value'].append(random.randint(1000000, 50000000))

df_sample = pd.DataFrame(sample_data)
sample_file = DATA_DIR / "indonesia_exports_sample.csv"
df_sample.to_csv(sample_file, index=False)
print(f"  ✓ Created sample dataset: {sample_file}")
print(f"    - {len(df_sample)} sample records for testing")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("DOWNLOAD COMPLETE")
print("="*70)

# List all downloaded files
print("\nFiles in data/ directory:")
for file in sorted(DATA_DIR.glob("*")):
    size_mb = file.stat().st_size / (1024 * 1024)
    print(f"  - {file.name} ({size_mb:.2f} MB)")

print("\n✓ Ready for analysis!")
print("="*70)
