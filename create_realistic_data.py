"""
Create realistic Indonesian trade data based on actual export patterns.
This provides a comprehensive dataset for demonstrating the economic complexity analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)  # For reproducibility

# Create data directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

print("Creating comprehensive trade dataset...")

# ============================================================================
# REALISTIC COUNTRY-PRODUCT MATRIX
# ============================================================================

# Countries (mix of developed and developing)
countries = {
    'IDN': 'Indonesia',
    'USA': 'United States',
    'CHN': 'China',
    'JPN': 'Japan',
    'DEU': 'Germany',
    'SGP': 'Singapore',
    'MYS': 'Malaysia',
    'THA': 'Thailand',
    'VNM': 'Vietnam',
    'IND': 'India',
    'KOR': 'South Korea',
    'AUS': 'Australia',
    'BRA': 'Brazil',
    'MEX': 'Mexico',
    'CHL': 'Chile',
    'POL': 'Poland',
    'TUR': 'Turkey',
    'SAU': 'Saudi Arabia',
    'ZAF': 'South Africa',
    'NGA': 'Nigeria'
}

# HS 4-digit products categorized by complexity
# (Based on economic complexity literature and Indonesia's export basket)

# Low complexity products (resource-based)
low_complexity = {
    '2709': ('Petroleum oils, crude', 'Natural Resources'),
    '2701': ('Coal', 'Natural Resources'),
    '2710': ('Petroleum oils, refined', 'Natural Resources'),
    '1511': ('Palm oil', 'Agriculture'),
    '0801': ('Coconuts, Brazil nuts', 'Agriculture'),
    '4001': ('Natural rubber', 'Agriculture'),
    '4407': ('Wood sawn lengthwise', 'Forestry'),
    '4403': ('Wood in the rough', 'Forestry'),
    '0803': ('Bananas', 'Agriculture'),
    '0901': ('Coffee', 'Agriculture'),
    '1801': ('Cocoa beans', 'Agriculture'),
    '2603': ('Copper ores', 'Mining'),
    '2608': ('Zinc ores', 'Mining'),
    '7108': ('Gold', 'Mining'),
}

# Medium complexity products (basic manufacturing)
medium_complexity = {
    '6403': ('Footwear, leather', 'Light Manufacturing'),
    '6110': ('Jerseys, pullovers', 'Textiles'),
    '6203': ('Mens suits, cotton', 'Textiles'),
    '6204': ('Womens suits', 'Textiles'),
    '4011': ('Pneumatic tires, rubber', 'Manufacturing'),
    '7210': ('Flat-rolled iron/steel', 'Metals'),
    '7308': ('Structures of iron/steel', 'Metals'),
    '3004': ('Medicaments', 'Pharmaceuticals'),
    '3923': ('Articles for transport/packing, plastic', 'Plastics'),
    '3920': ('Plates, sheets, plastic', 'Plastics'),
    '4819': ('Cartons, boxes, paper', 'Paper'),
    '8704': ('Motor vehicles for goods', 'Automotive'),
    '8708': ('Vehicle parts', 'Automotive'),
}

# High complexity products (advanced manufacturing)
high_complexity = {
    '8517': ('Telephone sets, smartphones', 'Electronics'),
    '8471': ('Automatic data processing machines', 'Electronics'),
    '8473': ('Parts for office machines', 'Electronics'),
    '8542': ('Electronic integrated circuits', 'Electronics'),
    '8541': ('Diodes, transistors', 'Electronics'),
    '8528': ('Monitors, projectors', 'Electronics'),
    '8703': ('Motor cars', 'Automotive'),
    '8407': ('Spark-ignition engines', 'Machinery'),
    '8483': ('Transmission shafts', 'Machinery'),
    '9018': ('Medical instruments', 'Medical'),
    '2933': ('Heterocyclic compounds', 'Chemicals'),
    '2902': ('Cyclic hydrocarbons', 'Chemicals'),
    '3002': ('Human blood, vaccines', 'Pharmaceuticals'),
}

# Combine all products
all_products = {}
all_products.update(low_complexity)
all_products.update(medium_complexity)
all_products.update(high_complexity)

# ============================================================================
# GENERATE COUNTRY-PRODUCT EXPORT MATRIX
# ============================================================================

# Define export patterns for different country types
def generate_country_exports(country_code, country_name):
    """Generate realistic export pattern based on country development level."""

    exports = []

    # Developed countries (high complexity focus)
    if country_code in ['USA', 'DEU', 'JPN', 'KOR', 'SGP']:
        # High probability for high complexity
        for hs_code, (product_name, category) in high_complexity.items():
            if np.random.random() > 0.2:  # 80% export these
                value = np.random.lognormal(15, 2)  # Large exports
                exports.append({
                    'country_code': country_code,
                    'country_name': country_name,
                    'hs_code': hs_code,
                    'product_name': product_name,
                    'category': category,
                    'export_value_usd': value,
                    'complexity_tier': 'high'
                })

        # Medium probability for medium complexity
        for hs_code, (product_name, category) in medium_complexity.items():
            if np.random.random() > 0.3:  # 70% export these
                value = np.random.lognormal(14, 2)
                exports.append({
                    'country_code': country_code,
                    'country_name': country_name,
                    'hs_code': hs_code,
                    'product_name': product_name,
                    'category': category,
                    'export_value_usd': value,
                    'complexity_tier': 'medium'
                })

        # Lower probability for low complexity
        for hs_code, (product_name, category) in low_complexity.items():
            if np.random.random() > 0.6:  # 40% export these
                value = np.random.lognormal(12, 1.5)
                exports.append({
                    'country_code': country_code,
                    'country_name': country_name,
                    'hs_code': hs_code,
                    'product_name': product_name,
                    'category': category,
                    'export_value_usd': value,
                    'complexity_tier': 'low'
                })

    # Middle-income countries (mixed, resource + manufacturing)
    elif country_code in ['IDN', 'MYS', 'THA', 'CHN', 'MEX', 'BRA', 'POL', 'TUR']:
        # Indonesia has strong resource base + growing manufacturing

        # High probability for low complexity (resources)
        for hs_code, (product_name, category) in low_complexity.items():
            if np.random.random() > 0.15:  # 85% export these
                # Indonesia is especially strong in palm oil, coal, gas
                multiplier = 1.5 if country_code == 'IDN' and hs_code in ['1511', '2701', '2709'] else 1.0
                value = np.random.lognormal(14, 2) * multiplier
                exports.append({
                    'country_code': country_code,
                    'country_name': country_name,
                    'hs_code': hs_code,
                    'product_name': product_name,
                    'category': category,
                    'export_value_usd': value,
                    'complexity_tier': 'low'
                })

        # Medium probability for medium complexity
        for hs_code, (product_name, category) in medium_complexity.items():
            if np.random.random() > 0.35:  # 65% export these
                value = np.random.lognormal(13, 2)
                exports.append({
                    'country_code': country_code,
                    'country_name': country_name,
                    'hs_code': hs_code,
                    'product_name': product_name,
                    'category': category,
                    'export_value_usd': value,
                    'complexity_tier': 'medium'
                })

        # Lower probability for high complexity
        for hs_code, (product_name, category) in high_complexity.items():
            # China is exception - high in electronics
            prob = 0.3 if country_code == 'CHN' else 0.6
            if np.random.random() > prob:
                value = np.random.lognormal(12, 2)
                exports.append({
                    'country_code': country_code,
                    'country_name': country_name,
                    'hs_code': hs_code,
                    'product_name': product_name,
                    'category': category,
                    'export_value_usd': value,
                    'complexity_tier': 'high'
                })

    # Lower-income countries (resource-focused)
    else:
        # High probability for low complexity only
        for hs_code, (product_name, category) in low_complexity.items():
            if np.random.random() > 0.25:  # 75% export these
                value = np.random.lognormal(13, 1.8)
                exports.append({
                    'country_code': country_code,
                    'country_name': country_name,
                    'hs_code': hs_code,
                    'product_name': product_name,
                    'category': category,
                    'export_value_usd': value,
                    'complexity_tier': 'low'
                })

        # Low probability for medium complexity
        for hs_code, (product_name, category) in medium_complexity.items():
            if np.random.random() > 0.7:  # 30% export these
                value = np.random.lognormal(11, 1.5)
                exports.append({
                    'country_code': country_code,
                    'country_name': country_name,
                    'hs_code': hs_code,
                    'product_name': product_name,
                    'category': category,
                    'export_value_usd': value,
                    'complexity_tier': 'medium'
                })

        # Very low probability for high complexity
        for hs_code, (product_name, category) in high_complexity.items():
            if np.random.random() > 0.9:  # 10% export these
                value = np.random.lognormal(10, 1.5)
                exports.append({
                    'country_code': country_code,
                    'country_name': country_name,
                    'hs_code': hs_code,
                    'product_name': product_name,
                    'category': category,
                    'export_value_usd': value,
                    'complexity_tier': 'high'
                })

    return exports

# Generate exports for all countries
print("Generating export data for 20 countries...")
all_exports = []
for code, name in countries.items():
    exports = generate_country_exports(code, name)
    all_exports.extend(exports)
    print(f"  ✓ {name}: {len(exports)} products exported")

# Create DataFrame
df_exports = pd.DataFrame(all_exports)

# Add year column (use 2022 as reference year)
df_exports['year'] = 2022

print(f"\nDataset created:")
print(f"  - Total records: {len(df_exports):,}")
print(f"  - Countries: {df_exports['country_code'].nunique()}")
print(f"  - Products (HS codes): {df_exports['hs_code'].nunique()}")
print(f"  - Indonesia exports: {len(df_exports[df_exports['country_code']=='IDN'])} products")

# Save full dataset
full_file = DATA_DIR / "country_product_exports.csv"
df_exports.to_csv(full_file, index=False)
print(f"\n✓ Saved: {full_file}")

# Save Indonesia-only dataset
df_indonesia = df_exports[df_exports['country_code'] == 'IDN'].copy()
idn_file = DATA_DIR / "indonesia_exports.csv"
df_indonesia.to_csv(idn_file, index=False)
print(f"✓ Saved: {idn_file}")

# Create summary statistics
print("\n" + "="*70)
print("INDONESIA EXPORT SUMMARY (2022)")
print("="*70)
print(f"Total products exported: {len(df_indonesia)}")
print(f"Total export value: ${df_indonesia['export_value_usd'].sum()/1e9:.2f}B")
print(f"\nBy complexity tier:")
for tier in ['low', 'medium', 'high']:
    df_tier = df_indonesia[df_indonesia['complexity_tier'] == tier]
    count = len(df_tier)
    value = df_tier['export_value_usd'].sum() / 1e9
    pct = (value / df_indonesia['export_value_usd'].sum()) * 100
    print(f"  {tier.capitalize()}: {count} products, ${value:.2f}B ({pct:.1f}%)")

print(f"\nTop 10 Indonesian exports:")
top10 = df_indonesia.nlargest(10, 'export_value_usd')[['product_name', 'export_value_usd', 'complexity_tier']]
for idx, row in top10.iterrows():
    print(f"  - {row['product_name']}: ${row['export_value_usd']/1e9:.2f}B ({row['complexity_tier']})")

print("\n" + "="*70)
print("✓ Dataset ready for economic complexity analysis!")
print("="*70)
