# Economic Complexity Optimization for Indonesia

Implementation of **"Optimizing Economic Complexity"** by Stojkoski & Hidalgo (2025) for analyzing Indonesia's export structure and identifying strategic diversification opportunities.

## Overview

This project implements a comprehensive economic complexity analysis framework based on the paper:
> **Stojkoski, V., & Hidalgo, C. A. (2025).** Optimizing Economic Complexity. [arXiv:2503.04476](https://arxiv.org/abs/2503.04476)

The framework identifies diversification opportunities by minimizing a cost function that captures constraints imposed by an economy's pattern of specialization, providing a target-oriented optimization layer beyond traditional relatedness-complexity diagrams.

## Data Source

This analysis uses **real Harvard Atlas of Economic Complexity** trade data:
- **Classification**: HS92 (Harmonized System 1992)
- **Coverage**: 242 countries, 2485 products, years 1995-2023
- **Pre-calculated metrics**: RCA, PCI, COG (Complexity Outlook Gain), and distance

### Required Data Files

Download from the Harvard Atlas data repository and place in your data directory:
- `hs92_country_product_year_4.csv` - Main country-product trade data with pre-calculated metrics
- `product_hs92.csv` - Product metadata and names
- `umap_layout_hs92.csv` - UMAP coordinates for Product Space visualization
- `top_edges_hs92.csv` - Product Space network edges

## Key Features

### 1. Economic Complexity Metrics
- **Economic Complexity Index (ECI)**: Measures the knowledge intensity of an economy using Method of Reflections
- **Product Complexity Index (PCI)**: Measures the knowledge required to produce a product (pre-calculated from Harvard Atlas)
- **Revealed Comparative Advantage (RCA)**: Pre-calculated export competitiveness indicators

### 2. Product Space Analysis
- **Proximity Matrix**: Measures relatedness between products based on co-export patterns
- **Density**: Quantifies how close a product is to a country's current capabilities
- **UMAP Visualization**: Modern dimensionality reduction for Product Space network

### 3. Strategic Diversification Analysis
- **Complexity Outlook Gain (COG)**: Identifies high-potential diversification targets
- **Distance Metrics**: Measures feasibility of acquiring new export capabilities
- **Balanced Optimization**: Trade-off between complexity gains and feasibility constraints

## Project Structure

```
cottonborry/
├── indonesia_economic_complexity_FIXED.ipynb  # Main analysis notebook (uses real data)
├── output/                                    # Generated analysis results
└── figures/                                   # Generated visualizations
    ├── eci_rankings.png
    ├── product_space_indonesia.png
    └── strategic_opportunities.png
```

## Getting Started

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn networkx scipy scikit-learn
```

### Running the Analysis

1. **Prepare Data**:
   - Download Harvard Atlas data files to your data directory
   - Update the `DATA_DIR` path in the notebook configuration

2. **Run Analysis**:
   Open and execute `indonesia_economic_complexity_FIXED.ipynb` in Jupyter:
   ```bash
   jupyter notebook indonesia_economic_complexity_FIXED.ipynb
   ```

3. **View Results**:
   The notebook generates:
   - Indonesia's trade summary (2023: $253B exports, 1,212 products traded)
   - RCA analysis (225 products with comparative advantage)
   - ECI/PCI calculations using Method of Reflections
   - Strategic diversification recommendations
   - Product Space visualizations

## Key Results (2023 Data)

### Indonesia's Trade Profile
- **Total Exports**: $253,349,308,484
- **Total Imports**: $205,765,435,201
- **Trade Balance**: $47,583,873,283
- **Products with RCA ≥ 1**: 225

### Top Exports by RCA
1. Lignite (RCA: 75.0)
2. Edible animal products (RCA: 49.6)
3. Nickel mattes (RCA: 43.9)
4. Stainless steel ingots (RCA: 43.7)
5. Palm oil (RCA: 41.7)

### Most Complex Exports
1. Musical instruments, wind (PCI: 1.853)
2. Ceramic wares for technical use (PCI: 1.633)
3. Electrical machines with individual functions (PCI: 1.430)
4. Transparent paper (PCI: 1.257)
5. Electrical resistors (PCI: 1.235)

## Methodology

### 1. Data Loading
- Load multi-country trade data from Harvard Atlas
- Filter for analysis year (default: 2023)
- Merge with product metadata for interpretable results

### 2. RCA & Complexity Calculation
- Use pre-calculated RCA from Harvard Atlas
- Build binary country-product matrix (M_cp = 1 if RCA ≥ 1)
- Apply Method of Reflections to compute ECI and PCI

### 3. Product Space Construction
- Calculate product proximity: φ_{i,j} = min{P(RCA_i|RCA_j), P(RCA_j|RCA_i)}
- Compute density for all products relative to Indonesia's basket
- Visualize using UMAP layout from Harvard Atlas

### 4. Strategic Analysis
- Identify products with high COG (Complexity Outlook Gain)
- Analyze distance (feasibility) for potential diversification targets
- Generate strategic recommendations balancing complexity and feasibility

## Data Sources

- **Harvard Atlas of Economic Complexity**: Primary data source with pre-calculated metrics
- **UN Comtrade**: Underlying official international trade statistics
- **BACI**: CEPII's harmonized trade database

## References

### Primary Paper
- Stojkoski, V., & Hidalgo, C. A. (2025). Optimizing Economic Complexity. arXiv:2503.04476

### Foundational Papers
- Hausmann, R., & Hidalgo, C. A. (2009). The building blocks of economic complexity. PNAS, 106(26), 10570-10575.
- Hidalgo, C. A., et al. (2007). The product space conditions the development of nations. Science, 317(5837), 482-487.

## License

MIT License - see [LICENSE](LICENSE) file for details

## Contributing

Contributions are welcome! Key areas for enhancement:
- Integration with real-time data APIs
- Additional optimization algorithms
- Multi-year temporal analysis
- Sector-specific constraints
- Environmental/sustainability considerations

## Authors

- Ginanjar Utama
- Nadira Firinda

## Acknowledgments

- César A. Hidalgo and Viktor Stojkoski for the optimization framework
- The Growth Lab at Harvard University for the Atlas of Economic Complexity data
- UN Comtrade for underlying trade statistics