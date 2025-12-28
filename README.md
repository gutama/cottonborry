# Economic Complexity Optimization for Indonesia

Implementation of **"Optimizing Economic Complexity"** by Stojkoski & Hidalgo (2025) for analyzing Indonesia's export structure and identifying strategic diversification opportunities.

## Overview

This project implements a comprehensive economic complexity analysis framework based on the paper:
> **Stojkoski, V., & Hidalgo, C. A. (2025).** Optimizing Economic Complexity. [arXiv:2503.04476](https://arxiv.org/abs/2503.04476)

The framework identifies diversification opportunities by minimizing a cost function that captures constraints imposed by an economy's pattern of specialization, providing a target-oriented optimization layer beyond traditional relatedness-complexity diagrams.

## Key Features

### 1. Economic Complexity Metrics
- **Economic Complexity Index (ECI)**: Measures the knowledge intensity of an economy
- **Product Complexity Index (PCI)**: Measures the knowledge required to produce a product
- **Method of Reflections**: Iterative algorithm for calculating ECI and PCI

### 2. Product Space Analysis
- **Proximity Matrix**: Measures relatedness between products based on co-export patterns
- **Density**: Quantifies how close a product is to a country's current capabilities
- **Product Space Network**: Visualizes the network of related products

### 3. Optimization Framework
- **Cost Function**: Balances complexity gains with feasibility constraints
- **Multiple Strategies**: Complexity-focused, feasibility-focused, and balanced optimization
- **Strategic Recommendations**: Data-driven identification of diversification opportunities

## Project Structure

```
cottonborry/
├── economic_complexity_indonesia.ipynb  # Main analysis notebook
├── create_realistic_data.py             # Generate synthetic trade data
├── fetch_trade_data.py                  # Download from public APIs
├── download_data.py                     # Helper for Google Drive downloads
├── data/                                # Trade data directory
│   ├── country_product_exports.csv      # Full dataset (20 countries, 40 products)
│   └── indonesia_exports.csv            # Indonesia-specific data
└── figures/                             # Generated visualizations
    ├── eci_by_country.png
    ├── relatedness_complexity_indonesia.png
    └── product_space_network_indonesia.png
```

## Getting Started

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn networkx scipy scikit-learn
```

### Running the Analysis

1. **Generate/Load Data**:
   ```bash
   # Option 1: Create synthetic realistic data
   python create_realistic_data.py

   # Option 2: Download from public APIs (requires external access)
   python fetch_trade_data.py
   ```

2. **Run Analysis**:
   Open and execute `economic_complexity_indonesia.ipynb` in Jupyter:
   ```bash
   jupyter notebook economic_complexity_indonesia.ipynb
   ```

3. **View Results**:
   The notebook generates:
   - Economic Complexity Index rankings
   - Product Complexity Index for all products
   - Strategic diversification recommendations for Indonesia
   - Visualizations in the `figures/` directory

## Methodology

### 1. Data Processing
- Calculate Revealed Comparative Advantage (RCA) for country-product pairs
- Convert to binary matrix (RCA ≥ 1)

### 2. Complexity Calculation
- Apply Method of Reflections to compute ECI and PCI
- Standardize indices (mean=0, std=1)

### 3. Product Space Construction
- Calculate product proximity: φ_{i,j} = min{P(RCA_i|RCA_j), P(RCA_j|RCA_i)}
- Compute density for all products relative to Indonesia's basket

### 4. Optimization
- Define cost function balancing complexity and feasibility
- Run greedy optimization with different α/β parameters
- Generate strategic recommendations

## Key Results (Sample Data)

### Indonesia's Economic Complexity
- **Current ECI**: Calculated relative to 20 comparison countries
- **Current Exports**: Mix of resource-based and manufactured products
- **Complexity Distribution**: Analyzed across low/medium/high complexity tiers

### Strategic Recommendations
The optimization framework identifies products that:
1. Build on existing capabilities (high density/relatedness)
2. Move toward higher complexity (higher PCI)
3. Balance feasibility with ambition

## Visualization Examples

1. **ECI by Country**: Bar chart showing economic complexity rankings
2. **Relatedness-Complexity Diagram**: Scatter plot of opportunities with optimal recommendations highlighted
3. **Product Space Network**: Network visualization showing product relationships

## Data Sources

When using real data, recommended sources include:
- **UN Comtrade**: Official international trade statistics
- **Atlas of Economic Complexity**: Pre-processed complexity data
- **BACI**: CEPII's harmonized trade database
- **OEC (Observatory of Economic Complexity)**: Visualization and data platform

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

## Author

Ginanjar Utama

## Acknowledgments

- César A. Hidalgo and Viktor Stojkoski for the optimization framework
- The Atlas of Economic Complexity team
- UN Comtrade for trade statistics