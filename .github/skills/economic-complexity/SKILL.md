---
name: economic-complexity
description: Calculate Economic Complexity Index (ECI), Product Complexity Index (PCI), Revealed Comparative Advantage (RCA), and related metrics from international trade data. Use when working with (1) ECI/PCI calculations from trade data, (2) Product Space networks and proximity matrices, (3) strategic diversification analysis (density, complexity outlook, opportunity gain), (4) country similarity embeddings using SVD or Node2Vec, (5) trade data from Harvard Atlas, OEC, CEPII BACI, or UN Comtrade. Covers py-ecomplexity, econci libraries, eigenvalue methods, and visualization with NetworkX/Plotly.
---

# Economic Complexity Analysis

This skill provides workflows for calculating economic complexity metrics, building Product Space networks, and conducting strategic diversification analysis.

## Quick Start

Install the primary library:
```bash
pip install ecomplexity --break-system-packages
```

Calculate ECI from Harvard Atlas data:
```python
from ecomplexity import ecomplexity, proximity
import pandas as pd

data = pd.read_csv("https://intl-atlas-downloads.s3.amazonaws.com/country_hsproduct2digit_year.csv.zip",
                   compression="zip", low_memory=False)
trade_cols = {'time':'year', 'loc':'location_code', 'prod':'hs_product_code', 'val':'export_value'}

cdata = ecomplexity(data, trade_cols)  # Returns ECI, PCI, RCA, density
prox = proximity(data, trade_cols)      # Returns product proximity matrix
```

## Core Workflows

### Workflow 1: ECI/PCI Calculation

**Option A - Use py-ecomplexity (recommended for production)**

```python
from ecomplexity import ecomplexity
trade_cols = {'time':'year', 'loc':'country', 'prod':'product', 'val':'export_value'}
results = ecomplexity(df, trade_cols, rca_mcp_threshold=1)
# Results contain: eci, pci, rca, mcp (binary matrix), density, coi (complexity outlook)
```

**Option B - Use econci with NetworkX integration**

```python
import econci
comp = econci.Complexity(df, c='country', p='product', values='export')
comp.calculate_indexes()
eci_scores = comp.eci
comp.create_product_space()  # Creates NetworkX graph
G = comp.product_space
```

**Option C - Manual eigenvalue implementation**

See `scripts/calculate_eci.py` for full implementation. Core steps:
1. Calculate RCA: `RCA_cp = (X_cp/ΣpX_cp) / (ΣcX_cp/ΣcpX_cp)`
2. Binarize: `M_cp = 1 if RCA >= 1 else 0`
3. Compute diversity `k_c = Σp M_cp` and ubiquity `k_p = Σc M_cp`
4. Build transition matrix `M̃ = D⁻¹ @ M @ U⁻¹ @ M.T`
5. ECI = second eigenvector of M̃ (z-score normalized)

### Workflow 2: Product Space Network

Build and visualize the Product Space:

```python
# 1. Calculate proximity (minimum conditional probability)
# φ_pp' = min(P(RCA_p≥1|RCA_p'≥1), P(RCA_p'≥1|RCA_p≥1))
from ecomplexity import proximity
prox_df = proximity(data, trade_cols)

# 2. Build graph with threshold
import networkx as nx
G = nx.Graph()
threshold = 0.55  # Standard threshold
for p1 in prox_df.columns:
    for p2 in prox_df.columns:
        if prox_df.loc[p1, p2] >= threshold:
            G.add_edge(p1, p2, weight=prox_df.loc[p1, p2])

# 3. Extract backbone via Maximum Spanning Tree
mst = nx.maximum_spanning_tree(G)
```

For visualization, see `scripts/visualize_product_space.py` or references/visualization.md.

### Workflow 3: Strategic Diversification Analysis

Identify optimal products for diversification using ECI-OPT framework:

1. **Density** - How connected is product p to country c's current basket:
   ```
   d_cp = Σp' M_cp' × φ_pp' / Σp' φ_pp'
   ```

2. **Effort** - Inverse of density (cost to acquire capabilities):
   ```
   Effort_cp = 1 - d_cp
   ```

3. **Complexity Outlook Gain (COG)** - Strategic value of acquiring product p:
   ```
   COG_cp = Σp' (PCI_p' × φ_pp') for products not yet exported
   ```

4. **Efficiency Score** - Ranking metric for opportunity selection:
   ```
   Efficiency = PCI / Effort
   ```

See `scripts/calculate_opportunity.py` for implementation.

### Workflow 4: Country Embeddings and Comparison

**SVD-based embeddings:**
```python
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=10)
country_vectors = svd.fit_transform(rca_binary_matrix)
# First component ≈ ECI, remaining capture structural variation
```

**Node2Vec for Product Space:**
```python
from node2vec import Node2Vec
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, p=1, q=1)
model = node2vec.fit(window=10)
product_vectors = {node: model.wv[node] for node in G.nodes()}
```

**Country similarity:**
```python
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(country_vectors)
```

## Data Sources

| Source | Pre-calculated | URL/Access |
|--------|---------------|------------|
| Harvard Atlas | ECI, PCI, RCA, density | `https://intl-atlas-downloads.s3.amazonaws.com/` (no API key) |
| OEC | ECI, PCI | API requires Pro subscription ($99/mo) |
| CEPII BACI | Raw bilateral HS6 | Free CSV download after registration |
| UN Comtrade | Raw trade data | `pip install comtradeapicall` (500 records/free call) |

Harvard Atlas bulk download URLs:
```python
harvard_urls = {
    'hs2': "https://intl-atlas-downloads.s3.amazonaws.com/country_hsproduct2digit_year.csv.zip",
    'hs4': "https://intl-atlas-downloads.s3.amazonaws.com/country_hsproduct4digit_year.csv.zip",
    'sitc4': "https://intl-atlas-downloads.s3.amazonaws.com/country_sitcproduct4digit_year.csv.zip"
}
```

## Key Libraries

| Library | Install | Primary Use |
|---------|---------|-------------|
| ecomplexity | `pip install ecomplexity` | Harvard's official ECI/PCI/proximity |
| econci | `pip install econci` | ECI + NetworkX Product Space |
| economic-complexity | `pip install economic-complexity` | Pandas/Polars support |
| networkx | `pip install networkx` | Graph algorithms |
| node2vec | `pip install node2vec` | Graph embeddings |
| pyvis | `pip install pyvis` | Interactive network visualization |

## Reference Files

- **references/mathematical-foundations.md** - Detailed eigenvalue formulation, Method of Reflections, sign conventions
- **references/optimization-strategy.md** - ECI-OPT framework, capability accumulation, stepping-stone strategy
- **references/visualization.md** - Plotly, pyvis, Gephi export for Product Space

## Scripts

- **scripts/calculate_eci.py** - Manual ECI calculation with eigenvalue decomposition
- **scripts/calculate_opportunity.py** - Density, effort, COG, efficiency scoring
- **scripts/visualize_product_space.py** - NetworkX + Plotly visualization
- **scripts/country_embeddings.py** - SVD and clustering for country comparison
