# Strategic ECI Optimization

## The Product Space Framework

Countries develop by moving to products "nearby" in capability space. The Product Space is a network where:
- **Nodes** = Products
- **Edges** = Proximity (capability overlap)
- **Dense core** = Complex products (machinery, electronics, chemicals)
- **Sparse periphery** = Commodities (agriculture, raw materials)

### Key Finding
Countries have ~15% probability of developing RCA in a product at proximity 0.55 over 5 years, dropping to near-zero at proximity 0.1.

## Strategic Concepts

### 1. Capability Accumulation

Development hinges on acquiring tacit knowledge (~40,000 hours to master). Knowledge transfer mechanisms:
- Skill migration (diaspora, expats)
- FDI with technology transfer
- Joint ventures with IP licensing
- Education and training programs

### 2. Adjacent Possible

Target products that are:
- **High proximity** to current exports (achievable)
- **High PCI** (valuable)

Simple heuristic: `Opportunity = Density × PCI`

### 3. Relatedness-Complexity Frontier

Optimal diversification balances:
- **Achievability** (high relatedness/density)
- **Growth potential** (high complexity/PCI)

Trade-off: Very complex products are often distant from simple economies.

### 4. Stepping-Stone Strategy

Intermediate products can bridge capability gaps:

```
Current basket → Intermediate products → Target complex products
```

Example: Agriculture → Agro-processing → Food machinery → Industrial equipment

## ECI-OPT Framework (Stojkoski & Hidalgo, 2025)

### The Cost Function

Instead of just "nearest" products, minimize **Effort**:

```
Effort_cp = 1 - Density_cp
```

Where:
```
Density_cp = Σp' (M_cp' × φ_pp') / Σp' φ_pp'
```

### Optimization Problem

Select target products T that maximize complexity gain subject to effort budget:

```
Maximize:   ECI(M_cp + T) - ECI(M_cp)
Subject to: Σp∈T (1 - d_cp) ≤ Ω
```

Where Ω is the effort budget constraint.

### Complexity Outlook Index (COI)

Measures future diversification potential:

```
COI_c = Σp (1 - M_cp) × Density_cp × PCI_p
```

High COI = Many nearby high-complexity opportunities.

### Complexity Outlook Gain (COG)

Increase in COI if product p is acquired:

```
COG_cp = ΔCOI if product p is added to country c's basket
```

High COG products are "strategic bets" that unlock future clusters.

## Python Implementation

```python
def calculate_optimization_metrics(m_cp, proximity_df, pci_series):
    """
    Calculate strategic diversification metrics.
    
    Returns DataFrame with:
    - density: How connected to current basket
    - effort: Cost to develop (1 - density)
    - pci: Product complexity
    - efficiency: PCI / effort (greedy ranking)
    """
    # 1. Density Matrix
    numerator = m_cp @ proximity_df
    denominator = proximity_df.sum(axis=0)
    density = numerator / denominator
    
    # 2. Effort
    effort = 1 - density
    
    # 3. Build opportunity dataframe for non-exported products
    results = []
    for country in m_cp.index:
        for product in m_cp.columns:
            if m_cp.loc[country, product] == 0:  # Not currently exported
                results.append({
                    'country': country,
                    'product': product,
                    'density': density.loc[country, product],
                    'effort': effort.loc[country, product],
                    'pci': pci_series.get(product, 0)
                })
    
    df = pd.DataFrame(results)
    df['efficiency'] = df['pci'] / df['effort'].replace(0, 0.001)
    
    return df.sort_values(['country', 'efficiency'], ascending=[True, False])
```

## Case Study Examples

### South Korea
- Started as poor agricultural economy (1950s)
- Targeted technology-intensive diversification
- R&D intensity: 0.61% (1978) → >4% (today)
- Current ECI rank: #2 globally
- Key: Domestic capability building through chaebols

### Vietnam
- Current ECI rank: #51
- FDI-led pathway (Samsung = 30% of exports)
- Gap: Only 12.5% of patents are domestic
- Challenge: Capability transfer vs assembly dependence

### Ethiopia (UNCTAD Analysis)
- Low-hanging fruit: Light manufacturing (textiles, fabrics)
- Strategic bet: Iron and steel industries
- Rationale: At edge of capabilities, would significantly boost ECI

## Practical Recommendations

1. **Start with density analysis**: Identify products with d_cp > 0.3
2. **Filter by PCI**: Focus on products with above-median complexity
3. **Compute efficiency**: Rank by PCI/Effort for resource allocation
4. **Check COG**: Identify stepping-stones that unlock future clusters
5. **Consider sectors**: Group products by industry for coherent policy

### Warning Signs

- **Low-complexity trap**: All nearby products are also low-PCI
- **Enclave industrialization**: High-complexity exports without domestic linkages
- **FDI dependence**: Capability gaps despite export sophistication
