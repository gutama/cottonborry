# Mathematical Foundations of Economic Complexity

## Revealed Comparative Advantage (RCA)

The Balassa Index determines competitive export status:

```
RCA_cp = (X_cp / Σp X_cp) / (Σc X_cp / Σc,p X_cp)
```

Where:
- `X_cp` = Export value of country c in product p
- Numerator = Country c's share of product p in its portfolio
- Denominator = World's share of product p in total trade

**Interpretation**: RCA ≥ 1 indicates comparative advantage.

## The Binary Matrix M_cp

```python
M_cp = 1 if RCA_cp >= 1 else 0
```

This matrix is the foundation for all complexity calculations.

## Diversity and Ubiquity

**Diversity** (k_c,0): Number of products country c exports
```
k_c,0 = Σp M_cp
```

**Ubiquity** (k_p,0): Number of countries exporting product p
```
k_p,0 = Σc M_cp
```

High ECI requires: High diversity of low-ubiquity products.

## Eigenvalue Formulation of ECI

The ECI is the eigenvector of the second-largest eigenvalue of transition matrix M̃:

```
M̃ = D⁻¹ S

Where:
S_cc' = Σp (M_cp × M_c'p) / k_p,0    (Country similarity weighted by inverse ubiquity)
D = diag(k_c,0)                       (Diversity diagonal matrix)
```

**Why second eigenvector?** 
- M̃ is row-stochastic (rows sum to 1)
- Largest eigenvalue is always 1 with constant eigenvector
- Second eigenvector captures meaningful variance

### Python Implementation

```python
import numpy as np
from scipy import linalg
from sklearn.preprocessing import StandardScaler

def calculate_eci_eigenvalue(m_cp):
    """
    Calculate ECI using eigendecomposition.
    
    Args:
        m_cp: Binary Country-Product matrix (pandas DataFrame)
    
    Returns:
        pd.Series: Z-score normalized ECI values
    """
    M = m_cp.values
    countries = m_cp.index
    
    # Filter valid countries/products
    kc0 = M.sum(axis=1)
    kp0 = M.sum(axis=0)
    valid_c = kc0 > 0
    valid_p = kp0 > 0
    
    M_sub = M[valid_c][:, valid_p]
    kc0_sub = M_sub.sum(axis=1)
    kp0_sub = M_sub.sum(axis=0)
    
    # Build M̃ = D⁻¹ @ M @ U⁻¹ @ M.T
    weighted = M_sub / kp0_sub[None, :]   # M @ U⁻¹
    S = np.dot(weighted, M_sub.T)         # Similarity matrix
    M_tilde = S / kc0_sub[:, None]        # D⁻¹ @ S
    
    # Eigendecomposition
    eigvals, eigvecs = linalg.eig(M_tilde)
    idx = np.argsort(np.abs(eigvals))[::-1]
    
    # Second eigenvector (index 1)
    eci_raw = eigvecs[:, idx[1]].real
    
    # Z-score normalization
    eci = (eci_raw - eci_raw.mean()) / eci_raw.std()
    
    # Sign convention: positive correlation with diversity
    if np.corrcoef(eci, kc0_sub)[0, 1] < 0:
        eci = -eci
    
    return pd.Series(eci, index=countries[valid_c])
```

## Method of Reflections (Alternative)

Iterative approach equivalent to eigenvalue method:

```
k_c,n = (1/k_c,0) × Σp M_cp × k_p,n-1
k_p,n = (1/k_p,0) × Σc M_cp × k_c,n-1
```

Starting with `k_c,0` (diversity) and `k_p,0` (ubiquity), iterate until convergence.

**Limitations**: Convergence issues, arbitrary iteration count. Prefer eigenvalue method.

## Product Complexity Index (PCI)

PCI is the product-side dual of ECI. Calculate using the product-product matrix:

```
M̃_pp' = U⁻¹ @ M.T @ D⁻¹ @ M
```

The second eigenvector gives PCI values.

## Proximity Matrix

The proximity φ_pp' between products measures capability overlap:

```
φ_pp' = min(P(RCA_p≥1 | RCA_p'≥1), P(RCA_p'≥1 | RCA_p≥1))
```

Using conditional probabilities from co-occurrence:

```python
def calculate_proximity(m_cp):
    co_occurrence = m_cp.T @ m_cp      # Products × Products
    ubiquity = m_cp.sum(axis=0)
    
    # Conditional probability matrix
    prob_matrix = co_occurrence / ubiquity[:, None]
    
    # Minimum of both directions
    proximity = np.minimum(prob_matrix, prob_matrix.T)
    np.fill_diagonal(proximity, 0)
    
    return proximity
```

## Density Metric

Density d_cp measures how close product p is to country c's current basket:

```
d_cp = Σp' (M_cp' × φ_pp') / Σp' φ_pp'
```

Range: [0, 1]. High density = product is well within reach.

## Key Numerical Considerations

1. **Division by zero**: Add epsilon or filter zero-sum rows/columns
2. **Complex eigenvalues**: Take `.real` part (imaginary is numerical noise)
3. **Sign ambiguity**: Eigenvectors can flip sign; use diversity correlation
4. **Normalization**: Always z-score normalize final ECI values
5. **Use scipy.linalg**: More stable than numpy.linalg for large matrices
