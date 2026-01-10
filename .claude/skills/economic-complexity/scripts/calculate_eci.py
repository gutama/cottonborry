#!/usr/bin/env python3
"""
Economic Complexity Index (ECI) and Product Complexity Index (PCI) Calculator

Implements the eigenvalue-based method for calculating complexity indices
from trade data.

Usage:
    python calculate_eci.py <input_csv> [--output <output_csv>]
    
Input format: CSV with columns [country, product, value] or [country, product, year, value]
"""

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.preprocessing import StandardScaler
import argparse
import sys


def load_trade_data(filepath, country_col='country', product_col='product', 
                    value_col='value', year_col=None, year=None):
    """Load and optionally filter trade data."""
    df = pd.read_csv(filepath)
    
    if year_col and year:
        df = df[df[year_col] == year]
    
    return df[[country_col, product_col, value_col]].rename(
        columns={country_col: 'country', product_col: 'product', value_col: 'value'}
    )


def calculate_rca_matrix(df):
    """
    Calculate Revealed Comparative Advantage (RCA) matrix.
    
    RCA_cp = (X_cp / Σp X_cp) / (Σc X_cp / Σc,p X_cp)
    
    Returns:
        pd.DataFrame: RCA matrix (countries × products)
    """
    # Pivot to matrix form
    X_cp = df.pivot_table(
        index='country', 
        columns='product', 
        values='value', 
        fill_value=0,
        aggfunc='sum'
    )
    
    # Calculate shares
    X_c = X_cp.sum(axis=1)  # Total exports per country
    X_p = X_cp.sum(axis=0)  # Total world trade per product
    X_total = X_cp.sum().sum()  # Total world trade
    
    # RCA = (X_cp / X_c) / (X_p / X_total)
    country_share = X_cp.div(X_c, axis=0)
    world_share = X_p / X_total
    
    rca = country_share.div(world_share, axis=1)
    return rca.fillna(0)


def binarize_rca(rca_matrix, threshold=1.0):
    """Convert RCA to binary matrix (M_cp)."""
    return (rca_matrix >= threshold).astype(int)


def calculate_eci_pci(m_cp):
    """
    Calculate ECI and PCI using eigenvalue decomposition.
    
    Args:
        m_cp: Binary Country-Product matrix (pandas DataFrame)
    
    Returns:
        tuple: (eci_series, pci_series)
    """
    M = m_cp.values
    countries = m_cp.index
    products = m_cp.columns
    
    # Calculate diversity and ubiquity
    kc0 = M.sum(axis=1)
    kp0 = M.sum(axis=0)
    
    # Filter valid entries (non-zero)
    valid_c = kc0 > 0
    valid_p = kp0 > 0
    
    M_sub = M[valid_c][:, valid_p]
    kc0_sub = M_sub.sum(axis=1)
    kp0_sub = M_sub.sum(axis=0)
    
    # Avoid division by zero
    kc0_sub = np.where(kc0_sub == 0, 1e-10, kc0_sub)
    kp0_sub = np.where(kp0_sub == 0, 1e-10, kp0_sub)
    
    # --- Calculate ECI ---
    # M̃_cc' = D^-1 @ M @ U^-1 @ M.T
    weighted_products = M_sub / kp0_sub[None, :]  # M @ U^-1
    S_cc = np.dot(weighted_products, M_sub.T)      # Similarity matrix
    M_tilde_cc = S_cc / kc0_sub[:, None]          # D^-1 @ S
    
    # Eigendecomposition for countries
    eigvals_c, eigvecs_c = linalg.eig(M_tilde_cc)
    idx_c = np.argsort(np.abs(eigvals_c))[::-1]
    eigvecs_c = eigvecs_c[:, idx_c]
    
    # ECI = second eigenvector, normalized
    eci_raw = eigvecs_c[:, 1].real
    eci = (eci_raw - eci_raw.mean()) / eci_raw.std()
    
    # Sign convention: positive correlation with diversity
    if np.corrcoef(eci, kc0_sub)[0, 1] < 0:
        eci = -eci
    
    # --- Calculate PCI ---
    # M̃_pp' = U^-1 @ M.T @ D^-1 @ M
    weighted_countries = M_sub.T / kc0_sub[None, :]  # M.T @ D^-1
    S_pp = np.dot(weighted_countries, M_sub)          # Similarity matrix
    M_tilde_pp = S_pp / kp0_sub[:, None]             # U^-1 @ S
    
    # Eigendecomposition for products
    eigvals_p, eigvecs_p = linalg.eig(M_tilde_pp)
    idx_p = np.argsort(np.abs(eigvals_p))[::-1]
    eigvecs_p = eigvecs_p[:, idx_p]
    
    # PCI = second eigenvector, normalized
    pci_raw = eigvecs_p[:, 1].real
    pci = (pci_raw - pci_raw.mean()) / pci_raw.std()
    
    # Sign convention: positive correlation with ubiquity inverse (complex = rare)
    # Actually, we want PCI to correlate with ECI direction
    # Use correlation with average ECI of exporters
    avg_eci_per_product = np.dot(M_sub.T, eci) / np.maximum(kp0_sub, 1)
    if np.corrcoef(pci, avg_eci_per_product)[0, 1] < 0:
        pci = -pci
    
    # Create output series with proper indexing
    eci_series = pd.Series(index=countries, dtype=float)
    eci_series[valid_c] = eci
    eci_series[~valid_c] = np.nan
    
    pci_series = pd.Series(index=products, dtype=float)
    pci_series[valid_p] = pci
    pci_series[~valid_p] = np.nan
    
    return eci_series, pci_series


def calculate_proximity(m_cp):
    """
    Calculate product proximity matrix.
    
    φ_pp' = min(P(RCA_p≥1|RCA_p'≥1), P(RCA_p'≥1|RCA_p≥1))
    
    Returns:
        pd.DataFrame: Proximity matrix (products × products)
    """
    M = m_cp.values
    products = m_cp.columns
    
    # Co-occurrence matrix
    co_occurrence = M.T @ M
    
    # Ubiquity
    ubiquity = M.sum(axis=0)
    ubiquity = np.where(ubiquity == 0, 1e-10, ubiquity)
    
    # Conditional probability matrix
    prob_matrix = co_occurrence / ubiquity[:, None]
    
    # Minimum of both directions
    proximity = np.minimum(prob_matrix, prob_matrix.T)
    np.fill_diagonal(proximity, 0)
    
    return pd.DataFrame(proximity, index=products, columns=products)


def calculate_density(m_cp, proximity):
    """
    Calculate density matrix (how connected each product is to country's basket).
    
    d_cp = Σp' (M_cp' × φ_pp') / Σp' φ_pp'
    
    Returns:
        pd.DataFrame: Density matrix (countries × products)
    """
    numerator = m_cp.values @ proximity.values
    denominator = proximity.values.sum(axis=0)
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    density = numerator / denominator
    return pd.DataFrame(density, index=m_cp.index, columns=m_cp.columns)


def main():
    parser = argparse.ArgumentParser(description='Calculate Economic Complexity Index')
    parser.add_argument('input', help='Input CSV file with trade data')
    parser.add_argument('--output', '-o', help='Output CSV file', default='eci_results.csv')
    parser.add_argument('--country-col', default='country', help='Country column name')
    parser.add_argument('--product-col', default='product', help='Product column name')
    parser.add_argument('--value-col', default='value', help='Value column name')
    parser.add_argument('--year-col', default=None, help='Year column name (optional)')
    parser.add_argument('--year', type=int, default=None, help='Filter to specific year')
    parser.add_argument('--rca-threshold', type=float, default=1.0, help='RCA threshold')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}...")
    df = load_trade_data(
        args.input, 
        country_col=args.country_col,
        product_col=args.product_col,
        value_col=args.value_col,
        year_col=args.year_col,
        year=args.year
    )
    
    print(f"Loaded {len(df)} trade records")
    print(f"Countries: {df['country'].nunique()}, Products: {df['product'].nunique()}")
    
    print("Calculating RCA matrix...")
    rca = calculate_rca_matrix(df)
    
    print("Binarizing RCA matrix...")
    m_cp = binarize_rca(rca, threshold=args.rca_threshold)
    
    print("Calculating ECI and PCI...")
    eci, pci = calculate_eci_pci(m_cp)
    
    print("Calculating proximity matrix...")
    proximity = calculate_proximity(m_cp)
    
    print("Calculating density matrix...")
    density = calculate_density(m_cp, proximity)
    
    # Save results
    results = pd.DataFrame({
        'country': eci.index,
        'eci': eci.values,
        'diversity': m_cp.sum(axis=1).values
    }).dropna()
    results = results.sort_values('eci', ascending=False)
    results.to_csv(args.output, index=False)
    print(f"Saved country ECI to {args.output}")
    
    # Save PCI
    pci_output = args.output.replace('.csv', '_pci.csv')
    pci_results = pd.DataFrame({
        'product': pci.index,
        'pci': pci.values,
        'ubiquity': m_cp.sum(axis=0).values
    }).dropna()
    pci_results = pci_results.sort_values('pci', ascending=False)
    pci_results.to_csv(pci_output, index=False)
    print(f"Saved product PCI to {pci_output}")
    
    # Save proximity matrix
    prox_output = args.output.replace('.csv', '_proximity.csv')
    proximity.to_csv(prox_output)
    print(f"Saved proximity matrix to {prox_output}")
    
    print("\nTop 10 countries by ECI:")
    print(results.head(10).to_string(index=False))
    
    print("\nTop 10 products by PCI:")
    print(pci_results.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
