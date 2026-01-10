#!/usr/bin/env python3
"""
Strategic Diversification Opportunity Calculator

Calculates density, effort, and efficiency metrics for identifying
optimal diversification targets using the ECI-OPT framework.

Usage:
    python calculate_opportunity.py <m_cp_csv> <proximity_csv> <pci_csv> [--country <code>]
"""

import numpy as np
import pandas as pd
import argparse


def load_matrices(m_cp_path, proximity_path, pci_path):
    """Load pre-computed matrices."""
    m_cp = pd.read_csv(m_cp_path, index_col=0)
    proximity = pd.read_csv(proximity_path, index_col=0)
    pci = pd.read_csv(pci_path)
    
    # Convert PCI to series
    if 'product' in pci.columns and 'pci' in pci.columns:
        pci_series = pci.set_index('product')['pci']
    else:
        pci_series = pci.iloc[:, 0]  # Assume first column is index
    
    return m_cp, proximity, pci_series


def calculate_density_matrix(m_cp, proximity):
    """
    Calculate density for all country-product pairs.
    
    d_cp = Σp' (M_cp' × φ_pp') / Σp' φ_pp'
    """
    # Align indices
    products = m_cp.columns.intersection(proximity.columns)
    m_cp = m_cp[products]
    proximity = proximity.loc[products, products]
    
    numerator = m_cp.values @ proximity.values
    denominator = proximity.values.sum(axis=0)
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    density = numerator / denominator
    return pd.DataFrame(density, index=m_cp.index, columns=products)


def calculate_complexity_outlook(m_cp, density, pci_series):
    """
    Calculate Complexity Outlook Index (COI) for each country.
    
    COI_c = Σp (1 - M_cp) × d_cp × PCI_p
    """
    products = m_cp.columns.intersection(pci_series.index)
    m_cp = m_cp[products]
    density = density[products]
    pci = pci_series[products]
    
    # For non-exported products only
    non_exported = 1 - m_cp.values
    weighted = non_exported * density.values * pci.values[None, :]
    
    coi = weighted.sum(axis=1)
    return pd.Series(coi, index=m_cp.index, name='coi')


def calculate_opportunity_gain(m_cp, density, pci_series, proximity):
    """
    Calculate Complexity Outlook Gain (COG) for each country-product pair.
    
    This measures how much adding product p would improve country c's
    future diversification potential.
    """
    products = m_cp.columns.intersection(pci_series.index).intersection(proximity.columns)
    m_cp = m_cp[products]
    density = density[products]
    pci = pci_series[products]
    proximity = proximity.loc[products, products]
    
    results = []
    
    for country in m_cp.index:
        current_basket = m_cp.loc[country]
        non_exported = current_basket[current_basket == 0].index
        
        for product in non_exported:
            # Density for this product
            d = density.loc[country, product]
            
            # Estimate COG: how much would adding this product help reach other complex products?
            # Simplified: weighted sum of proximity × PCI for other non-exported products
            other_non_exported = [p for p in non_exported if p != product]
            if len(other_non_exported) > 0:
                prox_to_others = proximity.loc[product, other_non_exported]
                pci_of_others = pci[other_non_exported]
                cog = (prox_to_others * pci_of_others).sum()
            else:
                cog = 0
            
            results.append({
                'country': country,
                'product': product,
                'density': d,
                'effort': 1 - d,
                'pci': pci[product],
                'cog': cog
            })
    
    df = pd.DataFrame(results)
    
    # Calculate efficiency scores
    df['efficiency'] = df['pci'] / df['effort'].replace(0, 0.001)
    df['strategic_value'] = df['efficiency'] * (1 + df['cog'] / df['cog'].max())
    
    return df


def rank_opportunities(opportunities, country=None, top_n=20):
    """
    Rank opportunities for a specific country or all countries.
    """
    if country:
        df = opportunities[opportunities['country'] == country].copy()
    else:
        df = opportunities.copy()
    
    # Sort by strategic value
    df = df.sort_values('strategic_value', ascending=False)
    
    return df.head(top_n)


def summarize_country(opportunities, country):
    """Generate summary statistics for a country."""
    df = opportunities[opportunities['country'] == country]
    
    summary = {
        'country': country,
        'n_opportunities': len(df),
        'avg_density': df['density'].mean(),
        'avg_effort': df['effort'].mean(),
        'avg_pci_available': df['pci'].mean(),
        'max_pci_available': df['pci'].max(),
        'total_cog_potential': df['cog'].sum()
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Calculate diversification opportunities')
    parser.add_argument('m_cp', help='Binary M_cp matrix CSV (countries × products)')
    parser.add_argument('proximity', help='Proximity matrix CSV')
    parser.add_argument('pci', help='PCI CSV file')
    parser.add_argument('--country', '-c', help='Filter to specific country')
    parser.add_argument('--output', '-o', default='opportunities.csv', help='Output file')
    parser.add_argument('--top', '-n', type=int, default=20, help='Number of top opportunities')
    
    args = parser.parse_args()
    
    print("Loading matrices...")
    m_cp, proximity, pci_series = load_matrices(args.m_cp, args.proximity, args.pci)
    print(f"Loaded {len(m_cp)} countries, {len(m_cp.columns)} products")
    
    print("Calculating density matrix...")
    density = calculate_density_matrix(m_cp, proximity)
    
    print("Calculating Complexity Outlook Index...")
    coi = calculate_complexity_outlook(m_cp, density, pci_series)
    
    print("Calculating opportunity metrics...")
    opportunities = calculate_opportunity_gain(m_cp, density, pci_series, proximity)
    
    if args.country:
        print(f"\n=== Top {args.top} opportunities for {args.country} ===\n")
        top_opps = rank_opportunities(opportunities, args.country, args.top)
        print(top_opps[['product', 'density', 'effort', 'pci', 'cog', 'strategic_value']].to_string(index=False))
        
        summary = summarize_country(opportunities, args.country)
        print(f"\nSummary for {args.country}:")
        print(f"  Available opportunities: {summary['n_opportunities']}")
        print(f"  Average density: {summary['avg_density']:.3f}")
        print(f"  Average PCI of opportunities: {summary['avg_pci_available']:.3f}")
        print(f"  Complexity Outlook Index: {coi[args.country]:.3f}")
        
        # Save country-specific results
        output = args.output.replace('.csv', f'_{args.country}.csv')
        top_opps.to_csv(output, index=False)
        print(f"\nSaved to {output}")
    else:
        # Save all opportunities
        opportunities.to_csv(args.output, index=False)
        print(f"\nSaved all opportunities to {args.output}")
        
        # Save COI
        coi_output = args.output.replace('.csv', '_coi.csv')
        coi.to_csv(coi_output)
        print(f"Saved COI to {coi_output}")
        
        # Show top countries by COI
        print("\nTop 10 countries by Complexity Outlook Index:")
        print(coi.sort_values(ascending=False).head(10).to_string())


if __name__ == '__main__':
    main()
