#!/usr/bin/env python3
"""
Product Space Network Visualization

Builds and visualizes the Product Space network from a proximity matrix.
Supports multiple output formats: HTML (interactive), PNG (static), GEXF (Gephi).

Usage:
    python visualize_product_space.py <proximity_csv> [--pci <pci_csv>] [--output <file>]
"""

import numpy as np
import pandas as pd
import networkx as nx
import argparse
import sys


def load_data(proximity_path, pci_path=None, m_cp_path=None):
    """Load proximity matrix and optional PCI/trade data."""
    proximity = pd.read_csv(proximity_path, index_col=0)
    
    pci_series = None
    if pci_path:
        pci_df = pd.read_csv(pci_path)
        if 'product' in pci_df.columns and 'pci' in pci_df.columns:
            pci_series = pci_df.set_index('product')['pci']
        else:
            pci_series = pd.Series(pci_df.iloc[:, 1].values, index=pci_df.iloc[:, 0])
    
    m_cp = None
    if m_cp_path:
        m_cp = pd.read_csv(m_cp_path, index_col=0)
    
    return proximity, pci_series, m_cp


def build_product_space(proximity, threshold=0.55, use_mst=True):
    """
    Build Product Space graph using MST + threshold method.
    
    Args:
        proximity: Proximity matrix (products × products)
        threshold: Minimum proximity for edge inclusion
        use_mst: Whether to include Maximum Spanning Tree edges
    
    Returns:
        nx.Graph: Product Space network
    """
    products = proximity.columns.tolist()
    
    # Create full weighted graph
    G_full = nx.Graph()
    for i, p1 in enumerate(products):
        for j, p2 in enumerate(products[i+1:], i+1):
            weight = proximity.iloc[i, j]
            if weight > 0:
                G_full.add_edge(p1, p2, weight=weight)
    
    # Initialize output graph
    G = nx.Graph()
    G.add_nodes_from(products)
    
    if use_mst and len(G_full.edges()) > 0:
        # Add Maximum Spanning Tree edges (backbone)
        mst = nx.maximum_spanning_tree(G_full)
        for u, v, data in mst.edges(data=True):
            G.add_edge(u, v, weight=data['weight'])
    
    # Add edges above threshold
    for u, v, data in G_full.edges(data=True):
        if data['weight'] >= threshold and not G.has_edge(u, v):
            G.add_edge(u, v, weight=data['weight'])
    
    return G


def add_node_attributes(G, pci_series=None, m_cp=None, country=None):
    """Add PCI values and export status to nodes."""
    if pci_series is not None:
        for node in G.nodes():
            G.nodes[node]['pci'] = pci_series.get(node, 0)
    
    if m_cp is not None and country is not None:
        if country in m_cp.index:
            country_exports = m_cp.loc[country]
            for node in G.nodes():
                G.nodes[node]['exported'] = int(country_exports.get(node, 0))
    
    # Add degree
    for node in G.nodes():
        G.nodes[node]['degree'] = G.degree(node)


def detect_communities(G):
    """Detect communities using Louvain algorithm."""
    try:
        communities = nx.community.louvain_communities(G, weight='weight', seed=42)
        for i, comm in enumerate(communities):
            for node in comm:
                G.nodes[node]['community'] = i
        return len(communities)
    except:
        return 0


def visualize_plotly(G, output='product_space.html', title='Product Space'):
    """Create interactive Plotly visualization."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly not installed. Run: pip install plotly")
        return
    
    # Compute layout
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42, weight='weight')
    
    # Edge traces
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Node traces
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    
    # Color by PCI or community
    if 'pci' in G.nodes[list(G.nodes())[0]]:
        node_colors = [G.nodes[n].get('pci', 0) for n in G.nodes()]
        colorbar_title = 'PCI'
    elif 'community' in G.nodes[list(G.nodes())[0]]:
        node_colors = [G.nodes[n].get('community', 0) for n in G.nodes()]
        colorbar_title = 'Community'
    else:
        node_colors = [G.degree(n) for n in G.nodes()]
        colorbar_title = 'Degree'
    
    # Size by degree or exported status
    if 'exported' in G.nodes[list(G.nodes())[0]]:
        node_sizes = [15 if G.nodes[n].get('exported', 0) else 8 for n in G.nodes()]
    else:
        node_sizes = [8 + G.degree(n) for n in G.nodes()]
    
    # Hover text
    hover_text = []
    for n in G.nodes():
        text = f"<b>{n}</b><br>"
        if 'pci' in G.nodes[n]:
            text += f"PCI: {G.nodes[n]['pci']:.2f}<br>"
        text += f"Degree: {G.degree(n)}"
        if 'exported' in G.nodes[n]:
            text += f"<br>Exported: {'Yes' if G.nodes[n]['exported'] else 'No'}"
        hover_text.append(text)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=hover_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=node_sizes,
            color=node_colors,
            colorbar=dict(thickness=15, title=colorbar_title),
            line=dict(width=1, color='white')
        )
    )
    
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
    )
    
    fig.write_html(output)
    print(f"Saved interactive visualization to {output}")


def visualize_matplotlib(G, output='product_space.png', title='Product Space'):
    """Create static matplotlib visualization."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed. Run: pip install matplotlib")
        return
    
    fig, ax = plt.subplots(figsize=(15, 15))
    
    pos = nx.spring_layout(G, k=0.3, iterations=100, seed=42, weight='weight')
    
    # Node colors
    if 'pci' in G.nodes[list(G.nodes())[0]]:
        node_colors = [G.nodes[n].get('pci', 0) for n in G.nodes()]
    else:
        node_colors = [G.degree(n) for n in G.nodes()]
    
    # Node sizes
    if 'exported' in G.nodes[list(G.nodes())[0]]:
        node_sizes = [200 if G.nodes[n].get('exported', 0) else 50 for n in G.nodes()]
    else:
        node_sizes = [50 + 10 * G.degree(n) for n in G.nodes()]
    
    # Draw
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    nodes = nx.draw_networkx_nodes(
        G, pos, 
        node_color=node_colors, 
        node_size=node_sizes,
        cmap=plt.cm.viridis, 
        ax=ax
    )
    
    plt.colorbar(nodes, ax=ax, label='PCI / Degree')
    ax.set_title(title, fontsize=16)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved static visualization to {output}")
    plt.close()


def export_gexf(G, output='product_space.gexf'):
    """Export to GEXF format for Gephi."""
    nx.write_gexf(G, output)
    print(f"Exported GEXF to {output}")
    print("Open in Gephi for advanced visualization (ForceAtlas2 layout recommended)")


def print_stats(G):
    """Print network statistics."""
    print(f"\n=== Network Statistics ===")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Density: {nx.density(G):.4f}")
    print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    
    if nx.is_connected(G):
        print(f"Diameter: {nx.diameter(G)}")
        print(f"Average path length: {nx.average_shortest_path_length(G):.2f}")
    else:
        print(f"Connected components: {nx.number_connected_components(G)}")
    
    # Top products by degree
    degrees = dict(G.degree())
    top_hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop 10 hub products:")
    for product, degree in top_hubs:
        pci = G.nodes[product].get('pci', 'N/A')
        print(f"  {product}: degree={degree}, PCI={pci}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Product Space network')
    parser.add_argument('proximity', help='Proximity matrix CSV')
    parser.add_argument('--pci', help='PCI CSV file for node coloring')
    parser.add_argument('--m_cp', help='Binary M_cp matrix for country highlighting')
    parser.add_argument('--country', '-c', help='Country code to highlight')
    parser.add_argument('--threshold', '-t', type=float, default=0.55, help='Proximity threshold')
    parser.add_argument('--output', '-o', default='product_space.html', help='Output file')
    parser.add_argument('--format', '-f', choices=['html', 'png', 'gexf', 'all'], 
                        default='html', help='Output format')
    parser.add_argument('--no-mst', action='store_true', help='Disable MST backbone')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.proximity}...")
    proximity, pci_series, m_cp = load_data(args.proximity, args.pci, args.m_cp)
    print(f"Loaded proximity matrix: {proximity.shape}")
    
    print(f"Building Product Space (threshold={args.threshold})...")
    G = build_product_space(proximity, threshold=args.threshold, use_mst=not args.no_mst)
    
    print("Adding node attributes...")
    add_node_attributes(G, pci_series, m_cp, args.country)
    
    print("Detecting communities...")
    n_communities = detect_communities(G)
    print(f"Found {n_communities} communities")
    
    print_stats(G)
    
    # Generate visualizations
    title = f"Product Space"
    if args.country:
        title += f" - {args.country}"
    
    base_output = args.output.rsplit('.', 1)[0]
    
    if args.format in ['html', 'all']:
        visualize_plotly(G, f"{base_output}.html", title)
    
    if args.format in ['png', 'all']:
        visualize_matplotlib(G, f"{base_output}.png", title)
    
    if args.format in ['gexf', 'all']:
        export_gexf(G, f"{base_output}.gexf")


if __name__ == '__main__':
    main()
