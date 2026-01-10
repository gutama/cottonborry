#!/usr/bin/env python3
"""
Country Embeddings and Similarity Analysis

Creates vector embeddings of countries based on their export structure
for structural comparison and clustering.

Usage:
    python country_embeddings.py <m_cp_csv> [--method svd|node2vec] [--output <file>]
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import argparse


def load_rca_matrix(filepath):
    """Load binary RCA matrix."""
    df = pd.read_csv(filepath, index_col=0)
    return df


def compute_svd_embeddings(m_cp, n_components=50):
    """
    Compute country embeddings using Truncated SVD.
    
    The first component approximates ECI; remaining components
    capture additional structural variation.
    """
    # Ensure matrix is numeric
    M = m_cp.values.astype(float)
    
    # Truncated SVD
    svd = TruncatedSVD(n_components=min(n_components, min(M.shape) - 1), random_state=42)
    country_vectors = svd.fit_transform(M)
    
    # Product vectors from components
    product_vectors = svd.components_.T
    
    # Explained variance
    explained_var = svd.explained_variance_ratio_
    
    embeddings = pd.DataFrame(
        country_vectors,
        index=m_cp.index,
        columns=[f'dim_{i}' for i in range(country_vectors.shape[1])]
    )
    
    product_embeddings = pd.DataFrame(
        product_vectors,
        index=m_cp.columns,
        columns=[f'dim_{i}' for i in range(product_vectors.shape[1])]
    )
    
    return embeddings, product_embeddings, explained_var


def compute_node2vec_embeddings(m_cp, proximity_path, dimensions=64):
    """
    Compute product embeddings using Node2Vec on Product Space,
    then aggregate to country level.
    """
    try:
        from node2vec import Node2Vec
        import networkx as nx
    except ImportError:
        print("Node2Vec not installed. Run: pip install node2vec networkx")
        return None, None
    
    # Build graph from proximity
    proximity = pd.read_csv(proximity_path, index_col=0)
    G = nx.Graph()
    
    for i, p1 in enumerate(proximity.columns):
        for j, p2 in enumerate(proximity.columns[i+1:], i+1):
            weight = proximity.iloc[i, j]
            if weight > 0.3:  # Threshold
                G.add_edge(p1, p2, weight=weight)
    
    # Train Node2Vec
    node2vec = Node2Vec(
        G, 
        dimensions=dimensions, 
        walk_length=30, 
        num_walks=200, 
        p=1, 
        q=1,
        workers=4,
        seed=42
    )
    model = node2vec.fit(window=10, min_count=1)
    
    # Get product vectors
    products = list(G.nodes())
    product_vectors = np.array([model.wv[p] for p in products])
    product_embeddings = pd.DataFrame(
        product_vectors,
        index=products,
        columns=[f'dim_{i}' for i in range(dimensions)]
    )
    
    # Aggregate to country level (weighted by RCA)
    common_products = list(set(products) & set(m_cp.columns))
    m_cp_aligned = m_cp[common_products]
    prod_emb_aligned = product_embeddings.loc[common_products].values
    
    # Country vector = weighted average of product vectors
    country_vectors = m_cp_aligned.values @ prod_emb_aligned
    row_sums = m_cp_aligned.values.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    country_vectors = country_vectors / row_sums
    
    country_embeddings = pd.DataFrame(
        country_vectors,
        index=m_cp.index,
        columns=[f'dim_{i}' for i in range(dimensions)]
    )
    
    return country_embeddings, product_embeddings


def find_similar_countries(embeddings, target_country, top_n=10):
    """Find most similar countries to target."""
    if target_country not in embeddings.index:
        return None
    
    target_vec = embeddings.loc[target_country].values.reshape(1, -1)
    all_vecs = embeddings.values
    
    similarities = cosine_similarity(target_vec, all_vecs).flatten()
    sim_series = pd.Series(similarities, index=embeddings.index)
    
    return sim_series.drop(target_country).sort_values(ascending=False).head(top_n)


def cluster_countries(embeddings, n_clusters=10):
    """Cluster countries by structural similarity."""
    # Normalize
    scaler = StandardScaler()
    normalized = scaler.fit_transform(embeddings.values)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(normalized)
    
    return pd.Series(labels, index=embeddings.index, name='cluster')


def compute_similarity_matrix(embeddings):
    """Compute full country-country similarity matrix."""
    sim_matrix = cosine_similarity(embeddings.values)
    return pd.DataFrame(sim_matrix, index=embeddings.index, columns=embeddings.index)


def visualize_embeddings_2d(embeddings, output='country_embeddings.html', clusters=None):
    """Create 2D visualization of country embeddings."""
    try:
        import plotly.express as px
        import umap
    except ImportError:
        print("Install requirements: pip install plotly umap-learn")
        return
    
    # Reduce to 2D with UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    coords = reducer.fit_transform(embeddings.values)
    
    # Build dataframe for plotting
    plot_df = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'country': embeddings.index
    })
    
    if clusters is not None:
        plot_df['cluster'] = clusters.values.astype(str)
        fig = px.scatter(
            plot_df, x='x', y='y', 
            text='country', 
            color='cluster',
            title='Country Embeddings (UMAP projection)'
        )
    else:
        fig = px.scatter(
            plot_df, x='x', y='y', 
            text='country',
            title='Country Embeddings (UMAP projection)'
        )
    
    fig.update_traces(textposition='top center', marker=dict(size=10))
    fig.update_layout(showlegend=True)
    fig.write_html(output)
    print(f"Saved 2D visualization to {output}")


def main():
    parser = argparse.ArgumentParser(description='Country embedding analysis')
    parser.add_argument('m_cp', help='Binary M_cp matrix CSV')
    parser.add_argument('--proximity', help='Proximity matrix for Node2Vec method')
    parser.add_argument('--method', choices=['svd', 'node2vec'], default='svd',
                        help='Embedding method')
    parser.add_argument('--dimensions', '-d', type=int, default=50,
                        help='Embedding dimensions')
    parser.add_argument('--clusters', '-k', type=int, default=10,
                        help='Number of clusters')
    parser.add_argument('--country', '-c', help='Find similar countries to this one')
    parser.add_argument('--output', '-o', default='embeddings', help='Output prefix')
    parser.add_argument('--visualize', '-v', action='store_true', help='Create 2D plot')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.m_cp}...")
    m_cp = load_rca_matrix(args.m_cp)
    print(f"Loaded {len(m_cp)} countries, {len(m_cp.columns)} products")
    
    print(f"Computing {args.method.upper()} embeddings...")
    if args.method == 'svd':
        country_emb, product_emb, explained_var = compute_svd_embeddings(m_cp, args.dimensions)
        print(f"Explained variance (top 5): {explained_var[:5].round(4)}")
        print(f"Total explained: {explained_var.sum():.2%}")
    else:
        if not args.proximity:
            print("Error: Node2Vec requires --proximity argument")
            return
        country_emb, product_emb = compute_node2vec_embeddings(
            m_cp, args.proximity, args.dimensions
        )
        if country_emb is None:
            return
    
    print(f"Generated {country_emb.shape[1]}-dimensional embeddings")
    
    # Clustering
    print(f"Clustering into {args.clusters} groups...")
    clusters = cluster_countries(country_emb, args.clusters)
    
    # Show cluster distribution
    print("\nCluster sizes:")
    print(clusters.value_counts().sort_index())
    
    # Save embeddings
    country_emb.to_csv(f"{args.output}_countries.csv")
    print(f"\nSaved country embeddings to {args.output}_countries.csv")
    
    if product_emb is not None:
        product_emb.to_csv(f"{args.output}_products.csv")
        print(f"Saved product embeddings to {args.output}_products.csv")
    
    # Save clusters
    cluster_df = pd.DataFrame({'country': clusters.index, 'cluster': clusters.values})
    cluster_df.to_csv(f"{args.output}_clusters.csv", index=False)
    print(f"Saved clusters to {args.output}_clusters.csv")
    
    # Save similarity matrix
    sim_matrix = compute_similarity_matrix(country_emb)
    sim_matrix.to_csv(f"{args.output}_similarity.csv")
    print(f"Saved similarity matrix to {args.output}_similarity.csv")
    
    # Find similar countries
    if args.country:
        print(f"\n=== Countries most similar to {args.country} ===")
        similar = find_similar_countries(country_emb, args.country)
        if similar is not None:
            for country, score in similar.items():
                print(f"  {country}: {score:.4f}")
        else:
            print(f"Country '{args.country}' not found in data")
    
    # Visualization
    if args.visualize:
        visualize_embeddings_2d(country_emb, f"{args.output}_2d.html", clusters)
    
    # Show sample clusters
    print("\n=== Sample cluster compositions ===")
    for i in range(min(3, args.clusters)):
        members = clusters[clusters == i].index.tolist()
        print(f"Cluster {i}: {', '.join(members[:10])}{'...' if len(members) > 10 else ''}")


if __name__ == '__main__':
    main()
