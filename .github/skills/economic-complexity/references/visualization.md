# Product Space Visualization

## Network Construction Strategy

Raw proximity matrices are fully connected ("hairball"). Use two-step sparsification:

1. **Maximum Spanning Tree (MST)**: Connect all products via strongest edges
2. **Threshold filtering**: Add back edges with φ ≥ 0.55

```python
import networkx as nx

def build_product_space(proximity_df, threshold=0.55):
    """Build Product Space graph with MST + threshold."""
    # Full graph from proximity matrix
    G_full = nx.from_pandas_adjacency(proximity_df)
    
    # Extract MST backbone
    mst = nx.maximum_spanning_tree(G_full)
    
    # Add strong edges above threshold
    G = nx.Graph()
    G.add_edges_from(mst.edges(data=True))
    
    for u, v, data in G_full.edges(data=True):
        if data['weight'] >= threshold and not G.has_edge(u, v):
            G.add_edge(u, v, **data)
    
    return G
```

## Community Detection

Identify product clusters using Louvain algorithm:

```python
communities = nx.community.louvain_communities(G, weight='weight')
for i, comm in enumerate(communities):
    for node in comm:
        G.nodes[node]['community'] = i
```

## Visualization Options

### Option 1: Plotly (Interactive Web)

```python
import plotly.graph_objects as go

def plot_product_space_plotly(G, pci_values=None):
    # Force-directed layout
    pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
    
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
    node_colors = [pci_values.get(n, 0) for n in G.nodes()] if pci_values else [0]*len(G)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=[str(n) for n in G.nodes()],
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=10,
            color=node_colors,
            colorbar=dict(thickness=15, title='PCI')
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
            title='Product Space Network',
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        ))
    return fig
```

### Option 2: PyVis (Interactive HTML)

```python
from pyvis.network import Network

def plot_product_space_pyvis(G, output='product_space.html'):
    net = Network(height='750px', width='100%', bgcolor='#222222')
    net.barnes_hut()  # Force-directed physics
    net.from_nx(G)
    net.show(output)
```

### Option 3: Gephi Export

For publication-quality static images:

```python
# Export to GEXF format
nx.write_gexf(G, 'product_space.gexf')

# Then open in Gephi for:
# - ForceAtlas2 layout
# - Node sizing by PCI
# - Color by community
# - Edge bundling
```

### Option 4: Matplotlib (Static)

```python
import matplotlib.pyplot as plt

def plot_product_space_mpl(G, pci_values=None):
    fig, ax = plt.subplots(figsize=(15, 15))
    pos = nx.spring_layout(G, k=0.3, iterations=100, seed=42)
    
    # Color nodes by PCI
    if pci_values:
        colors = [pci_values.get(n, 0) for n in G.nodes()]
    else:
        colors = 'lightblue'
    
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=colors, 
                           cmap=plt.cm.viridis, node_size=50, ax=ax)
    
    ax.set_title('Product Space')
    ax.axis('off')
    return fig
```

## Highlighting Country Position

Show which products a country exports:

```python
def highlight_country(G, country_rca, pos=None):
    """
    Highlight products exported by a country.
    
    Args:
        G: Product Space graph
        country_rca: Series of RCA values for one country
        pos: Pre-computed layout positions
    """
    if pos is None:
        pos = nx.spring_layout(G, k=0.15, seed=42)
    
    exported = country_rca[country_rca >= 1].index
    
    colors = ['red' if n in exported else 'lightgray' for n in G.nodes()]
    sizes = [100 if n in exported else 20 for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes)
    nx.draw_networkx_edges(G, pos, alpha=0.2)
```

## Country Embedding Visualization

Plot countries in 2D embedding space:

```python
from sklearn.decomposition import TruncatedSVD
import umap

def visualize_country_embeddings(rca_binary, method='umap'):
    """
    Embed countries in 2D for structural comparison.
    """
    # Get high-dimensional embeddings
    svd = TruncatedSVD(n_components=50, random_state=42)
    country_vectors = svd.fit_transform(rca_binary)
    
    # Reduce to 2D
    if method == 'umap':
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
        coords = reducer.fit_transform(country_vectors)
    else:  # PCA
        coords = country_vectors[:, :2]
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(coords[:, 0], coords[:, 1], alpha=0.7)
    
    for i, country in enumerate(rca_binary.index):
        plt.annotate(country, (coords[i, 0], coords[i, 1]), fontsize=8)
    
    plt.title('Country Embeddings (Structural Similarity)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    return plt.gcf()
```

## Layout Algorithms

| Algorithm | Use Case | NetworkX Function |
|-----------|----------|-------------------|
| Spring (Fruchterman-Reingold) | Default, clusters connected nodes | `nx.spring_layout()` |
| Kamada-Kawai | Better global structure | `nx.kamada_kawai_layout()` |
| Spectral | Based on graph Laplacian | `nx.spectral_layout()` |
| ForceAtlas2 | Large graphs (in Gephi) | Export to GEXF |

## Tips for Publication-Quality Figures

1. **Use consistent seed**: `seed=42` for reproducibility
2. **Adjust k parameter**: Higher k = more spacing between nodes
3. **Increase iterations**: More iterations = better convergence
4. **Edge alpha**: Keep low (0.2-0.4) to avoid clutter
5. **Node sizing**: Size by degree, betweenness, or PCI
6. **Color legend**: Always include colorbar for continuous values
7. **Export high-res**: Use `dpi=300` for print quality
