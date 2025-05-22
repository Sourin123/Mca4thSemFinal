import time

import exp2
import networkx as nx


def cluster_and_plot_graph():
    import exp2
    import networkx as nx
    import numpy as np
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt

    graph = exp2.DynamicGraphSimulator()
    # Get the current snapshot of the graph
    G = graph.get_graph_snapshot()

    # Get node degrees
    degrees = dict(G.degree())
    nodes = list(G.nodes())
    degree_values = np.array([degrees[n] for n in nodes]).reshape(-1, 1)

    # Set number of clusters (e.g., 4 or less if not enough nodes)
    num_clusters = min(4, len(nodes))

    # Run standard KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(degree_values)

    # Print cluster assignments
    clusters = {i: [] for i in range(num_clusters)}
    for idx, label in enumerate(labels):
        clusters[label].append(nodes[idx])

    for c, members in clusters.items():
        # Print clusters with different colors
        colors = ['\033[91m', '\033[92m', '\033[94m', '\033[93m']  # red, green, blue, yellow
        reset = '\033[0m'
        color = colors[c % len(colors)]
        print(f"{color}Cluster {c}: {members}{reset}")

    # Plot the clusters
    # Assign a color to each cluster for plotting
    plot_colors = ['red', 'green', 'blue', 'yellow']
    node_colors = [plot_colors[labels[nodes.index(n)] % len(plot_colors)] for n in G.nodes()]

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300)
    nx.draw_networkx_edges(G, pos, alpha=0.5)

    # Draw edge labels (weights)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=8)

    nx.draw_networkx_labels(G, pos, font_color='white')
    plt.title("Node Clusters by Degree (Edge Weights Shown)")
    plt.axis('off')
    plt.show()
    plt.pause(2)  # Pause to allow the plot to render before closing
    plt.close()  # Close the plot to avoid display issues in some environments
i = 0
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Take one snapshot
    graph = exp2.DynamicGraphSimulator()
    G = graph.get_graph_snapshot()

    # Compute shortest path from node 5 to node 18 using Dijkstra's algorithm (hash-based shortest path)
    try:
        shortest_path = nx.shortest_path(G, source=5, target=18, weight='weight')
        print(f"Shortest path from 5 to 18: {shortest_path}")
    except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
        print(f"No path found or node missing: {e}")
        shortest_path = []

    # Plot the graph and highlight the shortest path
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))

    # Draw all nodes and edges
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=300)
    nx.draw_networkx_edges(G, pos, alpha=0.5)

    # Highlight the shortest path
    if shortest_path and len(shortest_path) > 1:
        # Edges in the shortest path
        path_edges = list(zip(shortest_path[:-1], shortest_path[1:]))
        nx.draw_networkx_nodes(G, pos, nodelist=shortest_path, node_color='orange', node_size=400)
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='orange', width=3)

    # Draw edge labels (weights)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=8)

    nx.draw_networkx_labels(G, pos, font_color='black')
    plt.title("Shortest Path from 5 to 18 Highlighted")
    plt.axis('off')
    plt.show()
    plt.pause(10)  # Pause to allow the plot to render before closing