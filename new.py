import time
import exp2
import networkx as nx
import numpy as np
from community import community_louvain
import matplotlib.pyplot as plt

def cluster_graph_louvain(G):
    """Cluster the graph nodes using the Louvain algorithm."""
    # Compute the best partition
    partition = community_louvain.best_partition(G)
    return partition

def find_shortest_path_with_clustering(G, source, target, cluster_map):
    """Find the shortest path between source and target nodes using clustering."""
    source_cluster = cluster_map[source]
    target_cluster = cluster_map[target]

    if source_cluster == target_cluster:
        # Source and target are in the same cluster
        return nx.shortest_path(G, source=source, target=target, weight='weight')
    
    # Inter-cluster shortest path computation
    # Find cluster boundary nodes
    cluster_nodes = {c: [n for n, cluster in cluster_map.items() if cluster == c] for c in set(cluster_map.values())}

    # Create a reduced graph where clusters are nodes
    reduced_graph = nx.Graph()

    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1)
        cluster_u = cluster_map[u]
        cluster_v = cluster_map[v]
        if cluster_u != cluster_v:
            if reduced_graph.has_edge(cluster_u, cluster_v):
                reduced_graph[cluster_u][cluster_v]['weight'] = min(reduced_graph[cluster_u][cluster_v]['weight'], weight)
            else:
                reduced_graph.add_edge(cluster_u, cluster_v, weight=weight)

    # Shortest path between clusters
    cluster_path = nx.shortest_path(reduced_graph, source=source_cluster, target=target_cluster, weight='weight')

    # Combine intra-cluster paths
    full_path = []
    current_node = source

    for i in range(len(cluster_path) - 1):
        current_cluster = cluster_path[i]
        next_cluster = cluster_path[i + 1]

        # Find boundary nodes
        current_cluster_nodes = cluster_nodes[current_cluster]
        next_cluster_nodes = cluster_nodes[next_cluster]

        # Find shortest path between boundary nodes
        boundary_path = None
        min_distance = float('inf')

        for cn in current_cluster_nodes:
            for nn in next_cluster_nodes:
                if G.has_edge(cn, nn):
                    distance = G[cn][nn]['weight']
                    if distance < min_distance:
                        min_distance = distance
                        boundary_path = [cn, nn]

        if boundary_path:
            if not full_path:
                full_path.extend(nx.shortest_path(G, source=current_node, target=boundary_path[0], weight='weight'))
            else:
                full_path.extend(nx.shortest_path(G, source=full_path[-1], target=boundary_path[0], weight='weight')[1:])

            full_path.append(boundary_path[1])
            current_node = boundary_path[1]

    # Add the path from the last boundary node to the target
    full_path.extend(nx.shortest_path(G, source=current_node, target=target, weight='weight')[1:])

    return full_path

def cluster_and_plot_graph():
    graph = exp2.DynamicGraphSimulator()
    # Get the current snapshot of the graph
    G = graph.get_graph_snapshot()

    # Cluster the graph using Louvain algorithm
    cluster_map = cluster_graph_louvain(G)

    # Plot the clustered graph
    pos = nx.spring_layout(G, seed=42)
    
    cluster_colors = ['red', 'green', 'blue', 'yellow']
    node_colors = [cluster_colors[cluster_map[n] % len(cluster_colors)] for n in G.nodes()]

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_color='black')

    # plt.title("Clustered Graph (Louvain Algorithm)")
    plt.axis('off')
    plt.show()

    return G, cluster_map

if __name__ == "__main__":
    # Cluster the graph and retrieve clustering information
    G, cluster_map = cluster_and_plot_graph()

    # Compute the shortest path between two nodes using clustering
    source = 5
    target = 18
    try:
        # Path using clustering-based algorithm
        cluster_path = find_shortest_path_with_clustering(G, source, target, cluster_map)
        print(f"Clustering-based shortest path from {source} to {target}: {cluster_path}")

        # Path using conventional Dijkstra's algorithm
        dijkstra_path = nx.shortest_path(G, source=source, target=target, weight='weight')
        print(f"Dijkstra's shortest path from {source} to {target}: {dijkstra_path}")

        # Visualization
        pos = nx.spring_layout(G, seed=42)
        cluster_colors = ['red', 'green', 'blue', 'yellow']
        node_colors = [cluster_colors[cluster_map[n] % len(cluster_colors)] for n in G.nodes()]

        plt.figure(figsize=(10, 8))
        # nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300)
        nx.draw_networkx_edges(G, pos, alpha=0.3)
        nx.draw_networkx_labels(G, pos, font_color='black')

        # Draw clustering-based path in blue
        path_edges = list(zip(cluster_path, cluster_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='blue', width=3, label='Clustering Path')

        # Draw Dijkstra path in orange (if different)
        if dijkstra_path != cluster_path:
            dijkstra_edges = list(zip(dijkstra_path, dijkstra_path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=dijkstra_edges, edge_color='orange', width=3, style='dashed', label="Dijkstra's Path")

        plt.title("Routing Paths: Clustering vs Dijkstra's Algorithm")
        plt.axis('off')
        plt.legend(loc='lower left')
        plt.show()

    except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
        print(f"No path found or node missing: {e}")
