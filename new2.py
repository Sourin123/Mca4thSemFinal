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


def path_cost(G, path):
    """Calculate the total cost (weight) of a path."""
    total = 0
    for u, v in zip(path[:-1], path[1:]):
        total += G[u][v].get('weight', 1)
    return total


def find_shortest_path_with_clustering(G, source, target, cluster_map):
    """Find the shortest path between source and target nodes using clustering."""
    source_cluster = cluster_map[source]
    target_cluster = cluster_map[target]

    if source_cluster == target_cluster:
        # Source and target are in the same cluster
        path = nx.shortest_path(G, source=source, target=target, weight='weight')
        cost = path_cost(G, path)
        return path, cost

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

    # Combine intra-cluster paths and accumulate cost
    full_path = []
    current_node = source
    total_cost = 0

    for i in range(len(cluster_path) - 1):
        current_cluster = cluster_path[i]
        next_cluster = cluster_path[i + 1]

        # Find boundary nodes between current and next cluster
        current_cluster_nodes = cluster_nodes[current_cluster]
        next_cluster_nodes = cluster_nodes[next_cluster]

        boundary_path = None
        min_distance = float('inf')

        # Find minimum weight edge connecting two clusters (boundary edge)
        for cn in current_cluster_nodes:
            for nn in next_cluster_nodes:
                if G.has_edge(cn, nn):
                    distance = G[cn][nn]['weight']
                    if distance < min_distance:
                        min_distance = distance
                        boundary_path = [cn, nn]

        if boundary_path is None:
            raise nx.NetworkXNoPath(f"No boundary edge between cluster {current_cluster} and {next_cluster}")

        # Add intra-cluster path from current_node to boundary node
        if not full_path:
            intra_path = nx.shortest_path(G, source=current_node, target=boundary_path[0], weight='weight')
            full_path.extend(intra_path)
        else:
            intra_path = nx.shortest_path(G, source=full_path[-1], target=boundary_path[0], weight='weight')[1:]
            full_path.extend(intra_path)

        # Add the inter-cluster edge
        full_path.append(boundary_path[1])

        # Update current node for next iteration
        current_node = boundary_path[1]

    # Add path from last boundary node to target
    intra_path = nx.shortest_path(G, source=current_node, target=target, weight='weight')[1:]
    full_path.extend(intra_path)

    # Calculate total cost of full path
    total_cost = path_cost(G, full_path)

    return full_path, total_cost


def cluster_and_plot_graph():
    graph = exp2.DynamicGraphSimulator()
    # Get the current snapshot of the graph
    G = graph.get_graph_snapshot()

    # Cluster the graph using Louvain algorithm
    cluster_map = cluster_graph_louvain(G)

    # Plot the clustered graph
    pos = nx.spring_layout(G, seed=42)

    cluster_colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan']
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
    G, cluster_map = cluster_and_plot_graph()

    source = 100
    target = 800

    # Cluster-based path
    try:
        cluster_path, cluster_cost = find_shortest_path_with_clustering(G, source, target, cluster_map)
        print(f"Cluster-based shortest path: {cluster_path}")
        print(f"Total cost (cluster-based): {cluster_cost:.4f}")
    except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
        print(f"Cluster-based path error: {e}")
        cluster_path, cluster_cost = None, None

    # Conventional Dijkstra shortest path
    try:
        dijkstra_path = nx.shortest_path(G, source=source, target=target, weight='weight')
        dijkstra_cost = path_cost(G, dijkstra_path)
        print(f"Dijkstra's shortest path: {dijkstra_path}")
        print(f"Total cost (Dijkstra): {dijkstra_cost:.4f}")
    except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
        print(f"Dijkstra's path error: {e}")
