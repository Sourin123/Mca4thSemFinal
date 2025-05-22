import networkx as nx
import random
import time
import matplotlib.pyplot as plt

def generate_connected_edges(num_nodes):
    """Generate a connected graph (spanning tree) and add extra random edges."""
    edges = []
    nodes = list(range(num_nodes))
    random.shuffle(nodes)
    # Create a random spanning tree first
    for i in range(1, num_nodes):
        u = nodes[i]
        v = random.choice(nodes[:i])
        weight = round(random.uniform(1, 10), 2)
        edges.append((u, v, weight))
    # Add some extra random edges
    extra_edges = random.randint(num_nodes, num_nodes * 2)
    for _ in range(extra_edges):
        u, v = random.sample(range(num_nodes), 2)
        if (u, v) not in [(e[0], e[1]) for e in edges] and (v, u) not in [(e[0], e[1]) for e in edges]:
            weight = round(random.uniform(1, 10), 2)
            edges.append((u, v, weight))
    return edges

def draw_graph(graph, pos):
    plt.clf()
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.pause(0.1)
    # Randomly update weights of 3 to 4 edges not involved in creation/deletion
    all_edges = list(graph.edges())
    if len(all_edges) >= 4:
        num_updates = random.randint(3, 4)
        edges_to_update = random.sample(all_edges, num_updates)
        for u, v in edges_to_update:
            new_weight = round(random.uniform(1, 10), 2)
            graph[u][v]['weight'] = new_weight

def main():
    max_nodes = 15
    num_nodes = random.randint(6, max_nodes)
    is_directed = random.choice([True, False])
    G = nx.DiGraph() if is_directed else nx.Graph()
    G.add_nodes_from(range(num_nodes))
    pos = nx.spring_layout(G, seed=42)

    # Initialize with a connected set of edges
    edges = generate_connected_edges(num_nodes)
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    plt.ion()
    while True:
        num_updates = random.randint(2, 3)
        all_edges = list(G.edges())
        for _ in range(num_updates):
            if all_edges and random.random() < 0.5:
                # Try to remove a random edge, but keep the graph connected
                edge_to_remove = random.choice(all_edges)
                # Store the weight before removing the edge
                edge_weight = G.edges[edge_to_remove]['weight']
                G.remove_edge(*edge_to_remove)
                if (is_directed and not nx.is_weakly_connected(G)) or (not is_directed and not nx.is_connected(G)) or any(G.degree(n) == 0 for n in G.nodes()):
                    G.add_edge(*edge_to_remove, weight=edge_weight)  # Revert removal
                else:
                    all_edges.remove(edge_to_remove)
            else:
                # Add a new random edge
                u, v = random.sample(range(num_nodes), 2)
                if not G.has_edge(u, v):
                    weight = round(random.uniform(1, 10), 2)
                    G.add_edge(u, v, weight=weight)
        draw_graph(G, pos)
        time.sleep(5)

if __name__ == "__main__":
    main()