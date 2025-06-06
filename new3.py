import networkx as nx
import random
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

# Parameters
num_nodes = 120
initial_edges = 225
steps = 50

# Create initial connected graph
G = nx.gnm_random_graph(num_nodes, initial_edges)
while not nx.is_connected(G):
    G = nx.gnm_random_graph(num_nodes, initial_edges)
for u, v in G.edges():
    G[u][v]['weight'] = random.randint(1, 10)

# Fix node positions
pos = nx.spring_layout(G, seed=42)

def ensure_connected(G):
    # Remove isolated nodes
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    # If disconnected, connect components
    while not nx.is_connected(G) and G.number_of_nodes() > 1:
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            u = random.choice(list(components[i]))
            v = random.choice(list(components[i + 1]))
            G.add_edge(u, v, weight=random.randint(1, 10))

def update_graph(G):
    action = random.choices(
        ['weight_update', 'edge_change', 'node_change'],
        weights=[0.4, 0.3, 0.25],
        k=1
    )[0]

    if action == 'weight_update' and G.number_of_edges() > 0:
        # Update weight of a random edge
        u, v = random.choice(list(G.edges()))
        G[u][v]['weight'] = random.randint(1, 10)
    elif action == 'edge_change':
        if random.random() < 0.5 and G.number_of_edges() > 0:
            # Remove a random edge
            G.remove_edge(*random.choice(list(G.edges())))
        else:
            # Add a random edge
            possible = list(nx.non_edges(G))
            if possible:
                u, v = random.choice(possible)
                G.add_edge(u, v, weight=random.randint(1, 10))
    elif action == 'node_change':
        if random.random() < 0.5 and G.number_of_nodes() > 1:
            # Remove a random node
            G.remove_node(random.choice(list(G.nodes())))
        else:
            # Add a new node
            new_node = max(G.nodes(), default=-1) + 1
            G.add_node(new_node)
            # Optionally connect it to existing nodes
            targets = random.sample(list(G.nodes()), k=min(2, G.number_of_nodes()-1))
            for t in targets:
                if t != new_node:
                    G.add_edge(new_node, t, weight=random.randint(1, 10))
            # Add position for new node
            pos[new_node] = [random.uniform(-1, 1), random.uniform(-1, 1)]
    ensure_connected(G)

fig, ax = plt.subplots(figsize=(8, 6))

def animate(i):
    ax.clear()
    update_graph(G)
    # Remove positions for deleted nodes
    for node in list(pos.keys()):
        if node not in G.nodes():
            del pos[node]
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='skyblue', edge_color='gray')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    ax.set_title(f"Step {i+1}")

ani = FuncAnimation(fig, animate, frames=steps, interval=800, repeat=False)
plt.show()