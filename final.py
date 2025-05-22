import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import random

# -----------------------------
# Dynamic Graph Simulator
# -----------------------------
class DynamicGraphSimulator:
    def __init__(self, n_nodes=30, edge_prob=0.08):
        self.n_nodes = n_nodes
        self.edge_prob = edge_prob
        self.G = nx.erdos_renyi_graph(n_nodes, edge_prob)
        self.pos = nx.spring_layout(self.G, seed=42)
    
    def step(self):
        # Randomly add or remove edges to simulate dynamics
        for _ in range(random.randint(1, 5)):
            u, v = random.sample(range(self.n_nodes), 2)
            if self.G.has_edge(u, v):
                self.G.remove_edge(u, v)
            else:
                self.G.add_edge(u, v)
        # Optionally, add/remove nodes for more dynamics
    
    def draw(self, clusters=None, title="Dynamic Graph"):
        plt.figure(figsize=(8, 6))
        if clusters is not None:
            colors = [clusters.get(node, -1) for node in self.G.nodes()]
            nx.draw(self.G, self.pos, node_color=colors, with_labels=True, cmap=plt.cm.Set1)
        else:
            nx.draw(self.G, self.pos, with_labels=True)
        plt.title(title)
        plt.show()

# -----------------------------
# Clustering (DBSCAN on Node Features)
# -----------------------------
def graph_to_features(G):
    degrees = np.array([G.degree(n) for n in G.nodes()])
    clustering = np.array([nx.clustering(G, n) for n in G.nodes()])
    features = np.vstack((degrees, clustering)).T
    return StandardScaler().fit_transform(features)

def cluster_graph(G, eps=1.0, min_samples=2):
    features = graph_to_features(G)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(features)
    clusters = {node: int(label) for node, label in zip(G.nodes(), labels)}
    return clusters

# -----------------------------
# Meta-Graph Construction
# -----------------------------
def build_meta_graph(G, clusters):
    meta_graph = nx.Graph()
    cluster_nodes = set(clusters.values())
    cluster_nodes.discard(-1)  # Remove noise/unclustered
    for c in cluster_nodes:
        meta_graph.add_node(c)
    # Add edges between clusters if any node in cluster A connects to cluster B
    for u, v in G.edges():
        cu, cv = clusters[u], clusters[v]
        if cu != cv and cu != -1 and cv != -1:
            meta_graph.add_edge(cu, cv)
    return meta_graph

def are_clusters_adjacent(clusters, G, src_cluster, tgt_cluster):
    # Check if there is any edge between nodes in src_cluster and tgt_cluster
    for u, v in G.edges():
        if clusters[u] == src_cluster and clusters[v] == tgt_cluster:
            return True
        if clusters[v] == src_cluster and clusters[u] == tgt_cluster:
            return True
    return False

def get_gateway_pairs(G, clusters, src_cluster, tgt_cluster):
    gateways_src = [n for n in G.nodes() if clusters[n] == src_cluster and
                    any(clusters[nb] == tgt_cluster for nb in G.neighbors(n))]
    gateways_tgt = [n for n in G.nodes() if clusters[n] == tgt_cluster and
                    any(clusters[nb] == src_cluster for nb in G.neighbors(n))]
    return gateways_src, gateways_tgt

# -----------------------------
# Cluster-Aware and Meta-Cluster Routing
# -----------------------------
def find_cluster_path(meta_graph, src_cluster, tgt_cluster):
    try:
        return nx.shortest_path(meta_graph, src_cluster, tgt_cluster)
    except nx.NetworkXNoPath:
        return None

def find_path(G, clusters, source, target):
    src_cluster = clusters[source]
    tgt_cluster = clusters[target]
    if src_cluster == -1 or tgt_cluster == -1:
        print("Source or target is noise/unclustered.")
        return None, "Unclustered"
    if src_cluster == tgt_cluster:
        # Intra-cluster path
        try:
            return nx.shortest_path(G, source, target), "Intra-cluster"
        except nx.NetworkXNoPath:
            return None, "No path"
    elif are_clusters_adjacent(clusters, G, src_cluster, tgt_cluster):
        # Adjacent-cluster routing via gateways
        gateways_src, gateways_tgt = get_gateway_pairs(G, clusters, src_cluster, tgt_cluster)
        min_path = None
        min_len = float('inf')
        for g_src in gateways_src:
            for g_tgt in gateways_tgt:
                try:
                    path1 = nx.shortest_path(G, source, g_src)
                    path2 = nx.shortest_path(G, g_src, g_tgt)
                    path3 = nx.shortest_path(G, g_tgt, target)
                    full_path = path1[:-1] + path2[:-1] + path3
                    if len(full_path) < min_len:
                        min_len = len(full_path)
                        min_path = full_path
                except nx.NetworkXNoPath:
                    continue
        return min_path, "Adjacent-cluster"
    else:
        # Meta-cluster routing
        meta_graph = build_meta_graph(G, clusters)
        cluster_path = find_cluster_path(meta_graph, src_cluster, tgt_cluster)
        if not cluster_path:
            return None, "No meta-cluster path"
        # Find gateway nodes between clusters along cluster_path
        full_path = [source]
        current = source
        for i in range(len(cluster_path)-1):
            c_from, c_to = cluster_path[i], cluster_path[i+1]
            gateways_src, gateways_tgt = get_gateway_pairs(G, clusters, c_from, c_to)
            # Find the closest gateway from current node
            min_subpath = None
            min_len = float('inf')
            for g_src in gateways_src:
                try:
                    subpath = nx.shortest_path(G, current, g_src)
                    if len(subpath) < min_len:
                        min_len = len(subpath)
                        min_subpath = subpath
                except nx.NetworkXNoPath:
                    continue
            if min_subpath is None:
                return None, "No gateway path"
            full_path += min_subpath[1:]
            # Move to gateway in next cluster
            # Pick any gateway in next cluster (could optimize further)
            next_gateway = gateways_tgt[0] if gateways_tgt else None
            if next_gateway:
                full_path.append(next_gateway)
                current = next_gateway
            else:
                return None, "No gateway path"
        # Final hop to target
        try:
            subpath = nx.shortest_path(G, current, target)
            full_path += subpath[1:]
        except nx.NetworkXNoPath:
            return None, "No path to target"
        return full_path, "Meta-cluster"
    
# -----------------------------
# Main Simulation Loop
# -----------------------------
if __name__ == "__main__":
    sim = DynamicGraphSimulator(n_nodes=30, edge_prob=0.08)
    for step in range(10):
        print(f"Step {step+1}:")
        sim.step()
        clusters = cluster_graph(sim.G, eps=1.0, min_samples=2)
        sim.draw(clusters, title=f"Dynamic Graph at Step {step+1}")
        
        # Example path search between two random nodes
        nodes = list(sim.G.nodes())
        src, tgt = random.sample(nodes, 2)
        path, mode = find_path(sim.G, clusters, src, tgt)
        print(f"Path from {src} to {tgt} ({mode}): {path}")
        print("-" * 40)
