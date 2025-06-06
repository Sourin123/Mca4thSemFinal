import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

# -----------------------------
# Dynamic Graph Simulator
# -----------------------------
class DynamicGraphSimulator:
    def _init_(self, n_nodes=30, edge_prob=0.08):
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
# Seed-Oriented Clustering
# -----------------------------
def seed_oriented_clustering(G, cluster_ratio=0.2, edge_cost_limit=None):
    n = G.number_of_nodes()
    num_clusters = max(1, int(cluster_ratio * n))
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
    seeds = sorted_nodes[:num_clusters]
    assigned = set()
    clusters = {}
    cluster_id = 0

    for seed in seeds:
        if seed in assigned:
            continue
        cluster_nodes = {seed}
        frontier = set(G.neighbors(seed)) - assigned
        total_cost = 0
        while frontier:
            min_cost = float('inf')
            min_node = None
            for node in frontier:
                cost = sum(G[u][node].get('weight', 1) for u in cluster_nodes if G.has_edge(u, node))
                if cost < min_cost:
                    min_cost = cost
                    min_node = node
            if min_node is None:
                break
            if edge_cost_limit is not None and (total_cost + min_cost > edge_cost_limit):
                break
            cluster_nodes.add(min_node)
            assigned.add(min_node)
            total_cost += min_cost
            frontier.update(set(G.neighbors(min_node)) - assigned - cluster_nodes)
        for node in cluster_nodes:
            clusters[node] = cluster_id
        assigned.update(cluster_nodes)
        cluster_id += 1

    # Assign any unassigned nodes to the nearest seed's cluster
    unassigned = set(G.nodes()) - assigned
    for node in unassigned:
        min_seed = min(seeds, key=lambda s: nx.shortest_path_length(G, source=node, target=s))
        for n, cid in clusters.items():
            if n == min_seed:
                clusters[node] = cid
                break
    return clusters

# -----------------------------
# Meta-Graph Construction
# -----------------------------
def build_meta_graph(G, clusters):
    meta_graph = nx.Graph()
    cluster_nodes = set(clusters.values())
    for c in cluster_nodes:
        meta_graph.add_node(c)
    for u, v in G.edges():
        cu, cv = clusters[u], clusters[v]
        if cu != cv:
            meta_graph.add_edge(cu, cv)
    return meta_graph

def are_clusters_adjacent(clusters, G, src_cluster, tgt_cluster):
    for u, v in G.edges():
        if (clusters[u] == src_cluster and clusters[v] == tgt_cluster) or \
           (clusters[v] == src_cluster and clusters[u] == tgt_cluster):
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
    if src_cluster == tgt_cluster:
        try:
            return nx.shortest_path(G, source, target), "Intra-cluster"
        except nx.NetworkXNoPath:
            return None, "No path"
    elif are_clusters_adjacent(clusters, G, src_cluster, tgt_cluster):
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
        meta_graph = build_meta_graph(G, clusters)
        cluster_path = find_cluster_path(meta_graph, src_cluster, tgt_cluster)
        if not cluster_path:
            return None, "No meta-cluster path"
        full_path = [source]
        current = source
        for i in range(len(cluster_path)-1):
            c_from, c_to = cluster_path[i], cluster_path[i+1]
            gateways_src, gateways_tgt = get_gateway_pairs(G, clusters, c_from, c_to)
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
            if min_subpath is None or not gateways_tgt:
                return None, "No gateway path"
            full_path += min_subpath[1:]
            next_gateway = gateways_tgt[0]
            full_path.append(next_gateway)
            current = next_gateway
        try:
            subpath = nx.shortest_path(G, current, target)
            full_path += subpath[1:]
        except nx.NetworkXNoPath:
            return None, "No path to target"
        return full_path, "Meta-cluster"

# -----------------------------
# Main Simulation Loop
# -----------------------------
if __name__ == "_main_":
    sim = DynamicGraphSimulator(n_nodes=30, edge_prob=0.08)
    for step in range(10):
        print(f"Step {step+1}:")
        sim.step()
        clusters = seed_oriented_clustering(sim.G, cluster_ratio=0.2, edge_cost_limit=None)
        sim.draw(clusters, title=f"Dynamic Graph at Step {step+1}")
        
        # Example path search between two random nodes
        nodes = list(sim.G.nodes())
        src, tgt = random.sample(nodes, 2)
        path, mode = find_path(sim.G, clusters, src, tgt)
        print(f"Path from {src} to {tgt} ({mode}): {path}")
        print("-" * 40)