import random
import networkx as nx


class DynamicGraphSimulator:
    DURATION = 100  # seconds
    INIT_NODES = 130
    MIN_NODES = 123
    MAX_NODES = 137
    EDGE_UPDATE_PROB = 0.15  # 15% chance to update 1 or 2 edges
    EDGE_WEIGHT_UPDATE_PROB = 0.35  # 35% chance to update edge weight
    NODE_APPEAR_PROB = 0.15
    NODE_DISAPPEAR_PROB = 0.15
    EDGE_WEIGHT_MIN = 10
    EDGE_WEIGHT_MAX = 99

    def __init__(self):
        self.G = nx.Graph()
        self.next_node_id = self.INIT_NODES

    def ensure_connected(self):
        while not nx.is_connected(self.G):
            components = list(nx.connected_components(self.G))
            u = random.choice(list(components[0]))
            v = random.choice(list(components[1]))
            self.G.add_edge(u, v, weight=random.randint(self.EDGE_WEIGHT_MIN, self.EDGE_WEIGHT_MAX))

    def random_edge(self):
        nodes = list(self.G.nodes)
        u, v = random.sample(nodes, 2)
        return (u, v)

    def add_random_edge(self):
        nodes = list(self.G.nodes)
        possible = [(u, v) for u in nodes for v in nodes if u != v and not self.G.has_edge(u, v)]
        if possible:
            u, v = random.choice(possible)
            self.G.add_edge(u, v, weight=random.randint(self.EDGE_WEIGHT_MIN, self.EDGE_WEIGHT_MAX))

    def remove_random_edge(self):
        if self.G.number_of_edges() > self.G.number_of_nodes() - 1:
            edge = random.choice(list(self.G.edges))
            weight = self.G.edges[edge]['weight']
            self.G.remove_edge(*edge)
            if not nx.is_connected(self.G):
                self.G.add_edge(*edge, weight=weight)  # revert if disconnects

    def add_node(self):
        self.G.add_node(self.next_node_id)
        existing_nodes = set(self.G.nodes) - {self.next_node_id}
        existing = random.choice(list(existing_nodes))
        self.G.add_edge(self.next_node_id, existing, weight=random.randint(self.EDGE_WEIGHT_MIN, self.EDGE_WEIGHT_MAX))
        self.next_node_id += 1

    def remove_node(self):
        if self.G.number_of_nodes() > self.MIN_NODES:
            node = random.choice(list(self.G.nodes))
            edges = list(self.G.edges(node, data=True))
            self.G.remove_node(node)
            if not nx.is_connected(self.G):
                self.G.add_node(node)
                existing_nodes = set(self.G.nodes) - {node}
                existing = random.choice(list(existing_nodes))
                self.G.add_edge(node, existing, weight=random.randint(self.EDGE_WEIGHT_MIN, self.EDGE_WEIGHT_MAX))

    def update_random_edge_weight(self):
        if self.G.number_of_edges() > 0:
            edge = random.choice(list(self.G.edges))
            self.G.edges[edge]['weight'] = random.randint(self.EDGE_WEIGHT_MIN, self.EDGE_WEIGHT_MAX)

    def get_graph_snapshot(self):
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.INIT_NODES))
        for _ in range(self.INIT_NODES * 2):
            self.add_random_edge()
        self.ensure_connected()
        self.next_node_id = self.INIT_NODES

        for _ in range(5):
            if random.random() < self.NODE_APPEAR_PROB and self.G.number_of_nodes() < self.MAX_NODES:
                self.add_node()
            elif random.random() < self.NODE_DISAPPEAR_PROB and self.G.number_of_nodes() > self.MIN_NODES:
                self.remove_node()

            num_edge_updates = 1 if random.random() < self.EDGE_UPDATE_PROB else 2
            for _ in range(num_edge_updates):
                if random.random() < 0.5:
                    self.add_random_edge()
                else:
                    self.remove_random_edge()

            # Edge weight update
            if random.random() < self.EDGE_WEIGHT_UPDATE_PROB:
                self.update_random_edge_weight()

            self.ensure_connected()
        return self.G