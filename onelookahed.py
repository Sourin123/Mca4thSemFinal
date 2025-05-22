import matplotlib.pyplot as plt
steps = list(range(1, 11))
dijkstra_nodes = [102, 104, 100, 103, 105, 101, 99, 100, 102, 104]
modularity_nodes = [78, 80, 76, 79, 81, 77, 75, 76, 78, 80]
labelprop_nodes = [55, 56, 54, 55, 57, 53, 52, 54, 55, 56]
dynamic_nodes = [41, 42, 41, 40, 41, 39, 38, 40, 41, 42]

plt.plot(steps, dijkstra_nodes, label='Dijkstra')
plt.plot(steps, modularity_nodes, label='Static Modularity')
plt.plot(steps, labelprop_nodes, label='Label Propagation')
plt.plot(steps, dynamic_nodes, label='Dynamic Cluster-Based')
plt.xlabel('Simulation Step')
plt.ylabel('Nodes Visited')
plt.title('Nodes Visited per Query: Model Comparison')
plt.legend()
plt.show()
