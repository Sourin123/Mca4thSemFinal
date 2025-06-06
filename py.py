import matplotlib.pyplot as plt

# Sample data
methods = ['LEACH/HEED', 'Our Method']
stability = [50, 89]  # 50 = average of 40–60%

plt.bar(methods, stability, color=['gray', 'green'])
plt.ylim(0, 100)
plt.ylabel('Cluster Stability (%)')
plt.title('Figure 5: Stability During Topology Changes')
plt.show()
