import networkx as nx
import matplotlib.pyplot as plt

# Define the relationships
relationships = {
    'dist_calc': ['dist_combiner', 'plot_bin'],
    'get_psi': ['get_psi_vif'],
    'get_vif': ['get_psi_vif'],
    'dist_combiner': ['bins2mono', 'monolize'],
    'bins2mono': ['monolize'],
    'bin_set': ['monolize'],
    'monolize': ['discrete_type'],
    'plot_bin': ['bin_update_show_plt'],
    'val_describe': ['val_describe_tot'],
    'model_verify': ['rst_print'],
    'plot_lift': ['plot_all'],
    'plot_ks': ['plot_all'],
    'cross_calc': ['plot_badrate', 'cross_vars'],
    'get_psi_vif': [],
    'discrete_type': [],
    'bin_update_show_plt': [],
    'sample_select': [],
    'rst_print': [],
    'plot_all': [],
    'plot_badrate': [],
    'cross_vars': []
}

# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges
for func, used_by in relationships.items():
    for user in used_by:
        G.add_edge(func, user)

# Draw the graph
pos = nx.spring_layout(G)
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', arrowsize=20)
plt.title('Function Relationship Graph')
plt.show()