import torch
import numpy as np
from starry_net import StarryNet

def plot_network():
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError:
        print("Please install networkx and matplotlib to run the visualizer:")
        print("pip install networkx matplotlib")
        return

    # Initialize model to match dimensions
    num_neurons = 32
    net = StarryNet(num_neurons=num_neurons, input_indices=[0, 1], output_indices=[-1, -2])
    
    # Load trained state dict
    try:
        net.load_state_dict(torch.load('starry_net.pth'))
        print("Loaded trained model from 'starry_net.pth'")
    except FileNotFoundError:
        print("Could not find 'starry_net.pth'. Plotting initial random/untrained network.")
    
    mask = net.mask.cpu().numpy()
    weights = (net.W_dense * net.mask).detach().cpu().numpy()
    
    # Create networkx graph
    # Weight matrices in RNNs represent directed connections from j to i
    # (i.e. input state @ W.T means state_t = W @ state_{t-1}, so connection is j -> i)
    G = nx.DiGraph()
    
    # Add nodes
    for i in range(num_neurons):
        G.add_node(i)
        
    # Add edges
    for i in range(num_neurons):
        for j in range(num_neurons):
            if mask[i, j] > 0.5:
                G.add_edge(j, i, weight=weights[i, j])
                
    # Layout: arrange nodes in a circle to mimic a 2D projection of a sphere
    pos = nx.circular_layout(G)
    
    # Define colors for different types of nodes
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node == 0:
            node_colors.append('#2ecc71')  # Light green for Task 1 Input
            node_sizes.append(400)
        elif node == 1:
            node_colors.append('#27ae60')  # Dark green for Task 2 Input
            node_sizes.append(400)
        elif node == num_neurons - 1:
            node_colors.append('#e67e22')  # Light orange for Task 1 Output
            node_sizes.append(400)
        elif node == num_neurons - 2:
            node_colors.append('#d35400')  # Dark orange for Task 2 Output
            node_sizes.append(400)
        else:
            node_colors.append('#3498db')  # Blue for hidden neurons
            node_sizes.append(150)
            
    # Set up matplotlib figure
    plt.figure(figsize=(12, 12), facecolor='#111111')
    ax = plt.gca()
    ax.set_facecolor('#111111')
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, 
        node_color=node_colors, 
        node_size=node_sizes, 
        alpha=0.9,
        edgecolors='#ffffff',
        linewidths=0.5
    )
    
    # Separate positive and negative edges for styling
    edges_pos = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > 0]
    edges_neg = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] <= 0]
    
    # Normalize weights for alpha/width mapping
    all_weights = [d['weight'] for u, v, d in G.edges(data=True)]
    if all_weights:
        max_w = max(max(np.abs(all_weights)), 1e-5)
    else:
        max_w = 1.0
        
    # Draw positive edges in cyan
    nx.draw_networkx_edges(
        G, pos, 
        edgelist=edges_pos,
        edge_color='#00ffcc',
        width=[np.abs(G[u][v]['weight']) / max_w * 2.0 for u, v in edges_pos],
        alpha=0.4,
        arrows=True,
        arrowsize=10,
        connectionstyle="arc3,rad=0.1"  # curved edges to distinguish bi-directional
    )
    
    # Draw negative edges in magenta/pink
    nx.draw_networkx_edges(
        G, pos, 
        edgelist=edges_neg,
        edge_color='#ff007f',
        width=[np.abs(G[u][v]['weight']) / max_w * 2.0 for u, v in edges_neg],
        alpha=0.4,
        arrows=True,
        arrowsize=10,
        connectionstyle="arc3,rad=0.1"
    )
    
    # Add labels for IO nodes
    labels = {
        0: 'In-Add',
        1: 'In-Par',
        num_neurons - 1: 'Out-Add',
        num_neurons - 2: 'Out-Par'
    }
    nx.draw_networkx_labels(
        G, pos, 
        labels=labels, 
        font_size=10, 
        font_color='#ffffff', 
        font_family='sans-serif',
        font_weight='bold'
    )
    
    density = mask.sum() / (num_neurons**2)
    plt.title(
        f"StarryNN Trained Topology\nNeurons: {num_neurons} | Active Connections: {int(mask.sum())} ({density:.1%})", 
        color='#ffffff', 
        fontsize=16, 
        fontweight='bold',
        pad=20
    )
    
    # Save figure
    filename = 'starry_net_topology.png'
    plt.savefig(filename, facecolor='#111111', edgecolor='none', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Successfully generated and saved network topology plot to '{filename}'")

if __name__ == "__main__":
    plot_network()
