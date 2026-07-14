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
    num_neurons = 64
    net = StarryNet(
        num_neurons=num_neurons, 
        input_indices=list(range(13)), 
        output_indices=[-3, -2, -1]  # 3 class exits
    )
    
    # Load trained state dict
    try:
        net.load_state_dict(torch.load('models/starry_net.pth'))
        print("Loaded trained model from 'models/starry_net.pth'")
    except FileNotFoundError:
        print("Could not find 'models/starry_net.pth'. Plotting initial random/untrained network.")
    
    mask = net.mask.cpu().numpy()
    weights = (net.W_dense * net.mask).detach().cpu().numpy()
    
    # Create networkx graph
    G = nx.DiGraph()
    
    # Add nodes
    for i in range(num_neurons):
        G.add_node(i)
        
    # Add edges
    for i in range(num_neurons):
        for j in range(num_neurons):
            if mask[i, j] > 0.5:
                G.add_edge(j, i, weight=weights[i, j])
                
    # Define functional brain modules sizes
    num_sensory = 30
    num_assoc = 30
    num_motor = 4
    
    # Create custom layout positioning by functional region
    pos = {}
    
    # 1. Sensory (left vertical arch)
    for idx in range(num_sensory):
        theta = -np.pi/2 + np.pi * (idx / max(num_sensory - 1, 1))
        pos[idx] = np.array([0.15 - 0.05 * np.cos(theta), 0.5 + 0.4 * np.sin(theta)])
        
    # 2. Association (central processing circle)
    for idx in range(num_sensory, num_sensory + num_assoc):
        assoc_idx = idx - num_sensory
        phi = 2 * np.pi * (assoc_idx / max(num_assoc, 1))
        pos[idx] = np.array([0.5 + 0.16 * np.cos(phi), 0.5 + 0.16 * np.sin(phi)])
        
    # 3. Motor (right vertical column)
    for idx in range(num_sensory + num_assoc, num_neurons):
        motor_idx = idx - (num_sensory + num_assoc)
        pos[idx] = np.array([0.85, 0.3 + 0.4 * (motor_idx / max(num_motor - 1, 1))])
    
    # Define colors for different types of nodes
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node < 13:
            node_colors.append('#2ecc71')  # Active Input features (light green)
            node_sizes.append(180)
        elif node < num_sensory:
            node_colors.append('#27ae60')  # Hidden Sensory nodes (dark green)
            node_sizes.append(100)
        elif node >= num_neurons - 3:
            node_colors.append('#e74c3c')  # Red for the 3 Class Exit Nodes (61, 62, 63)
            node_sizes.append(350)
        elif node >= num_sensory + num_assoc:
            node_colors.append('#e67e22')  # Orange for the rest of Motor module
            node_sizes.append(200)
        else:
            node_colors.append('#3498db')  # Blue for Association Hub
            node_sizes.append(100)
            
    # Set up matplotlib figure
    plt.figure(figsize=(14, 10), facecolor='#111111')
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
        width=[np.abs(G[u][v]['weight']) / max_w * 1.5 for u, v in edges_pos],
        alpha=0.25,
        arrows=True,
        arrowsize=7,
        connectionstyle="arc3,rad=0.03"
    )
    
    # Draw negative edges in magenta/pink
    nx.draw_networkx_edges(
        G, pos, 
        edgelist=edges_neg,
        edge_color='#ff007f',
        width=[np.abs(G[u][v]['weight']) / max_w * 1.5 for u, v in edges_neg],
        alpha=0.25,
        arrows=True,
        arrowsize=7,
        connectionstyle="arc3,rad=0.03"
    )
    
    # Add label only for the exit nodes
    labels = {
        61: 'Class 0',
        62: 'Class 1',
        63: 'Class 2'
    }
    nx.draw_networkx_labels(
        G, pos, 
        labels=labels, 
        font_size=9, 
        font_color='#ffffff', 
        font_family='sans-serif',
        font_weight='bold'
    )
    
    # Add titles for the visual modules
    plt.text(0.12, 0.95, "SENSORY MODULE\n(Inputs 0-12 active)", color='#2ecc71', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
    plt.text(0.5, 0.95, "ASSOCIATION HUB\n(Cognitive Core)", color='#3498db', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
    plt.text(0.88, 0.95, "MOTOR MODULE\n(3 Class Outputs)", color='#e67e22', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
    
    density = mask.sum() / (num_neurons**2)
    prior_density = net.prior_mask.sum().item() / (num_neurons**2)
    plt.title(
        f"StarryNN Modular Hebbian Multi-Class Topology (Wine Dataset)\nNeurons: {num_neurons} | Allowed Wiring Density: {prior_density:.1%} | Active Pruned Density: {density:.1%}", 
        color='#ffffff', 
        fontsize=16, 
        fontweight='bold',
        pad=25
    )
    
    # Save figure
    filename = 'plots/starry_net_topology.png'
    plt.savefig(filename, facecolor='#111111', edgecolor='none', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Successfully generated and saved network topology plot to '{filename}'")

if __name__ == "__main__":
    plot_network()
