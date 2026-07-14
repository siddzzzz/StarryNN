import torch
import numpy as np
from tasks import load_wine_data, get_tabular_batches
from starry_net import StarryNet

def train():
    # Hyperparameters
    num_neurons = 64
    steps = 8         # Relaxation steps for settling tabular activations
    batch_size = 32   # Smaller batch size because Wine dataset is small (178 samples)
    epochs = 400
    lr = 0.03         # Hebbian learning rate
    leak_rate = 0.4
    beta = 1.0        # Feedback nudging strength for clamped phase
    
    # Sparsity / Plasticity parameters
    target_density = 0.3
    prune_threshold = 0.03
    plasticity_interval = 10
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading Wine Chemical Classification Dataset (3 classes, 13 features)...")
    X_train, Y_train, X_test, Y_test = load_wine_data()
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)
    
    print(f"Dataset summary | Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
    print(f"Number of features: {X_train.shape[1]} | Classes: {Y_train.shape[1]}")
    
    # Initialize modular network
    # Sensory module inputs will map to the first 13 neurons (0 to 12).
    # Exit nodes will map to the last 3 neurons (61, 62, 63).
    net = StarryNet(
        num_neurons=num_neurons,
        leak_rate=leak_rate,
        input_indices=list(range(13)),
        output_indices=[-3, -2, -1],  # Nodes 61, 62, 63
        initial_density=0.8
    ).to(device)
    
    print(f"Initial connection density: {net.mask.sum().item() / net.prior_mask.sum().item():.2%}")
    print("Beginning Multi-Class Contrastive Hebbian Learning on Wine Dataset...")
    
    # Enable grad-free mode globally for local Hebbian training
    with torch.no_grad():
        for epoch in range(1, epochs + 1):
            # Train in mini-batches
            for X_batch, Y_batch in get_tabular_batches(X_train, Y_train, batch_size=batch_size, shuffle=True):
                # 1. Free Phase: inputs settle for 'steps' time steps
                outputs_free, states_free = net(X_batch, task_idx=2, targets=None, beta=beta, steps=steps)
                
                # 2. Clamped Phase: inputs settle while 3 exit nodes are nudged towards one-hot targets
                _, states_clamped = net(X_batch, task_idx=2, targets=Y_batch, beta=beta, steps=steps)
                
                # 3. Apply Hebbian update
                net.hebbian_update(states_free, states_clamped, task_idx=2, targets=Y_batch, lr=lr)
                
            # --- Structural Plasticity ---
            if epoch % plasticity_interval == 0:
                net.prune_connections(threshold=prune_threshold)
                net.grow_connections(target_density=target_density)
                
            # Evaluation & Logging
            if epoch % 20 == 0 or epoch == 1:
                # Training metrics (classification determined by argmax of predictions)
                p_train, _ = net(X_train, task_idx=2, steps=steps)
                loss_train = torch.mean((p_train - Y_train) ** 2).item()
                acc_train = (p_train.argmax(dim=1) == Y_train.argmax(dim=1)).float().mean().item()
                
                # Test metrics
                p_test, _ = net(X_test, task_idx=2, steps=steps)
                loss_test = torch.mean((p_test - Y_test) ** 2).item()
                acc_test = (p_test.argmax(dim=1) == Y_test.argmax(dim=1)).float().mean().item()
                
                density = net.mask.sum().item() / net.prior_mask.sum().item()
                print(f"Epoch {epoch:03d} | Train Loss: {loss_train:.4f} | Train Acc: {acc_train:.2%} | Test Acc: {acc_test:.2%} | Sparsity: {1.0 - density:.1%}")
                
    # Save the model
    model_path = 'models/starry_net.pth'
    torch.save(net.state_dict(), model_path)
    print(f"Hebbian training finished! Saved model parameters to '{model_path}'.")

if __name__ == "__main__":
    train()
