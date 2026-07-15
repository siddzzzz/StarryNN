import torch
import numpy as np
from tasks import load_wine_data, get_tabular_batches
from starry_net import StarryNet

def train():
    # SNN Simulation Hyperparameters
    num_neurons = 64
    steps = 60        # T_sim simulation steps for spiking settling
    batch_size = 32
    epochs = 200
    lr = 0.03         # Spiking Contrastive Hebbian learning rate
    leak_rate = 0.25  # Membrane potential leak rate
    beta = 7.0        # Injected teacher current strength
    
    # Sparsity / Plasticity parameters
    target_density = 0.35
    prune_threshold = 0.05
    plasticity_interval = 10
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading Wine dataset for Spiking Network classification...")
    X_train, Y_train, X_test, Y_test = load_wine_data()
    
    # Move datasets to GPU/CPU
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)
    
    print(f"Dataset summary | Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
    
    # Initialize StarryNet configured as an SNN
    net = StarryNet(
        num_neurons=num_neurons,
        leak_rate=leak_rate,
        input_indices=list(range(13)),
        output_indices=[-3, -2, -1],
        initial_density=0.8
    ).to(device)
    
    print(f"Initial connection density: {net.mask.sum().item() / net.prior_mask.sum().item():.2%}")
    print("Beginning Contrastive Spiking Hebbian training on SNN...")
    
    with torch.no_grad():
        for epoch in range(1, epochs + 1):
            # Train in mini-batches
            for X_batch, Y_batch in get_tabular_batches(X_train, Y_train, batch_size=batch_size, shuffle=True):
                # 1. Free Phase (No teacher clamping)
                counts_free, spikes_free = net(
                    X_batch, task_idx=2, teacher_targets=None, steps=steps
                )
                
                # 2. Clamped Phase (Teacher clamping targets exits)
                _, spikes_clamped = net(
                    X_batch, task_idx=2, teacher_targets=Y_batch, beta=beta, steps=steps
                )
                
                # 3. Update recurrent weights using contrastive spike correlations
                net.hebbian_update(spikes_free, spikes_clamped, task_idx=2, targets=None, lr=lr)
                
            # --- Structural Plasticity ---
            if epoch % plasticity_interval == 0:
                net.prune_connections(threshold=prune_threshold)
                net.grow_connections(target_density=target_density)
                
            # Evaluation & Logging
            if epoch % 20 == 0 or epoch == 1:
                # Evaluation (WITHOUT teacher clamping - nodes spike purely by network propagation)
                # Training evaluation
                p_train, _ = net(X_train, task_idx=2, teacher_targets=None, steps=steps)
                acc_train = (p_train.argmax(dim=1) == Y_train.argmax(dim=1)).float().mean().item()
                
                # Test evaluation
                p_test, _ = net(X_test, task_idx=2, teacher_targets=None, steps=steps)
                acc_test = (p_test.argmax(dim=1) == Y_test.argmax(dim=1)).float().mean().item()
                
                density = net.mask.sum().item() / net.prior_mask.sum().item()
                avg_spikes = p_test.mean(dim=0).cpu().numpy()
                print(f"Epoch {epoch:03d} | Train Acc: {acc_train:.2%} | Test Acc: {acc_test:.2%} | Sparsity: {1.0 - density:.1%} | Avg Test Spikes/Node: {np.round(avg_spikes, 1)}")
                
    # Save the spiking model parameters
    model_path = 'models/starry_net.pth'
    torch.save(net.state_dict(), model_path)
    print(f"SNN training finished! Saved model parameters to '{model_path}'.")

if __name__ == "__main__":
    train()
