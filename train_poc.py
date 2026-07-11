import torch
import torch.nn as nn
import torch.optim as optim
from tasks import generate_addition_data, generate_echo_data
from starry_net import StarryNet

def train():
    # Hyperparameters
    num_neurons = 32
    seq_len = 15
    batch_size = 64
    epochs = 300
    lr = 0.01
    leak_rate = 0.4
    
    # Sparsity / Plasticity parameters
    target_density = 0.3
    prune_threshold = 0.05
    plasticity_interval = 10  # Apply structural changes every X epochs
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize network
    net = StarryNet(
        num_neurons=num_neurons,
        leak_rate=leak_rate,
        input_indices=[0, 1],       # Node 0 for Addition, Node 1 for Echo
        output_indices=[-1, -2],    # Node N-1 for Addition, Node N-2 for Echo
        initial_density=0.8
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    # Loss functions
    mse_loss_fn = nn.MSELoss()
    bce_loss_fn = nn.BCEWithLogitsLoss()
    
    print("Beginning StarryNN training...")
    print(f"Initial connection density: {net.mask.sum().item() / (num_neurons**2):.2%}")
    
    for epoch in range(1, epochs + 1):
        net.train()
        optimizer.zero_grad()
        
        # --- Task 1: Addition ---
        X1_raw, Y1_raw = generate_addition_data(batch_size, seq_len)
        X1, Y1 = X1_raw.to(device), Y1_raw.to(device)
        
        pred1, _ = net(X1, task_idx=0)
        # Squeeze output to match target: (batch_size, seq_len)
        loss1 = mse_loss_fn(pred1, Y1.squeeze(-1))
        
        # --- Task 2: Delayed Echo ---
        X2_raw, Y2_raw = generate_echo_data(batch_size, seq_len)
        X2, Y2 = X2_raw.to(device), Y2_raw.to(device)
        
        pred2, _ = net(X2, task_idx=1)
        loss2 = bce_loss_fn(pred2, Y2.squeeze(-1))
        
        # Combined Loss
        # We weigh them to balance magnitudes (MSE loss can be larger)
        loss = loss1 + 2.0 * loss2
        
        loss.backward()
        
        # Clip gradients to stabilize recurrent training
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        
        # Zero out gradients of pruned connections to ensure they don't get updated
        net.W_dense.grad.data.mul_(net.mask)
        
        optimizer.step()
        
        # --- Structural Plasticity ---
        if epoch % plasticity_interval == 0:
            pruned = net.prune_connections(threshold=prune_threshold)
            grown = net.grow_connections(target_density=target_density)
            
            # Re-verify mask weight updates (optimizer state reset isn't strictly needed for Adam,
            # but setting pruned grads to zero is good practice)
            # print(f"[Epoch {epoch}] Plasticity: Pruned {pruned}, Grown {grown}")
            
        # Logging & Metrics
        if epoch % 10 == 0 or epoch == 1:
            net.eval()
            with torch.no_grad():
                # Compute metrics on new test batches
                test_X1, test_Y1 = generate_addition_data(100, seq_len)
                test_X1, test_Y1 = test_X1.to(device), test_Y1.to(device)
                p1, _ = net(test_X1, task_idx=0)
                mae_task1 = torch.mean(torch.abs(p1 - test_Y1.squeeze(-1))).item()
                
                test_X2, test_Y2 = generate_echo_data(100, seq_len)
                test_X2, test_Y2 = test_X2.to(device), test_Y2.to(device)
                p2, _ = net(test_X2, task_idx=1)
                accuracy_task2 = ((p2 > 0.0) == test_Y2.squeeze(-1)).float().mean().item()
                
                density = net.mask.sum().item() / (num_neurons**2)
                
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Task1 MAE: {mae_task1:.4f} | Task2 Acc: {accuracy_task2:.2%} | Sparsity: {1.0 - density:.1%}")
                
    # Save the model
    torch.save(net.state_dict(), 'starry_net.pth')
    print("Training finished! Saved model parameters to 'starry_net.pth'.")

if __name__ == "__main__":
    train()
