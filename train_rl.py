import torch
import numpy as np
import os
from starry_net import StarryNet

# Robust Gym / Gymnasium importer
try:
    import gymnasium as gym
except ImportError:
    try:
        import gym
    except ImportError:
        raise ImportError("Please install gymnasium or gym to run reinforcement learning: pip install gymnasium")

def train_rl():
    # SNN Configuration
    num_neurons = 64
    steps = 30        # T_sim steps per action selection
    episodes = 200
    lr = 0.015        # Episode-end Hebbian REINFORCE learning rate
    leak_rate = 0.25  # Membrane potential leak rate
    
    # Plasticity boundaries
    target_density = 0.35
    prune_threshold = 0.04
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize environment
    try:
        env = gym.make('CartPole-v1')
        print("Successfully initialized CartPole-v1 environment.")
    except Exception as e:
        print("Could not create CartPole-v1 environment:", e)
        return
        
    # Initialize StarryNet as SNN
    net = StarryNet(
        num_neurons=num_neurons,
        leak_rate=leak_rate,
        input_indices=[0, 1, 2, 3],
        output_indices=[-2, -1],  # Action 0 (Left), Action 1 (Right)
        initial_density=0.8
    ).to(device)
    
    print(f"Initial connection density: {net.mask.sum().item() / net.prior_mask.sum().item():.2%}")
    print("Starting Spiking RL training via Episode-End Hebbian REINFORCE...")
    
    # Track survival lengths (rewards)
    episode_rewards = []
    baseline = 10.0  # Initial baseline reward
    
    # Feature scaling: map observations to SNN currents
    scale = torch.tensor([5.0, 2.0, 15.0, 5.0], device=device)
    
    for ep in range(1, episodes + 1):
        # Reset environment with API compatibility
        reset_res = env.reset()
        if isinstance(reset_res, tuple):
            obs, info = reset_res
        else:
            obs = reset_res
            
        total_reward = 0
        done = False
        
        # Initialize episode eligibility traces
        episode_eligibility = torch.zeros(num_neurons, num_neurons, device=device)
        episode_dbias = torch.zeros(num_neurons, device=device)
        
        while not done:
            state_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            scaled_state = state_tensor * scale
            
            # 1. Run forward SNN pass (weights are static during the episode!)
            counts, spikes = net(scaled_state, task_idx=2, teacher_targets=None, steps=steps)
            
            # Action selection: argmax exit nodes spike counts
            if counts.sum() == 0:
                action = env.action_space.sample()  # Explore randomly if no spikes
            else:
                action = counts.argmax(dim=1).item()
                
            # 2. Step environment
            step_res = env.step(action)
            if len(step_res) == 5:
                next_obs, reward, terminated, truncated, info = step_res
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_res
                terminated = done
                
            total_reward += reward
            
            # Accumulate eligibility trace (spike correlations) for this step
            c_t = spikes[:, 1:, :].reshape(-1, num_neurons)
            c_t_prev = spikes[:, :-1, :].reshape(-1, num_neurons)
            step_corr = torch.matmul(c_t.t(), c_t_prev) / c_t.shape[0]
            
            episode_eligibility += step_corr
            episode_dbias += c_t.mean(dim=0)
            
            obs = next_obs
            
        # --- Episode End Update ---
        # Advantage: how much better/worse this episode was compared to average
        advantage = total_reward - baseline
        # Clamp advantage to prevent runaway weights on lucky runs
        advantage_clamped = np.clip(advantage, -15.0, 15.0)
        
        # Apply three-factor update at episode end
        net.W_dense.data += lr * advantage_clamped * episode_eligibility * net.mask
        net.W_dense.data.clamp_(-2.0, 2.0)
        net.W_dense.data -= lr * 0.01 * net.W_dense.data * net.mask
        
        # Update biases homeostatically modulated by advantage
        h_mask = torch.ones(num_neurons, device=device)
        h_mask[net.output_indices] = 0.0
        net.bias.data += lr * 0.1 * advantage_clamped * (episode_dbias / max(total_reward, 1)) * h_mask
        net.bias.data.clamp_(-1.0, 1.0)
        
        # Update running baseline
        baseline = 0.95 * baseline + 0.05 * total_reward
        
        # --- Structural Plasticity ---
        if ep % 10 == 0:
            net.prune_connections(threshold=prune_threshold)
            net.grow_connections(target_density=target_density)
            
        episode_rewards.append(total_reward)
        
        # Logging progress
        if ep % 10 == 0 or ep == 1:
            recent_avg = np.mean(episode_rewards[-10:])
            density = net.mask.sum().item() / net.prior_mask.sum().item()
            print(f"Episode {ep:03d} | Survival Steps: {total_reward:.0f} | 10-Ep Avg: {recent_avg:.1f} | Sparsity: {1.0 - density:.1%}")
            
    # Save the trained controller model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/starry_net.pth'
    torch.save(net.state_dict(), model_path)
    print(f"RL training finished! Saved trained model parameters to '{model_path}'.")
    env.close()

if __name__ == "__main__":
    train_rl()
