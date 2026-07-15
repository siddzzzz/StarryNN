import torch
import torch.nn as nn
import numpy as np

class StarryNet(nn.Module):
    def __init__(self, num_neurons=32, leak_rate=0.25, input_indices=[0, 1], output_indices=[-1, -2], initial_density=0.8):
        super(StarryNet, self).__init__()
        self.num_neurons = num_neurons
        self.leak_rate = leak_rate # Leak rate alpha for LIF membrane potential decay
        self.input_indices = input_indices
        # Convert negative indices to positive for ease of mapping
        self.output_indices = [i if i >= 0 else num_neurons + i for i in output_indices]
        
        # Define functional brain modules
        if num_neurons == 64:
            num_sensory, num_assoc, num_motor = 30, 30, 4
        else:
            num_sensory = num_neurons // 3
            num_motor = num_neurons // 6 if num_neurons // 6 > 0 else 1
            num_assoc = num_neurons - num_sensory - num_motor
            
        # Create structural prior mask: 1 = allowed, 0 = blocked
        prior_mask = torch.ones(num_neurons, num_neurons)
        motor_start = num_sensory + num_assoc
        
        # Block direct Sensory <-> Motor connections
        prior_mask[:num_sensory, motor_start:] = 0.0  # Sensory to Motor
        prior_mask[motor_start:, :num_sensory] = 0.0  # Motor to Sensory
        self.register_buffer('prior_mask', prior_mask)
        
        # Dense weight matrix and bias (scaled for healthy SNN spike propagation)
        self.W_dense = nn.Parameter((torch.randn(num_neurons, num_neurons) / np.sqrt(num_neurons)) * 4.0)
        self.bias = nn.Parameter(torch.zeros(num_neurons))
        
        # Mask connections: must be a subset of the allowed prior_mask
        initial_mask = (torch.rand(num_neurons, num_neurons) < initial_density).float() * prior_mask
        self.register_buffer('mask', initial_mask)
        
    def forward(self, inputs, task_idx, teacher_targets=None, beta=6.0, steps=50):
        """
        inputs: shape (batch_size, num_features)
        task_idx: ignored (kept for API consistency)
        teacher_targets: shape (batch_size, num_outputs) (one-hot values {-1.0, 1.0})
        beta: teacher feedback current strength
        steps: T_sim simulation steps
        Returns:
            spike_counts: shape (batch_size, num_outputs) (spikes fired by exit nodes)
            spike_history: shape (batch_size, T_sim, num_neurons)
        """
        batch_size, num_features = inputs.shape
        device = inputs.device
        T_sim = steps
        
        # Initialize membrane potentials and spike history
        v = torch.zeros(batch_size, self.num_neurons, device=device)
        spike_history = []
        
        # Masked weights
        W = self.W_dense * self.mask
        
        exit_nodes = self.output_indices
        spikes = torch.zeros(batch_size, self.num_neurons, device=device)
        
        # Simulation loop over time
        for t in range(T_sim):
            # Direct Current Coding: constant feature currents
            input_currents = torch.zeros(batch_size, self.num_neurons, device=device)
            input_currents[:, 0:num_features] = inputs * 3.0
            
            # Apply teacher clamping currents to the output nodes during training
            if teacher_targets is not None:
                teacher_currents = torch.zeros(batch_size, self.num_neurons, device=device)
                teacher_currents[:, exit_nodes] = teacher_targets * beta
                input_currents += teacher_currents
                
            # Recurrent current from other neurons' spikes at t-1
            recurrent_current = torch.matmul(spikes, W.t())
            
            # Total input current
            total_current = recurrent_current + input_currents + self.bias
            
            # Update membrane potentials (leak dynamics)
            v = (1.0 - self.leak_rate) * v + total_current
            
            # Determine firing: threshold is 1.0
            spikes = (v >= 1.0).float()
            
            # Reset potentials for spiked neurons
            v = v * (1.0 - spikes)
            
            # Record state history
            spike_history.append(spikes.unsqueeze(1))
            
        spike_history = torch.cat(spike_history, dim=1)
        
        # Extract spike counts from the exit nodes
        exit_spikes = spike_history[:, :, exit_nodes]
        spike_counts = exit_spikes.sum(dim=1)
        
        return spike_counts, spike_history

    @torch.no_grad()
    def hebbian_update(self, spikes_free, spikes_clamped, task_idx, targets, lr=0.03):
        """
        Applies Contrastive Hebbian updates directly to the binary spike histories.
        """
        batch_size, T_sim, _ = spikes_free.shape
        
        # Recurrent updates using contrastive spike correlations
        c_t = spikes_clamped[:, 1:, :].reshape(-1, self.num_neurons)
        c_t_prev = spikes_clamped[:, :-1, :].reshape(-1, self.num_neurons)
        corr_clamped = torch.matmul(c_t.t(), c_t_prev) / c_t.shape[0]
        
        f_t = spikes_free[:, 1:, :].reshape(-1, self.num_neurons)
        f_t_prev = spikes_free[:, :-1, :].reshape(-1, self.num_neurons)
        corr_free = torch.matmul(f_t.t(), f_t_prev) / f_t.shape[0]
        
        dW = corr_clamped - corr_free
        self.W_dense.data += lr * dW * self.mask
        
        # Keep weights bounded in [-2.0, 2.0] and apply metabolic decay
        self.W_dense.data.clamp_(-2.0, 2.0)
        self.W_dense.data -= lr * 0.01 * self.W_dense.data * self.mask
        
        # Excitability homeostasis ONLY for non-exit nodes
        dbias = c_t.mean(dim=0) - f_t.mean(dim=0)
        h_mask = torch.ones(self.num_neurons, device=spikes_free.device)
        h_mask[self.output_indices] = 0.0
        
        self.bias.data += lr * dbias * h_mask
        self.bias.data.clamp_(-1.0, 1.0)

    @torch.no_grad()
    def prune_connections(self, threshold=0.1):
        """
        Prunes active connections whose weight magnitudes are below the threshold.
        """
        active_weights = torch.abs(self.W_dense) * self.mask
        prune_mask = (active_weights < threshold) & (self.mask == 1.0)
        self.mask[prune_mask] = 0.0
        num_pruned = prune_mask.sum().item()
        return num_pruned

    @torch.no_grad()
    def grow_connections(self, target_density=0.3):
        """
        Grows new random connections (restricted by prior_mask) to maintain target density.
        """
        max_allowed_connections = self.prior_mask.sum().item()
        current_active = self.mask.sum().item()
        current_density = current_active / max_allowed_connections
        
        if current_density >= target_density:
            return 0
        
        num_to_grow = int((target_density - current_density) * max_allowed_connections)
        pruned_indices = torch.where((self.mask == 0.0) & (self.prior_mask == 1.0))
        num_candidates = len(pruned_indices[0])
        
        if num_candidates == 0 or num_to_grow <= 0:
            return 0
        
        num_to_grow = min(num_to_grow, num_candidates)
        choice = np.random.choice(num_candidates, size=num_to_grow, replace=False)
        
        grow_rows = pruned_indices[0][choice]
        grow_cols = pruned_indices[1][choice]
        
        self.mask[grow_rows, grow_cols] = 1.0
        new_weights = torch.randn(num_to_grow, device=self.W_dense.device) * 0.01
        self.W_dense.data[grow_rows, grow_cols] = new_weights
        
        return num_to_grow

if __name__ == "__main__":
    # Test SNN creation and a forward pass
    net = StarryNet(num_neurons=8, input_indices=[0], output_indices=[-1])
    print(net)
    x = torch.randn(2, 5) # Batch size 2, 5 features
    counts, spikes = net(x, task_idx=0, steps=10)
    print("Spike counts shape:", counts.shape)
    print("Spike history shape:", spikes.shape)
    net.hebbian_update(spikes, spikes, task_idx=0, targets=None)
    print("SNN self-test successful!")
