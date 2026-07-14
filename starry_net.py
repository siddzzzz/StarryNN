import torch
import torch.nn as nn
import numpy as np

class StarryNet(nn.Module):
    def __init__(self, num_neurons=32, leak_rate=0.5, input_indices=[0, 1], output_indices=[-1, -2], initial_density=0.8):
        super(StarryNet, self).__init__()
        self.num_neurons = num_neurons
        self.leak_rate = leak_rate
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
        
        # Dense weight matrix and bias
        self.W_dense = nn.Parameter(torch.randn(num_neurons, num_neurons) / np.sqrt(num_neurons))
        self.bias = nn.Parameter(torch.zeros(num_neurons))
        
        # Mask connections: must be a subset of the allowed prior_mask
        initial_mask = (torch.rand(num_neurons, num_neurons) < initial_density).float() * prior_mask
        self.register_buffer('mask', initial_mask)
        
        # Task readouts: since exit neurons themselves represent the output,
        # we map their scalar state to the target scale.
        # Task 1 (Addition) needs a linear scaling (weight & bias)
        self.readout1_w = nn.Parameter(torch.tensor(1.0))
        self.readout1_b = nn.Parameter(torch.tensor(0.0))
        
        # Task 2 (Parity/Echo) needs a linear scaling for classification logits
        self.readout2_w = nn.Parameter(torch.tensor(1.0))
        self.readout2_b = nn.Parameter(torch.tensor(0.0))
        
        # Tabular Task: We support multiple outputs based on output_indices length
        self.num_outputs = len(output_indices)
        self.readout_tabular_w = nn.Parameter(torch.ones(self.num_outputs))
        self.readout_tabular_b = nn.Parameter(torch.zeros(self.num_outputs))
        
    def forward(self, inputs, task_idx, targets=None, beta=0.5, steps=8):
        """
        inputs: 
            - Sequence data: shape (batch_size, seq_len, 1)
            - Tabular data: shape (batch_size, num_features)
        task_idx: 
            - 0: Task 1 (Addition)
            - 1: Task 2 (Echo)
            - 2: Tabular Task (Classification)
        targets: 
            - Sequence targets: shape (batch_size, seq_len)
            - Tabular targets: shape (batch_size, num_outputs)
        beta: nudging/feedback strength
        steps: relaxation steps for tabular data
        Returns:
            outputs: shape (batch_size, seq_len) for sequences, (batch_size, num_outputs) for tabular
            states: shape (batch_size, seq_len/steps, num_neurons)
        """
        device = inputs.device
        
        # Dual-mode detection: Sequence vs Tabular
        is_tabular = (inputs.dim() == 2)
        
        if is_tabular:
            batch_size, num_features = inputs.shape
            seq_len = steps
        else:
            batch_size, seq_len, _ = inputs.shape
            num_features = 1
            
        # Initialize neuron states to zero
        state = torch.zeros(batch_size, self.num_neurons, device=device)
        states_history = []
        
        # Masked weights
        W = self.W_dense * self.mask
        
        # Determine exit nodes (list of indices)
        if is_tabular:
            exit_nodes = self.output_indices
            readout_w, readout_b = self.readout_tabular_w, self.readout_tabular_b
        else:
            entry_node = self.input_indices[task_idx]
            exit_nodes = [self.output_indices[task_idx]]
            if task_idx == 0:
                readout_w, readout_b = self.readout1_w, self.readout1_b
            else:
                readout_w, readout_b = self.readout2_w, self.readout2_b
        
        # Propagate through time
        for t in range(seq_len):
            # Prepare inputs to all nodes
            node_inputs = torch.zeros(batch_size, self.num_neurons, device=device)
            
            if is_tabular:
                # Clamp all features to nodes 0 to num_features-1
                node_inputs[:, 0:num_features] = inputs
            else:
                # Clamp single sequence input to entry_node
                node_inputs[:, entry_node] = inputs[:, t, :].squeeze(-1)
            
            # If targets are provided, apply feedback nudge
            if targets is not None:
                # Extract target for current step
                y_target = targets if is_tabular else targets[:, t].unsqueeze(-1)
                
                # Map target to target state space
                target_state = (y_target - readout_b) / (readout_w + 1e-5)
                target_state = torch.clamp(target_state, -1.0, 1.0)
                
                # Nudge exit neurons
                node_inputs[:, exit_nodes] += beta * target_state
            
            # Recurrent input
            recurrent_input = torch.matmul(state, W.t())
            
            # State update
            state_candidate = torch.tanh(recurrent_input + node_inputs + self.bias)
            state = (1.0 - self.leak_rate) * state + self.leak_rate * state_candidate
            states_history.append(state.unsqueeze(1))
            
        states_history = torch.cat(states_history, dim=1)
        exit_states = states_history[:, :, exit_nodes] # shape (batch_size, seq_len, num_outputs)
        
        if is_tabular:
            # For tabular classification, we only care about the final settled state of the exit nodes
            outputs = self.readout_tabular_w * exit_states[:, -1, :] + self.readout_tabular_b
            # Squeeze output to shape (batch_size) if num_outputs is 1 (for Breast Cancer compatibility)
            if self.num_outputs == 1:
                outputs = outputs.squeeze(-1)
        else:
            if task_idx == 0:
                outputs = self.readout1_w * exit_states.squeeze(-1) + self.readout1_b
            else:
                outputs = self.readout2_w * exit_states.squeeze(-1) + self.readout2_b
                
        return outputs, states_history

    @torch.no_grad()
    def hebbian_update(self, states_free, states_clamped, task_idx, targets, lr=0.01):
        """
        Applies Contrastive Hebbian Learning (CHL) rule to update W_dense and bias.
        Updates readout mapping parameters using the delta rule.
        """
        batch_size, seq_len, _ = states_free.shape
        is_tabular = (targets.dim() == 2)
        
        if is_tabular:
            exit_nodes = self.output_indices
            w, b = self.readout_tabular_w, self.readout_tabular_b
        else:
            exit_nodes = [self.output_indices[task_idx]]
            if task_idx == 0:
                w, b = self.readout1_w, self.readout1_b
            else:
                w, b = self.readout2_w, self.readout2_b
        
        # 1. Update W_dense using recurrent correlations: W_ij connects j -> i
        clamped_t = states_clamped[:, 1:, :]
        clamped_t_prev = states_clamped[:, :-1, :]
        c_t = clamped_t.reshape(-1, self.num_neurons)
        c_t_prev = clamped_t_prev.reshape(-1, self.num_neurons)
        corr_clamped = torch.matmul(c_t.t(), c_t_prev) / c_t.shape[0]
        
        free_t = states_free[:, 1:, :]
        free_t_prev = states_free[:, :-1, :]
        f_t = free_t.reshape(-1, self.num_neurons)
        f_t_prev = free_t_prev.reshape(-1, self.num_neurons)
        corr_free = torch.matmul(f_t.t(), f_t_prev) / f_t.shape[0]
        
        # Contrastive Hebbian update rule: dW = Clamped_corr - Free_corr
        dW = corr_clamped - corr_free
        self.W_dense.data += lr * dW * self.mask
        
        # Clamp weights and apply weight decay to simulate synaptic metabolic decay
        self.W_dense.data.clamp_(-2.0, 2.0)
        self.W_dense.data -= lr * 0.01 * self.W_dense.data * self.mask
        
        # 2. Update node biases (intrinsic excitability homeostasis)
        dbias = c_t.mean(dim=0) - f_t.mean(dim=0)
        self.bias.data += lr * dbias
        self.bias.data.clamp_(-1.0, 1.0)
        
        # 3. Update task-specific readouts using local delta rule
        if is_tabular:
            # final_exit_states shape: (batch_size, num_outputs)
            final_exit_states = states_free[:, -1, exit_nodes]
            predictions = w * final_exit_states + b
            errors = predictions - targets # shape (batch_size, num_outputs)
            
            dw = - (errors * final_exit_states).mean(dim=0)
            db = - errors.mean(dim=0)
            
            self.readout_tabular_w.data += lr * dw
            self.readout_tabular_b.data += lr * db
        else:
            exit_states = states_free[:, :, exit_nodes].squeeze(-1)
            predictions = w * exit_states + b
            errors = predictions - targets
            
            dw = - (errors * exit_states).mean()
            db = - errors.mean()
            
            if task_idx == 0:
                self.readout1_w.data += lr * dw
                self.readout1_b.data += lr * db
            else:
                self.readout2_w.data += lr * dw
                self.readout2_b.data += lr * db

    @torch.no_grad()
    def prune_connections(self, threshold=0.1):
        """
        Prunes active connections whose weight magnitudes are below the threshold.
        """
        # Find active connections
        active_weights = torch.abs(self.W_dense) * self.mask
        # Create a pruning mask: where active weights are below threshold
        prune_mask = (active_weights < threshold) & (self.mask == 1.0)
        
        # Prune connections
        self.mask[prune_mask] = 0.0
        num_pruned = prune_mask.sum().item()
        return num_pruned

    @torch.no_grad()
    def grow_connections(self, target_density=0.3):
        """
        Grows new random connections (restricted by prior_mask) to maintain the target density.
        The density is computed relative to the maximum allowed connections in prior_mask.
        """
        max_allowed_connections = self.prior_mask.sum().item()
        current_active = self.mask.sum().item()
        current_density = current_active / max_allowed_connections
        
        if current_density >= target_density:
            return 0
        
        # Determine how many connections to grow
        num_to_grow = int((target_density - current_density) * max_allowed_connections)
        
        # Find all candidate positions where mask is 0 AND prior_mask is 1
        pruned_indices = torch.where((self.mask == 0.0) & (self.prior_mask == 1.0))
        num_candidates = len(pruned_indices[0])
        
        if num_candidates == 0 or num_to_grow <= 0:
            return 0
        
        # Randomly choose indices to grow
        num_to_grow = min(num_to_grow, num_candidates)
        choice = np.random.choice(num_candidates, size=num_to_grow, replace=False)
        
        grow_rows = pruned_indices[0][choice]
        grow_cols = pruned_indices[1][choice]
        
        # Update mask and initialize weights
        self.mask[grow_rows, grow_cols] = 1.0
        # Initialize new weights with small normal values
        new_weights = torch.randn(num_to_grow, device=self.W_dense.device) * 0.01
        self.W_dense.data[grow_rows, grow_cols] = new_weights
        
        return num_to_grow

if __name__ == "__main__":
    # Test network creation and a forward pass
    net = StarryNet(num_neurons=8, input_indices=[0, 1], output_indices=[-1, -2])
    print(net)
    x = torch.randn(2, 5, 1) # Batch size 2, Seq len 5
    y1, _ = net(x, task_idx=0)
    y2, _ = net(x, task_idx=1)
    print("Task 1 output shape:", y1.shape)
    print("Task 2 output shape:", y2.shape)
    
    # Test pruning
    pruned = net.prune_connections(threshold=0.2)
    print(f"Pruned {pruned} connections. Current density: {net.mask.sum().item() / 64:.2f}")
    
    # Test growing
    grown = net.grow_connections(target_density=0.5)
    print(f"Grown {grown} connections. Current density: {net.mask.sum().item() / 64:.2f}")
