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
        
        # Dense weight matrix and bias
        self.W_dense = nn.Parameter(torch.randn(num_neurons, num_neurons) / np.sqrt(num_neurons))
        self.bias = nn.Parameter(torch.zeros(num_neurons))
        
        # Binary connection mask (1 = active, 0 = pruned)
        # We register it as a buffer so it is saved in the state_dict but not optimized by SGD
        initial_mask = (torch.rand(num_neurons, num_neurons) < initial_density).float()
        # Self-connections can be allowed or disabled. Let's allow them but mask can block them.
        self.register_buffer('mask', initial_mask)
        
        # Task readouts: since exit neurons themselves represent the output,
        # we map their scalar state to the target scale.
        # Task 1 (Addition) needs a linear scaling (weight & bias)
        self.readout1_w = nn.Parameter(torch.tensor(1.0))
        self.readout1_b = nn.Parameter(torch.tensor(0.0))
        
        # Task 2 (Parity) needs a linear scaling for classification logits
        self.readout2_w = nn.Parameter(torch.tensor(1.0))
        self.readout2_b = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, inputs, task_idx, targets=None, beta=0.5, steps=8):
        """
        inputs: 
            - Sequence data: shape (batch_size, seq_len, 1)
            - Tabular data: shape (batch_size, num_features)
        task_idx: 
            - 0: Task 1 (Addition)
            - 1: Task 2 (Echo)
            - 2: Tabular Task (Breast Cancer Classification)
        targets: 
            - Sequence targets: shape (batch_size, seq_len)
            - Tabular targets: shape (batch_size, 1)
        beta: nudging/feedback strength
        steps: relaxation steps for tabular data
        Returns:
            outputs: shape (batch_size, seq_len) for sequences, (batch_size) for tabular
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
        
        # Determine exit node (always Node N-1 for tabular, or task-specific for sequences)
        if is_tabular:
            exit_node = self.num_neurons - 1
            readout_w, readout_b = self.readout2_w, self.readout2_b  # Reuse readout2 for classification
        else:
            entry_node = self.input_indices[task_idx]
            exit_node = self.output_indices[task_idx]
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
                y_target = targets if is_tabular else targets[:, t]
                
                # Map target to target state space
                target_state = (y_target.squeeze(-1) if y_target.dim() > 1 else y_target) - readout_b
                target_state = target_state / (readout_w + 1e-5)
                target_state = torch.clamp(target_state, -1.0, 1.0)
                
                # Nudge exit neuron
                node_inputs[:, exit_node] += beta * target_state
            
            # Recurrent input
            recurrent_input = torch.matmul(state, W.t())
            
            # State update
            state_candidate = torch.tanh(recurrent_input + node_inputs + self.bias)
            state = (1.0 - self.leak_rate) * state + self.leak_rate * state_candidate
            states_history.append(state.unsqueeze(1))
            
        states_history = torch.cat(states_history, dim=1)
        exit_states = states_history[:, :, exit_node]
        
        if is_tabular:
            # For tabular classification, we only care about the final settled state of the exit node
            outputs = self.readout2_w * exit_states[:, -1] + self.readout2_b
        else:
            if task_idx == 0:
                outputs = self.readout1_w * exit_states + self.readout1_b
            else:
                outputs = self.readout2_w * exit_states + self.readout2_b
                
        return outputs, states_history

    @torch.no_grad()
    def hebbian_update(self, states_free, states_clamped, task_idx, targets, lr=0.01):
        """
        Applies Contrastive Hebbian Learning (CHL) rule to update W_dense and bias.
        Updates readout mapping parameters using the delta rule.
        """
        batch_size, seq_len, _ = states_free.shape
        is_tabular = (targets.dim() == 2 and targets.shape[1] == 1)  # targets: (batch_size, 1)
        
        if is_tabular:
            exit_node = self.num_neurons - 1
            w, b = self.readout2_w, self.readout2_b
        else:
            exit_node = self.output_indices[task_idx]
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
            # For tabular, targets is shape (batch_size, 1), prediction is shape (batch_size)
            final_exit_states = states_free[:, -1, exit_node]
            predictions = w * final_exit_states + b
            errors = predictions - targets.squeeze(-1)
            
            dw = - (errors * final_exit_states).mean()
            db = - errors.mean()
            
            self.readout2_w.data += lr * dw
            self.readout2_b.data += lr * db
        else:
            exit_states = states_free[:, :, exit_node]
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
        Grows new random connections to maintain the target density.
        The new connections are initialized with small random weights.
        """
        current_density = self.mask.sum().item() / (self.num_neurons ** 2)
        if current_density >= target_density:
            return 0
        
        # Determine how many connections to grow
        total_connections = self.num_neurons ** 2
        num_to_grow = int((target_density - current_density) * total_connections)
        
        # Find all candidate positions where mask is 0
        pruned_indices = torch.where(self.mask == 0.0)
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
