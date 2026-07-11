import torch

def generate_addition_data(batch_size, seq_len, sparse_prob=0.3):
    """
    Generates data for the addition task.
    Input: A sparse sequence of numbers in [-1, 1].
    Target: The running cumulative sum of the sequence.
    """
    # Create mask for sparsity
    mask = (torch.rand(batch_size, seq_len, 1) < sparse_prob).float()
    values = (torch.rand(batch_size, seq_len, 1) * 2.0 - 1.0) * mask
    
    # Running cumulative sum
    targets = torch.cumsum(values, dim=1)
    
    return values, targets

def generate_echo_data(batch_size, seq_len, delay=2, sparse_prob=0.4):
    """
    Generates data for the delayed echo task.
    Input: A binary sequence (0 or 1).
    Target: The input value delayed by 'delay' steps (zeros for the first 'delay' steps).
    """
    inputs = (torch.rand(batch_size, seq_len, 1) < sparse_prob).float()
    targets = torch.zeros_like(inputs)
    targets[:, delay:] = inputs[:, :-delay]
    return inputs, targets

if __name__ == "__main__":
    # Simple self-test
    x_add, y_add = generate_addition_data(2, 5)
    print("Addition Input:\n", x_add)
    print("Addition Target:\n", y_add)
    
    x_echo, y_echo = generate_echo_data(2, 5)
    print("Echo Input:\n", x_echo)
    print("Echo Target:\n", y_echo)
