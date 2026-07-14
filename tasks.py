import torch
import numpy as np

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
    # Map binary target {0, 1} to {-1.0, 1.0} for easier feedback clamping
    targets = targets * 2.0 - 1.0
    return inputs, targets

def load_tabular_data(test_size=0.2):
    """
    Loads and normalizes the Breast Cancer Wisconsin Diagnostic dataset.
    Returns:
        X_train, Y_train, X_test, Y_test (as PyTorch tensors)
    """
    try:
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError("scikit-learn is required for tabular data loading. Run: pip install scikit-learn")
        
    data = load_breast_cancer()
    X, Y = data.data, data.target  # X has 30 features, Y has labels {0, 1}
    
    # Scale features to mean 0, variance 1
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Map target labels {0, 1} to {-1.0, 1.0} for Hebbian feedback compatibility
    Y = Y * 2.0 - 1.0
    
    # Split dataset
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=42, stratify=Y
    )
    
    # Convert arrays to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(-1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(-1)
    
    return X_train, Y_train, X_test, Y_test

def load_wine_data(test_size=0.2):
    """
    Loads and normalizes the Wine classification dataset.
    Returns:
        X_train, Y_train, X_test, Y_test (as PyTorch tensors)
        Y is one-hot encoded to {-1.0, 1.0} of shape (num_samples, 3)
    """
    try:
        from sklearn.datasets import load_wine
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError("scikit-learn is required. Run: pip install scikit-learn")
        
    data = load_wine()
    X, Y_labels = data.data, data.target  # X has 13 features, Y has labels {0, 1, 2}
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # One-hot encode targets into {-1.0, 1.0}
    num_classes = len(np.unique(Y_labels))
    Y_onehot = np.full((len(Y_labels), num_classes), -1.0)
    for idx, val in enumerate(Y_labels):
        Y_onehot[idx, val] = 1.0
        
    # Split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y_onehot, test_size=test_size, random_state=42, stratify=Y_labels
    )
    
    # Convert arrays to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)
    
    return X_train, Y_train, X_test, Y_test

def get_tabular_batches(X, Y, batch_size=64, shuffle=True):
    """
    Yields mini-batches of tabular data.
    """
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(indices)
    
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_idx = indices[start:end]
        yield X[batch_idx], Y[batch_idx]

if __name__ == "__main__":
    # Simple self-test
    print("Testing synthetic sequence generators...")
    x_add, y_add = generate_addition_data(2, 5)
    x_echo, y_echo = generate_echo_data(2, 5)
    print("Addition Input shape:", x_add.shape)
    print("Echo Input shape:", x_echo.shape)
    
    try:
        X_tr, Y_tr, X_te, Y_te = load_tabular_data()
        print("\nSuccessfully loaded Breast Cancer dataset:")
        print(f"Train features shape: {X_tr.shape} | Train labels shape: {Y_tr.shape}")
        print(f"Test features shape: {X_te.shape} | Test labels shape: {Y_te.shape}")
        print("Label values range:", torch.unique(Y_tr))
    except Exception as e:
        print("\nCould not load Breast Cancer dataset:", e)
        
    try:
        X_wtr, Y_wtr, X_wte, Y_wte = load_wine_data()
        print("\nSuccessfully loaded Wine dataset:")
        print(f"Train features shape: {X_wtr.shape} | Train labels shape: {Y_wtr.shape}")
        print(f"Test features shape: {X_wte.shape} | Test labels shape: {Y_wte.shape}")
        print("One-hot label representation check (first 3 samples):\n", Y_wtr[:3])
    except Exception as e:
        print("\nCould not load Wine dataset:", e)
