import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    if len(y) == 0:
        return 0.0
    
    # 2. Get the counts of each unique class
    _, counts = np.unique(y, return_counts=True)
    
    # 3. Calculate probabilities (pi)
    probabilities = counts / len(y)
    
    # 4. Compute entropy: H(S) = -sum(pi * log2(pi))
    # We use base 2 as per the requirements
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy)
   