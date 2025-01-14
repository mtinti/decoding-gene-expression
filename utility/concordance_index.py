import numpy as np

def concordance_index(y_true, y_pred):
    """
    Vectorized implementation of concordance index calculation.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Create matrices of pairwise differences
    true_diff = y_true[:, np.newaxis] - y_true
    pred_diff = y_pred[:, np.newaxis] - y_pred
    
    # Create mask for valid pairs (excluding self-comparisons and duplicates)
    mask = true_diff != 0
    
    # Calculate concordant and discordant pairs using sign comparison
    concordant = np.sum((true_diff * pred_diff > 0) & mask) / 2
    discordant = np.sum((true_diff * pred_diff < 0) & mask) / 2
    tied_pred = np.sum((pred_diff == 0) & mask) / 2
    
    total_pairs = concordant + discordant + tied_pred
    
    if total_pairs == 0:
        return np.nan
        
    return (concordant + 0.5 * tied_pred) / total_pairs

# Test cases remain the same
if __name__ == "__main__":
    y_true = [1, 2, 3, 4, 5]
    y_pred = [10, 20, 30, 40, 50]
    print(f"Perfect order: {concordance_index(y_true, y_pred)}")
    
    y_pred_rev = [50, 40, 30, 20, 10]
    print(f"Reverse order: {concordance_index(y_true, y_pred_rev)}")
    
    y_pred_rand = [30, 10, 50, 20, 40]
    print(f"Random order: {concordance_index(y_true, y_pred_rand)}")
    
    y_pred_tied = [10, 10, 20, 20, 20]
    print(f"Tied predictions: {concordance_index(y_true, y_pred_tied)}")
    
    y_true_ties = [1, 2, 2, 3, 3]
    y_pred_some = [1, 2, 3, 4, 5]
    print(f"Some equal true values: {concordance_index(y_true_ties, y_pred_some)}")