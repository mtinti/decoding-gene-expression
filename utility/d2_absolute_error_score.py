import numpy as np

def d2_absolute_error_score(y_true, y_pred):
    """
    Calculate the D² score using absolute error, which measures the relative
    improvement of predictions over always predicting the median.
    
    A score of 1.0 is perfect prediction, 0.0 represents prediction equivalent
    to always guessing the median, and negative values indicate worse than
    predicting the median.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth target values
    y_pred : array-like
        Predicted target values
        
    Returns
    -------
    float
        The D² score
    
    Examples
    --------
    >>> y_true = [1, 2, 3]
    >>> y_pred = [1, 3, 3]
    >>> d2_absolute_error_score(y_true, y_pred)
    0.5
    >>> d2_absolute_error_score(y_true, y_true)  # Perfect predictions
    1.0
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) < 2:
        return float('nan')
    
    # Calculate mean absolute error between predictions and true values
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Calculate the baseline error (using median as constant prediction)
    y_median = np.median(y_true)
    baseline_mae = np.mean(np.abs(y_true - y_median))
    
    # If baseline error is 0, predictions are perfect if they match true values
    if baseline_mae == 0:
        return float(mae == 0)
    
    # Calculate D² score
    score = 1 - (mae / baseline_mae)
    
    return score

# Example usage and test cases
if __name__ == "__main__":
    # Test case 1: Perfect predictions
    y_true = [1, 2, 3, 4, 5]
    y_pred = [1, 2, 3, 4, 5]
    print(f"Perfect predictions: {d2_absolute_error_score(y_true, y_pred)}")  # Should be 1.0
    
    # Test case 2: Always predicting median
    y_pred_median = [3, 3, 3, 3, 3]  # median of y_true
    print(f"Median predictions: {d2_absolute_error_score(y_true, y_pred_median)}")  # Should be 0.0
    
    # Test case 3: Worse than median
    y_pred_bad = [5, 5, 5, 5, 5]
    print(f"Bad predictions: {d2_absolute_error_score(y_true, y_pred_bad)}")  # Should be negative
    
    # Test case 4: The example from the original docstring
    y_true = [1, 2, 3]
    y_pred = [1, 3, 3]
    print(f"Original example: {d2_absolute_error_score(y_true, y_pred)}")  # Should be 0.5