import numpy as np
from concordance_index import concordance_index
from d2_absolute_error_score import d2_absolute_error_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

def abline(y_test, pred, ax, 
          visualize=['R2', 'mse', 'Pr', 'Sp', 'D2', 'Ci', 'mape', 'n_rmse'],
          text_loc=[0.05, 0.95]):
    
    # Add regression line
    m, b = np.polyfit(y_test, pred, 1)
    #ax.plot(y_test, m*y_test + b, 'r-')
    
    # Calculate metrics
    r2 = r2_score(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    pearson_r, _ = pearsonr(y_test, pred)
    spearman_r, _ = spearmanr(y_test, pred)
    d2 = d2_absolute_error_score(y_test, pred)
    ci = concordance_index(y_test, pred)
    mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
    rmse = np.sqrt(mse)
    normalized_rmse = rmse / y_test.mean()
    
    scores = {
        'R2': r2,
        'mse': mse,
        'Pr': pearson_r,
        'Sp': spearman_r,
        'D2': d2,
        'Ci': ci,
        'mape': mape,
        'n_rmse': normalized_rmse
    }
    
    stats_text = ''
    for indicator in visualize:
        stats_text += f'{indicator}: {scores[indicator]:.2f}\n'
        
    ax.text(text_loc[0], text_loc[1], stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=16)

if __name__ == "__main__":    
    # Test cases and plotting remain the same
    x1 = np.array([1, 2, 3, 4, 5])
    y1 = np.array([1, 2, 3, 4, 5])
    
    x2 = np.array([1, 2, 3, 4, 5])
    y2 = np.array([5, 4, 3, 2, 1])
    
    np.random.seed(42)
    x3 = np.random.normal(0, 1, 500)
    y3 = x3 + np.random.normal(0, 0.5, 500)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.scatter(x1, y1)
    ax1.set_title('Perfect Correlation')
    abline(x1, y1, ax1, visualize=['R2', 'Pr'])
    
    ax2.scatter(x2, y2)
    ax2.set_title('Negative Correlation')
    abline(x2, y2, ax2, visualize=['R2', 'Sp'])
    
    ax3.scatter(x3, y3)
    ax3.set_title('Random Data')
    abline(x3, y3, ax3, visualize=['R2', 'mse', 'mape'])
    
    plt.tight_layout()
    plt.show()