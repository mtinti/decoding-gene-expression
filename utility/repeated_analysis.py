import numpy as np
import pandas as pd
from tqdm import tqdm
from concordance_index import concordance_index
from d2_absolute_error_score import d2_absolute_error_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.datasets import make_regression
import scipy.stats as stats



def repeated_analysis(X, y, n_repeats=100):
    results = []
    for i in tqdm(range(n_repeats)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
        
        regr = RandomForestRegressor(max_depth=4, 
                                   n_estimators=100, 
                                   min_samples_leaf=20, 
                                   max_features=4,
                                   n_jobs=6,
                                   random_state=i)
        
        regr.fit(X_train, y_train)
        pred = regr.predict(X_test)
        
        r2 = r2_score(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        pearson_r, _ = pearsonr(y_test, pred)
        spearman_r, _ = spearmanr(y_test, pred)
        d2 = d2_absolute_error_score(y_test, pred)
        ci = concordance_index(y_test, pred)
        mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
        rmse = np.sqrt(mse)
        normalized_rmse = rmse / y_test.mean()
        
        tau, p_value = stats.kendalltau(y_test, pred)
        
        scores = {
            'R2': r2,
            'mse': mse,
            'Pr': pearson_r,
            'Sp': spearman_r,
            'D2': d2,
            'Ci': ci,
            'mape': mape,
            'n_rmse': normalized_rmse,
            'Kn': tau
        }
        
        results.append(scores)
        
    return pd.DataFrame(results)





if __name__ == "__main__":
    # Generate synthetic data
    X, y = make_regression(n_samples=1000, 
                          n_features=5, 
                          noise=0.1, 
                          random_state=42)

    # Add some non-linear relationships
    X[:, 0] = np.exp(X[:, 0])
    y = np.exp(y)

    # Run analysis
    results_df = repeated_analysis(X, y, n_repeats=10)

    # Print summary statistics
    print("\nMetrics Summary:")
    print(results_df.describe())

    # Print correlations between metrics
    print("\nMetrics Correlations:")
    print(results_df.corr().round(3))
