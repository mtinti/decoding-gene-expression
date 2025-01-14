import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

def cross_validate_random_forest(X, y, n_splits=3, random_state=42, params=None):
    """
    Perform cross-validation with Random Forest regression.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        n_splits (int): Number of CV folds
        random_state (int): Random seed
        params (dict): RandomForest parameters
    """
    if params is None:
        params = {
            'max_depth': 5,
            'n_estimators': 100,
            'min_samples_leaf': 10,
            'max_features': 5,
            'random_state': random_state
        }
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    regr = RandomForestRegressor(**params)
    
    all_predictions = pd.Series(index=y.index, dtype=float)
    all_true_values = pd.Series(index=y.index, dtype=float)
    fold_metrics = []
    
    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        regr.fit(X_train, y_train)
        fold_predictions = regr.predict(X_test)
        
        all_predictions.iloc[test_index] = fold_predictions
        all_true_values.iloc[test_index] = y_test
        
        metrics = {
            'fold': fold,
            'r2': r2_score(y_test, fold_predictions),
            'mse': mean_squared_error(y_test, fold_predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, fold_predictions))
        }
        fold_metrics.append(metrics)
        
        print(f"Fold {fold} - R2: {metrics['r2']:.4f}, "
              f"RMSE: {metrics['rmse']:.4f}")
    
    fold_metrics = pd.DataFrame(fold_metrics)
    print("\nOverall Metrics:")
    print(f"Mean R2: {fold_metrics['r2'].mean():.4f} ± {fold_metrics['r2'].std():.4f}")
    print(f"Mean RMSE: {fold_metrics['rmse'].mean():.4f} ± {fold_metrics['rmse'].std():.4f}")
    
    return {
        'predictions': all_predictions,
        'true_values': all_true_values,
        'model': regr,
        'fold_metrics': fold_metrics,
        'train_data': (X_train, y_train),
        'test_data': (X_test, y_test)
    }




if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_regression
    from sklearn.model_selection import KFold
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error

    # Generate synthetic dataset
    np.random.seed(42)
    X_synthetic, y_synthetic = make_regression(n_samples=1000, 
                                             n_features=10,
                                             noise=0.1)

    # Convert to pandas DataFrame/Series with index
    X = pd.DataFrame(X_synthetic, columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(y_synthetic)

    # Test basic functionality
    predictions, true_values, model, X_train, X_test, y_train, y_test = (
        cross_validate_random_forest(X, y, n_splits=3))

    # Print overall metrics
    print("\nOverall Metrics:")
    print(f"R2: {r2_score(true_values, predictions):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(true_values, predictions)):.4f}")

    # Test edge cases
    print("\nEdge Cases:")

    # Test with small dataset
    X_small = X.head(10)
    y_small = y.head(10)
    try:
        cross_validate_random_forest(X_small, y_small, n_splits=3)
        print("Small dataset: Passed")
    except Exception as e:
        print(f"Small dataset: Failed - {str(e)}")

    # Test with different n_splits
    try:
        cross_validate_random_forest(X, y, n_splits=5)
        print("Different n_splits: Passed")
    except Exception as e:
        print(f"Different n_splits: Failed - {str(e)}")

    # Test reproducibility
    pred1, true1, _, _, _, _, _ = cross_validate_random_forest(X, y, random_state=42)
    pred2, true2, _, _, _, _, _ = cross_validate_random_forest(X, y, random_state=42)
    is_reproducible = np.allclose(pred1, pred2)
    print(f"Reproducibility: {'Passed' if is_reproducible else 'Failed'}")    