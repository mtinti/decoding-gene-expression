import pandas as pd
import numpy as np

def recursive_feature_elimination(df, threshold=0.95):
    features_to_drop = []
    
    while True:
        # Calculate correlation matrix
        cor_matrix = df.corr().abs()
        
        # Create upper triangle mask
        upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
        
        # Find maximum correlation
        max_corr = upper_tri.max().max()
        
        if max_corr < threshold:
            break
            
        # Find feature to drop (we'll drop the one that has the highest mean correlation with other features)
        # Get the feature that has the highest correlation with other features
        max_corr_feature = upper_tri.max().idxmax()
        
        print(f"Dropping {max_corr_feature} (max correlation: {max_corr:.3f})")
        features_to_drop.append(max_corr_feature)
        df = df.drop(columns=[max_corr_feature])
        
    return df.columns.tolist(), features_to_drop


def test_recursive_feature_elimination():
    # Create test data
    np.random.seed(42)
    n = 100
    
    df = pd.DataFrame({
        'a': np.random.randn(n),
        'b': np.random.randn(n),
        'c': np.random.randn(n),
        'd': np.random.randn(n),
        
    })
    # Make 'b' highly correlated with 'a'
    df['b'] = df['a'] + np.random.randn(n) * 0.5
    df['c'] = df['b'] + np.random.randn(n) * 0.1
    
    print(df.corr())
    
    print('Test case 1: Basic functionality')
    remaining_features, dropped_features = recursive_feature_elimination(df, threshold=0.95)
    print(dropped_features)
    assert len(dropped_features) == 1

    print('Test case 2: No correlations above threshold')
    remaining_features, dropped_features = recursive_feature_elimination(df, threshold=0.999)
    print(dropped_features)
    assert len(dropped_features) == 0
    #assert len(remaining_features) == 3

    print('Test case 3: 2 correlations above threshold')
    remaining_features, dropped_features = recursive_feature_elimination(df, threshold=0.8)
    print(dropped_features)
    assert len(dropped_features) == 2
    