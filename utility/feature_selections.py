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
