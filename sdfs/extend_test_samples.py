from sdfs.distances import find_closest_dynamic_features
import numpy as np
import torch

def concat_dfs_to_test_samples(X_test, X_train, dynamic_features, distance_method="minkowski"):
    
    extended_X_test = []
    
    for i in range(X_test.shape[0]):
        
        test_sample = X_test[i]
        closest_dynamic_features = find_closest_dynamic_features(X_train, test_sample, dynamic_features, method=distance_method)
        concatenated_feature_set = np.concatenate((test_sample, closest_dynamic_features), axis=0)
        extended_X_test.append(concatenated_feature_set)
        
    return torch.tensor(extended_X_test)