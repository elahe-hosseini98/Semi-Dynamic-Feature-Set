from sdfs.models import SDFS
from sdfs.train import train
from sdfs.dynamic_features import (
    init_dynamic_features_based_on_mean_and_variance, 
    init_dynamic_features_based_on_PCA, 
    init_randomly
)
import numpy as np
from sdfs.concatenate_feature_sets import concat_feature_sets
from sdfs.extend_test_samples import concat_dfs_to_test_samples

def sdfs(X_train, X_val, X_test, y_train, y_val, y_test, num_classes=None,
         dynamic_input_size=5, init_method='PCA', distance_method='minkowski'):
    """
    Semi-Dynamic Feature Set (SDFS) expansion and model training.

    Parameters:
    - X_train, X_val, X_test: numpy arrays or tensors for the train, validation, and test sets
    - y_train, y_val, y_test: labels for the train, validation, and test sets
    - dynamic_input_size: size of the dynamic feature vector to be concatenated
    - init_method: method to initialize dynamic features ('PCA', 'mean_std', 'random')
    - distance_method: method to calculate distances for test sample expansion ('euclidean', 'manhattan', 'minkowski', 'cosine')

    Returns:
    - extended_X_train, extended_X_val, extended_X_test: expanded versions of input datasets
    """
    
    if init_method == 'PCA':
        init_method = init_dynamic_features_based_on_PCA
    elif init_method == 'mean_std':
        init_method = init_dynamic_features_based_on_mean_and_variance
    else:
        init_method = init_randomly
    
    output_size = len(np.unique(y_train))
    
    dynamic_features = init_method(X_train, dynamic_input_size=dynamic_input_size)
    
    model = SDFS(static_input_size=X_train.shape[1], dynamic_input_size=dynamic_input_size, output_size=num_classes)
    
    final_train_acc, final_train_loss, loss_values, dynamic_features_trained = train(
        model, X_train, y_train, X_val, y_val, dynamic_features
        )
    
    extended_X_train = concat_feature_sets(X_train, dynamic_features_trained)
    extended_X_val = concat_dfs_to_test_samples(X_val, X_train, dynamic_features_trained)
    extended_X_test = concat_dfs_to_test_samples(X_test, X_train, dynamic_features_trained)
    
    return extended_X_train, extended_X_val, extended_X_test
    
    
    
