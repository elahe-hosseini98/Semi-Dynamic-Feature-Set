import torch
from sklearn.decomposition import PCA

def init_dynamic_features_based_on_mean_and_variance(X_train, dynamic_input_size):
    dynamic_features_list = []
    for i in range(len(X_train)):
        mean = torch.mean(X_train[i], dim=0)
        std = torch.std(X_train[i], dim=0)
        dynamic_features = torch.normal(mean=mean, std=std, size=(dynamic_input_size,))
        dynamic_features.requires_grad_(True)
        dynamic_features_list.append(dynamic_features)
        
    return dynamic_features_list

def init_dynamic_features_based_on_PCA(X_train, dynamic_input_size):
    pca = PCA(n_components=dynamic_input_size)
    pca_result = pca.fit_transform(X_train.numpy())
    
    dynamic_features_list = []
    for i in range(len(X_train)):
        dynamic_features = torch.tensor(pca_result[i], dtype=torch.float32).requires_grad_(True)
        dynamic_features_list.append(dynamic_features)
    
    return dynamic_features_list

def init_randomly(X_train, dynamic_input_size):
    dynamic_features_list = [torch.randn(dynamic_input_size, requires_grad=True) for _ in range(len(X_train))]
    return dynamic_features_list
