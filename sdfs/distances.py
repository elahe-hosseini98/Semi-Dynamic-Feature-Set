import torch
from sklearn.metrics.pairwise import cosine_similarity

def find_closest_dynamic_features(X_train, test_sample, dynamic_features_list, method="euclidean", p=3):
    if method == "euclidean":
        distances = torch.norm(X_train - test_sample, dim=1)
        
    elif method == "manhattan":
        distances = torch.sum(torch.abs(X_train - test_sample), dim=1)
        
    elif method == "minkowski":
        distances = torch.sum(torch.abs(X_train - test_sample) ** p, dim=1) ** (1 / p)
        
    elif method == "cosine":
        similarities = cosine_similarity(X_train, test_sample.unsqueeze(0))
        distances = torch.tensor(similarities)
        return dynamic_features_list[torch.argmax(distances)]
    
    else:
        raise ValueError("Unknown method: {}".format(method))
    
    closest_index = torch.argmin(distances)
    return dynamic_features_list[closest_index]
