import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from sdfs.feature_expansion import sdfs


def load_wine_quality_data(test_size=0.2, validation_size=None, random_state=42):
    
    df = pd.read_csv(r'examples/winequalityN.csv')
    
    for col in df.columns[df.isnull().any()]:
        df[col] =  df[col].fillna(df[col].mean())
        
    X = df.iloc[:, 1: -1].values
    y = (df.iloc[:, -1] > 6).astype(int).values  # Binary classification: Good (1) or not good (0)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    if validation_size:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    return X_train, X_test, y_train, y_test


def run_example():
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_wine_quality_data(test_size=0.1, validation_size=0.1)
    
    expanded_X_train, expanded_X_val, expanded_X_test = sdfs(X_train, X_val, X_test, 
                                                             y_train, y_val, y_test,
                                                             num_classes=2,
                                                             dynamic_input_size=10,
                                                             init_method='PCA',
                                                             distance_method='minkowski')

    print(expanded_X_train.shape, expanded_X_val.shape, expanded_X_test.shape)
