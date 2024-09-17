import torch
import torch.nn as nn
import torch.optim as optim
from sdfs.early_stopping import EarlyStopping 
from sdfs.distances import find_closest_dynamic_features

def train(model, X_train, y_train, X_val, y_val, dynamic_features_list, 
          num_epochs=50, learning_rate=0.001, weight_decay=1e-5, patience=5):
    
    criterion = nn.CrossEntropyLoss()
    # Adding L2 regularization (weight decay)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_values = []
    
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    print("Training Semi-Dynamic Feature Set:")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        
        for i in range(len(X_train)):
            static_input = X_train[i].unsqueeze(0)
            label = y_train[i].unsqueeze(0)
            
            # Prepare dynamic features
            dynamic_features = dynamic_features_list[i].unsqueeze(0).clone().detach().requires_grad_(True)
            
            optimizer.zero_grad()
            output = model(static_input, dynamic_features)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            # Update dynamic features manually
            with torch.no_grad():
                dynamic_features -= dynamic_features.grad

            dynamic_features_list[i] = dynamic_features.detach().squeeze(0).clone()
            
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == label).sum().item()
            total_loss += loss.item()
        
        accuracy = correct / len(X_train)
        avg_loss = total_loss / len(X_train)
        loss_values.append(avg_loss)
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Accuracy: {accuracy * 100:.2f}%')

        val_loss, val_accuracy = validate(model, X_train, X_val, y_val, dynamic_features_list, criterion)
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%')
        print('-' * 10)
       
        # Check early stopping condition
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
    
    print("Training has been completed.")
    return accuracy * 100, avg_loss, loss_values, dynamic_features_list


def validate(model, X_train, X_val, y_val, dynamic_features_list, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for i in range(len(X_val)):
            static_input = X_val[i].unsqueeze(0)
            label = y_val[i].unsqueeze(0)
            dynamic_features = find_closest_dynamic_features(X_train, X_val[i], dynamic_features_list, method="minkowski").unsqueeze(0)

            output = model(static_input, dynamic_features)
            loss = criterion(output, label)
            total_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            correct += (predicted == label).sum().item()

    avg_loss = total_loss / len(X_val)
    accuracy = correct / len(X_val)
    
    return avg_loss, accuracy
