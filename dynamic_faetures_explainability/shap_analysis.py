import shap
import torch
import numpy as np


def shap_analysis(model, X_train, X_test, feature_names, sub_samples=100):
    model.eval()

    # Convert PyTorch model to a function that outputs numpy arrays (necessary for SHAP)
    def model_predict(data):
        data = torch.tensor(data, dtype=torch.float32)
        with torch.no_grad():
            output = model(data)
            probabilities = torch.softmax(output, dim=1).numpy()
        return probabilities

    explainer = shap.KernelExplainer(model_predict, X_train[:sub_samples].numpy())

    # Calculate SHAP values (This will return a list of arrays, one for each class)
    shap_values = explainer.shap_values(X_test[:sub_samples].numpy())

    print("Original SHAP values shape:", shap_values.shape)
    print("X_test_sample shape:", X_test[:sub_samples].shape)
    print("Feature names length:", len(feature_names))

    for class_idx in range(shap_values.shape[2]):
        print(f"Plotting SHAP values for class {class_idx}")
        shap.summary_plot(shap_values[:, :, class_idx], X_test[:sub_samples].numpy(), feature_names=feature_names)

    # Plot aggregated SHAP values across both classes
    shap_values_mean = np.mean(shap_values, axis=2)
    print("Plotting aggregated SHAP values across all classes")
    shap.summary_plot(shap_values_mean, X_test[:sub_samples].numpy(), feature_names=feature_names)