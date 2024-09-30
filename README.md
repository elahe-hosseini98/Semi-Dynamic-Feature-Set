# Semi-Dynamic Feature Set (SDFS)

The **Semi-Dynamic Feature Set (SDFS)** is a novel approach to combining static and dynamic features in machine learning models. This package provides an implementation of the SDFS model along with utilities for expanding feature sets using dynamic methods like PCA and mean-variance initialization.

This method is designed to enhance machine learning models by dynamically updating feature sets, which is particularly useful for improving performance in relevant scenarios such as classification tasks.

## Features
- Implements the SDFS model for combining static and dynamic features.
- Supports dynamic feature initialization using:
  - Principal Component Analysis (PCA)
  - Mean and Variance based initialization
  - Random initialization
- Includes methods for:
  - Training models with dynamic features
  - Expanding test datasets based on closest dynamic feature matches
  - Concatenating feature sets for better model performance.

## Installation

To install the SDFS package, use `pip`:

```bash
pip install sdfs
```
## Usage

Once installed, you can use the `sdfs` function to expand your feature sets and train the model. Here’s a simple usage example:

```python
from sdfs.feature_expansion import sdfs

# Assuming you have train, validation, and test sets ready:
extended_X_train, extended_X_val, extended_X_test = sdfs(
    X_train, X_val, X_test, y_train, y_val, y_test, 
    dynamic_input_size=5, init_method='PCA', distance_method='minkowski'
)
```
### Parameters:

- **X_train, X_val, X_test**: Training, validation, and test feature sets.
- **y_train, y_val, y_test**: Corresponding labels for the feature sets.
- **dynamic_input_size**: The size of the dynamic feature vector to be concatenated.
- **init_method**: Method to initialize dynamic features, choose from:
  - `PCA`: Principal Component Analysis for dimensionality reduction.
  - `mean_std`: Initialization based on mean and variance of training features.
  - `random`: Random initialization of dynamic features.
- **distance_method**: Method for calculating distances to dynamically expand test sets, choose from:
  - `euclidean`
  - `manhattan`
  - `minkowski`
  - `cosine`
## Example Workflow

Here’s an example of how to use the SDFS package in a typical workflow:

1. **Initialize dynamic features** using PCA or other methods.
2. **Expand the feature sets** by concatenating the static and dynamic features.
3. **Train the SDFS model** on the expanded feature sets.
4. **Evaluate** the model on validation and test sets by expanding those feature sets similarly.

```python
# Example usage
from sdfs.feature_expansion import sdfs

# Define your datasets
X_train, X_val, X_test = ...
y_train, y_val, y_test = ...

# Expand feature sets and train the model
expanded_X_train, expanded_X_val, expanded_X_test = sdfs(
    X_train, X_val, X_test, y_train, y_val, y_test, num_clsses=num_clsses,
    dynamic_input_size=10, init_method='PCA', distance_method='minkowski'
)
```
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

## Acknowledgements

This package was implemented for a relevant research project, aimed at exploring semi-dynamic feature sets to enhance the flexibility and accuracy of machine learning models.
