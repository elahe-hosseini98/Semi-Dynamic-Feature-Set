import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from sdfs.feature_expansion import sdfs
from .classifier import Classifier, train, evaluate_model
import numpy as np
from dynamic_faetures_explainability.shap_analysis import shap_analysis
from dynamic_faetures_explainability.llm_feature_namer import suggest_feature_names
np.set_printoptions(threshold=np.inf)
import shap
from dynamic_faetures_explainability.analyze_feature_correlation import analyze_features


def load_wine_quality_data(test_size=0.2, validation_size=None, random_state=42):
    df = pd.read_csv(r'examples/winequalityN.csv')

    for col in df.columns[df.isnull().any()]:
        df[col] = df[col].fillna(df[col].mean())

    X = df.iloc[:, 1: -1].values
    y = (df.iloc[:, -1] > 6).astype(int).values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if validation_size:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size,
                                                          random_state=random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test

    return X_train, X_test, y_train, y_test


def run_example():
    X_train, X_val, X_test, y_train, y_val, y_test = load_wine_quality_data(test_size=0.1, validation_size=0.1)

    feature_names = ["fixed acidity", "volatile acidity", "citric acid",
                     "residual sugar", "chlorides", "free sulfur dioxide",
                     "total sulfur dioxide", "density", "pH", "sulphates", "alcohol",
                     "dynamic_feature_1", "dynamic_feature_2", "dynamic_feature_3",
                     "dynamic_feature_4", "dynamic_feature_5"]

    # Create a DataFrame from the tensor and assign appropriate column names
    static_feature_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                              'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                              'pH', 'sulphates', 'alcohol'] 
    dynamic_feature_columns = [f'Dynamic Feature {i}' for i in range(1, 5)]


    print('Performance without SDFS:')
    classifier = Classifier(X_train.shape[1], 2)
    train(classifier, X_train, y_train, X_val, y_val, num_epochs=100)
    
    
    evaluate_model(classifier, X_test, y_test)

    print("-" * 50 + '\n\n')


    expanded_X_train, expanded_X_val, expanded_X_test = sdfs(X_train, X_val, X_test,
                                                             y_train, y_val, y_test,
                                                             num_classes=2,
                                                             dynamic_input_size=4,
                                                             init_method='PCA',
                                                             distance_method='minkowski')

    dynamic_correlations = analyze_features(
                    expanded_X_train, expanded_X_val, expanded_X_test, y_train, y_val, y_test,
                    static_feature_columns=static_feature_columns,
                    dynamic_feature_columns=dynamic_feature_columns
                    )

    print('Performance with SDFS:')
    classifier = Classifier(expanded_X_train.shape[1], 2)
    train(classifier, expanded_X_train, y_train, expanded_X_val, y_val, num_epochs=100)

    dynamic_features_namer(expanded_X_train, dynamic_feature_size=4, model=classifier, dynamic_correlations=dynamic_correlations)

    shap_analysis(classifier, expanded_X_train, expanded_X_test, feature_names=feature_names)

    evaluate_model(classifier, expanded_X_test, y_test)



def dynamic_features_namer(expanded_X_train, dynamic_feature_size, model, dynamic_correlations):
    sub_samples = 100

    def model_predict(data):
        data = torch.tensor(data, dtype=torch.float32)
        with torch.no_grad():
            output = model(data)
            probabilities = torch.softmax(output, dim=1).numpy()
        return probabilities

    explainer = shap.KernelExplainer(model_predict, expanded_X_train[:sub_samples].numpy())

    shap_values = explainer.shap_values(expanded_X_train[:sub_samples].numpy())

    shap_values_class = shap_values[:, :, 1]  # Assuming class 1 represents 'good' wine

    # Compute mean SHAP values (with sign) and mean absolute SHAP values
    mean_shap_values = shap_values_class.mean(axis=0)
    mean_abs_shap_values = np.abs(shap_values_class).mean(axis=0)

    static_features_names = [
        "fixed acidity", "volatile acidity", "citric acid",
        "residual sugar", "chlorides", "free sulfur dioxide",
        "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
    ]

    dynamic_feature_indices = np.arange(1, dynamic_feature_size + 1)
    dynamic_features_names = [f"Dynamic Feature {i}" for i in dynamic_feature_indices]

    all_feature_names = static_features_names + dynamic_features_names

    feature_importance = list(zip(all_feature_names, mean_shap_values, mean_abs_shap_values))

    # Sort by the absolute value of mean SHAP values
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

    shap_summary = "\n".join(
        [f"- {name}: Mean SHAP Value = {mean_value:.4f}, Mean Absolute SHAP Value = {abs_value:.4f}"
         for name, mean_value, abs_value in feature_importance]
    )

    dataset_description = (
        "This dataset contains chemical attributes of wine samples,"
        " used to classify wines as 'good' or 'bad' based on their properties."
    )

    dynamic_features = expanded_X_train[:sub_samples, -dynamic_feature_size:].cpu().numpy()
    static_features = expanded_X_train[:sub_samples, :-dynamic_feature_size].cpu().numpy()

    dynamic_features_mean_values = dynamic_features.mean(axis=0)
    static_features_mean_values = static_features.mean(axis=0)

    prompt = f"""
        I am working with a wine classification dataset that includes various chemical properties of wine samples, with the goal of classifying wines as either 'good' or 'bad' based on these properties. 

        Here is a description of the dataset:
        {dataset_description}

        **Static Features:** These original, normalized features include:
        {', '.join(static_features_names)}

        **Dynamic Features:** I have developed an approach called "feature expansion" to create these new, normalized dynamic features. They are derived by analyzing and transforming the patterns within the static features. 

        For reference, here are additional details about the first {sub_samples} samples within the extended version of the X_train set:
        - **Static Features Data Sample:** {static_features}
        - **Dynamic Features Data Sample:** {dynamic_features}
        - **Static Feature Mean Values:** {static_features_mean_values}
        - **Dynamic Feature Mean Values:** {dynamic_features_mean_values}

        **Dynamic Features Correlations with Static Features:**
        The following are the correlations between each dynamic feature and the static features:
        {dynamic_correlations}

        **SHAP Analysis of All Features:**
        The following are the mean SHAP values for all features, indicating both the direction and magnitude of their average contribution to the model's predictions for class 1 meaning 'good' wines:
        {shap_summary}

        Please assign meaningful, professional names to each of these dynamic features, considering their importance as indicated by the SHAP values and their correlations with the static features. The names should reflect the potential influence of each feature on wine quality classification, specifically based on their values, their correlations with static features, and their impact on the model's predictions. Ensure that each name is distinct from the original static feature names, yet remains relevant to wine chemistry, quality, and classification.
        """

    #print(prompt)

    print(suggest_feature_names(prompt=prompt))

