import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import torch


def analyze_features(expanded_X_train, expanded_X_val, expanded_X_test, y_train, y_val, y_test,
                     static_feature_columns, dynamic_feature_columns):

    X = torch.cat([expanded_X_train, expanded_X_val, expanded_X_test], dim=0)
    y = torch.cat([y_train, y_val, y_test], dim=0)

    if not isinstance(X, pd.DataFrame):
        # Determine the number of static and dynamic features
        num_total_features = X.shape[1]
        num_dynamic_features = len(dynamic_feature_columns)
        num_static_features = num_total_features - num_dynamic_features

        feature_columns = static_feature_columns + dynamic_feature_columns
        X = pd.DataFrame(X.numpy(), columns=feature_columns)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)


    feature_importance = pd.Series(rf.feature_importances_, index=X.columns)

    static_features = feature_importance[static_feature_columns]
    dynamic_features = feature_importance[dynamic_feature_columns]

    plt.rcParams.update({'font.size': 14})

    plt.figure(figsize=(8, 10))
    plt.barh(static_features.index, static_features.values, color='skyblue', label='Static Features')
    plt.barh(dynamic_features.index, dynamic_features.values, color='lightcoral', label='Dynamic Features')
    plt.xlabel('Importance Score', fontsize=18)
    plt.ylabel('Features', fontsize=18)
    plt.title('Feature Importance: Static vs. Dynamic Features\n(Wine Quality Dataset)', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 10))

    heatmap = sns.heatmap(X.corr(), annot=True, fmt=".1f", cmap='coolwarm', annot_kws={"size": 12},
                          cbar_kws={"shrink": 0.8})

    for dynamic_feature in dynamic_feature_columns:
        idx = X.columns.get_loc(dynamic_feature)
        heatmap.add_patch(plt.Rectangle((idx, 0), 1, len(X.columns), fill=False, edgecolor='yellow', lw=3))
        heatmap.add_patch(plt.Rectangle((0, idx), len(X.columns), 1, fill=False, edgecolor='yellow', lw=3))

    plt.title('Correlation Heatmap of Expanded Feature Set\n(Wine Quality Dataset)', fontsize=20)
    plt.xticks(fontsize=14, rotation=90)
    plt.yticks(fontsize=14, rotation=0)
    plt.tight_layout()
    plt.show()

    corr_matrix = X.corr()

    dynamic_static_correlations = corr_matrix.loc[dynamic_feature_columns, static_feature_columns]

    dynamic_correlations_list = []
    for dynamic_feature in dynamic_feature_columns:
        correlations = dynamic_static_correlations.loc[dynamic_feature].to_dict()
        dynamic_correlations_list.append({dynamic_feature: correlations})

    return dynamic_correlations_list
