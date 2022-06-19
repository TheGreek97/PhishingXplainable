import matplotlib.pyplot as plt
from lime import lime_tabular
from training import load_data
import warnings
from sklearn.exceptions import DataConversionWarning

import os
import pickle

if __name__ == "__main__":
    # Load data
    seed = 42
    x_training, x_test, y_training, y_test, feature_names = load_data(test_size=0.2, seed=seed)

    # Load models
    with open(os.path.join('output', 'decision_tree.obj'), 'rb') as file:
        dt_model = pickle.load(file)
    with open(os.path.join('output', 'svm.obj'), 'rb') as file:
        svm_model = pickle.load(file)
    with open(os.path.join('output', 'random_forest.obj'), 'rb') as file:
        rf_model = pickle.load(file)

    warnings.filterwarnings(action='ignore', category=UserWarning)

    # LIME
    explainer = lime_tabular.LimeTabularExplainer(x_training.values, mode="classification",
                                                  class_names=['Legit', 'Phishing'],
                                                  feature_names=feature_names)

    # LIME Explanations
    instance_to_explain = x_test.iloc[0]
    """
    # DECISION TREE
    explanation_dt = explainer.explain_instance(instance_to_explain, dt_model.predict_proba,
                                                num_features=len(feature_names))
    explanation_figure = explanation_dt.as_pyplot_figure()
    explanation_figure.set_size_inches(20, 18)
    explanation_figure.set_dpi(100)
    plt.title("Explanation DT")
    # plt.show()
    print(explanation_dt.as_list())

    # SVM
    explanation_svm = explainer.explain_instance(instance_to_explain, svm_model.predict_proba,
                                                 num_features=len(feature_names))
    explanation_figure = explanation_svm.as_pyplot_figure()
    explanation_figure.set_size_inches(20, 18)
    explanation_figure.set_dpi(100)
    plt.title("Explanation SVM")
    # plt.show()
    print(explanation_svm.as_list())
    """

    # RANDOM FOREST
    explanation_rf = explainer.explain_instance(instance_to_explain, rf_model.predict_proba,
                                                num_features=len(feature_names))
    """explanation_figure = explanation_rf.as_pyplot_figure()
    explanation_figure.set_size_inches(20, 18)
    explanation_figure.set_dpi(100)
    plt.title("Explanation RF")
    plt.show()"""
    predictions_instance = rf_model.predict_proba(instance_to_explain.values.reshape(1, -1))
    print(explanation_rf.as_list())
    explanation_rf.save_to_file(os.path.join('output', 'explanations', 'random_forest_explanation.html'))

    #

    tmp = instance_to_explain.copy()
    for i in range(0, 3):
        most_relevant_feature = explanation_rf.as_list()[i][0].split(' ')[0]
        tmp.at[most_relevant_feature] = 1

    explanation_rf_1 = explainer.explain_instance(tmp, rf_model.predict_proba,
                                                  num_features=len(feature_names))
    tmp_pred = rf_model.predict_proba(tmp.values.reshape(1, -1))
    print(explanation_rf_1.as_list())
    print(f"Real class: {'Legit (0)' if y_test.iloc[0].item() == 0 else 'Phishing (1)'}")

    print(f"Predictions: Legit = {predictions_instance[0][0]}, Phishing = {predictions_instance[0][1]}")
    print(f'Predictions with some changes in features: Legit = {tmp_pred[0][0]}, Phishing = {tmp_pred[0][1]}')
    print('Difference:', predictions_instance[0] - tmp_pred[0])

    explanation_rf_1.save_to_file(os.path.join('output', 'explanations', 'random_forest_explanation_1.html'))
