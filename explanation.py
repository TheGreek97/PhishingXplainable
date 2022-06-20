import matplotlib.pyplot as plt
from lime import lime_tabular
from matplotlib import pyplot

from data import load_data
import warnings

import os
import pickle


def load_model(file_name):
    with open(os.path.join('output', file_name), 'rb') as file:
        return pickle.load(file)


def show_explanation(explanation, title=""):
    explanation_figure = explanation.as_pyplot_figure()
    explanation_figure.set_size_inches(20, 18)
    explanation_figure.set_dpi(100)
    plt.title(title)
    plt.show()


def explain(explainer, instance_to_explain, model, num_features):
    explanation = explainer.explain_instance(instance_to_explain, model.predict_proba, num_features=num_features)
    pred = model.predict_proba(instance_to_explain.values.reshape(1, -1))
    print(f"Predictions: Legit = {pred[0][0]}, Phishing = {pred[0][1]}")
    print(explanation.as_list())

    return explanation, pred


def explain_without_features(explainer, instance_to_explain, model, explanation, num_features, features_to_remove=2, val=1):
    tmp = instance_to_explain.copy()
    for i in range(0, features_to_remove):
        most_relevant_feature = explanation.as_list()[i][0].split(' ')[0]
        tmp.at[most_relevant_feature] = val
    return explain(explainer, instance_to_explain, model, num_features)


def save_explanation_to_file(explanation, file_name, folder_name=None):
    base_path = os.path.join('output', 'explanations')
    if folder_name is not None:
        base_path = os.path.join(base_path, folder_name)
        if not os.path.exists(base_path):
            os.makedirs(base_path)
    explanation.save_to_file(os.path.join(base_path, file_name+".html"))


if __name__ == "__main__":
    # Load data
    seed = 42
    x_training, x_test, y_training, y_test, feature_names = load_data(test_size=0.2, seed=seed)

    # Load models
    dt_model = load_model('decision_tree.obj')
    svm_model = load_model('svm.obj')
    rf_model = load_model('random_forest.obj')

    warnings.filterwarnings(action='ignore', category=UserWarning)

    # Decision tree feature importance based on the Gini index (Gini importance)
    # Importance of a feature is computed as the (normalized) total reduction of the criterion brought by that feature.
    importance = dt_model.feature_importances_
    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d (%s), Score: %.5f' % (i, feature_names[i], v))
    # plot feature importance
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()

    # LIME Explanations for single instances of the test set
    lime_explainer = lime_tabular.LimeTabularExplainer(x_training.values, mode="classification",
                                                       class_names=['Legit', 'Phishing'],
                                                       feature_names=feature_names)
    for i in range(0, 0):
        instance = x_test.iloc[i]
        real_class = 'Legit (0)' if y_test.iloc[i].item() == 0 else 'Phishing (1)'
        print("Instance " + str(i) + f" - Real class: {real_class}")

        folder_name = str(i)+"_"+str(y_test.iloc[i].item())
        # DECISION TREE
        explanation_dt, _ = explain(lime_explainer, instance, dt_model, len(feature_names))
        # show_explanation(explanation_dt, "Explanation DT")
        save_explanation_to_file(explanation_dt, 'decision_tree', folder_name=folder_name)

        # SVM
        explanation_svm, _ = explain(lime_explainer, instance, svm_model, len(feature_names))
        # show_explanation(explanation_svm, "Explanation SVM")
        save_explanation_to_file(explanation_svm, 'svm', folder_name=folder_name)

        # RANDOM FOREST
        explanation_rf, predictions_instance = explain(lime_explainer, instance, rf_model, len(feature_names))
        # show_explanation(explanation_rf, "Explanation RF")
        save_explanation_to_file(explanation_rf, 'random_forest', folder_name=folder_name)
        print("\n")

    # explanation_rf_1, predictions_instance_1 = explain_without_features(lime_explainer, instance,
    #                                                                    rf_model, explanation_rf,
    #                                                                    num_features=len(feature_names),
    #                                                                    features_to_remove=3)
    # save_explanation_to_file(explanation_rf_1, 'random_forest_1')
    # print('Difference:', predictions_instance[0] - predictions_instance_1[0])
