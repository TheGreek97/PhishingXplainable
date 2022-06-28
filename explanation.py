import matplotlib.pyplot as plt
from lime import lime_tabular
from matplotlib import pyplot
from heapq import nsmallest
from data import load_data
import warnings
import shap
import numpy as np
import os
import pickle
import json


def load_model(file_name):
    with open(os.path.join('output', file_name), 'rb') as file:
        return pickle.load(file)


def show_explanation(explanation, title=""):
    explanation_figure = explanation.as_pyplot_figure()
    explanation_figure.set_size_inches(20, 18)
    explanation_figure.set_dpi(100)
    plt.title(title)
    plt.show()


def explain(explainer, instance_to_explain, model, num_features, print_out=True):
    explanation = explainer.explain_instance(instance_to_explain, model.predict_proba, num_features=num_features)
    pred = model.predict_proba(instance_to_explain.values.reshape(1, -1))
    if print_out:
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


def tree_global_explanation(tree_model):
    # Decision tree feature importance based on the Gini index (Gini importance)
    # Importance of a feature is computed as the total reduction of the criterion brought by that feature weighted by
    # the probability of reaching that node (# of samples that reach the node, divided by the tot # of samples)
    # The higher the value, the more important the feature.
    importance = tree_model.feature_importances_
    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d (%s), Score: %.5f' % (i, feature_names[i], v))
    # plot decision tree feature importance
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()


# TODO
def tree_local_explanation(clf, x_test, start_index=0, end_index=10, n_top_features=3):
    node_indicator = clf.decision_path(x_test)
    n_nodes = clf.tree_.node_count
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    leaf_id = clf.apply(x_test)
    sample_ids = range(start_index, end_index)

    for sample_id in sample_ids:
        # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
        node_index = node_indicator.indices[
                     node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]
                     ]

        print("Rules used to predict sample {id}:\n".format(id=sample_id))
        print(f"Sample {id}:", x_test.iloc[sample_id])
        # node = 0  # start at root
        impurity_at_node = 1
        features_impurity_differences = {f: 0 for f in feature_names}
        for node_id in node_index:
            # continue to the next node if it is a leaf node
            feature_name = x_test.iloc[sample_id].keys()[feature[node_id]]  # the feature considered for the node split
            if leaf_id[sample_id] == node_id:
                continue
            # check if value of the split feature for sample 0 is below threshold
            if x_test.iloc[sample_id, feature[node_id]] <= threshold[node_id]:
                threshold_sign = "<="
                # node = clf.tree_.children_left[node]
            else:
                threshold_sign = ">"
                # node = clf.tree_.children_right[node]
            if node_id != 0:
                diff_impurity = clf.tree_.impurity[node_id] - impurity_at_node
                features_impurity_differences[feature_name] += diff_impurity
            impurity_at_node = clf.tree_.impurity[node_id]

            print(
                "decision node {node} : ({feature_name} = {value}) "
                "{inequality} {threshold})".format(
                    node=node_id,
                    sample=sample_id,
                    feature_name= feature_name,
                    feature=feature[node_id],
                    value=x_test.iloc[sample_id, feature[node_id]],
                    inequality=threshold_sign,
                    threshold=threshold[node_id],
                )
            )

        # For a group of samples, we can determine the common nodes the samples go through
        common_nodes = node_indicator.toarray()[sample_ids].sum(axis=0) == len(sample_ids)
        # obtain node ids using position in array
        common_node_id = np.arange(n_nodes)[common_nodes]
        print(f"\nThe following samples {sample_ids} share the node(s) {common_node_id} in the tree.")
        print("This is {prop}% of all nodes.".format(prop=100 * len(common_node_id) / n_nodes))
        top_features_gini_importance = nsmallest(n_top_features, features_impurity_differences,
                                                 key=features_impurity_differences.get)
        print(f"Top features are: {top_features_gini_importance}")
        return top_features_gini_importance


def lime_explain(model, lime_explainer, x_test, y_test, file_name='lime', start_index=0, end_index=10, show=True, save_file=True):
    warnings.filterwarnings(action='ignore', category=UserWarning)
    explanations = []
    predictions = []
    for i in range(start_index, end_index):
        instance = x_test.iloc[i]
        real_class = 'Legit (0)' if y_test.iloc[i].item() == 0 else 'Phishing (1)'
        print("Instance " + str(i) + f" - Real class: {real_class}")
        folder_name = str(i)+"_"+str(y_test.iloc[i].item())
        # DECISION TREE
        explanation_model, prediction = explain(lime_explainer, instance, model, len(feature_names))
        if show:
            show_explanation(explanation_model, "Explanation " + file_name)
        if save_file:
            save_explanation_to_file(explanation_model, file_name, folder_name=folder_name)
        explanations.append(explanation_model.as_list())
        predictions.append(prediction)
    return explanations, predictions


# Calcola percentuale di quanto ciascuna feature viene tenuta in conto durante le explanation di LIME e SHAP
# (e anche quello dell'albero). Il miglior modello è quello con più eterogeneità nelle features
def calculate_global_feature_importance(explanations, feature_names, file_name):
    feature_presence = {f: 0 for f in feature_names}
    for e in explanations:
        for i in range(0, 3):  # take the top 3 features
            top_feature = e[i][0].split(' ')[0]
            if top_feature[0].isdigit():
                top_feature = e[i][0].split(' ')[2]
            feature_presence[top_feature] += 1  # increase the feature by one if it is in the top 3 features
    base_path = os.path.join('output', 'feature_importance_lime')
    file_path = os.path.join(base_path, file_name+'.json')
    mode = 'w' if os.path.exists(file_path) else 'x'
    with open(file_path, mode) as out:
        out.write(json.dumps(feature_presence))


if __name__ == "__main__":
    # Load data
    seed = 42
    x_training, x_test, y_training, y_test, feature_names = load_data(test_size=0.2, seed=seed)

    # Load models
    dt_model = load_model('decision_tree.obj')
    svm_model = load_model('svm.obj')
    rf_model = load_model('random_forest.obj')
    lr_model = load_model('logistic_regression.obj')

    start_test = 0
    end_test = len(y_test)

    # tree_global_explanation(dt_model)
    # tree_local_explanation(dt_model, x_test, start_test, end_test)
    # LIME Explanations for single instances of the test set
    lime_explainer = lime_tabular.LimeTabularExplainer(x_training.values, mode="classification",
                                                       class_names=['Legit', 'Phishing'],
                                                       feature_names=feature_names, random_state=seed)

    # lime_explain(dt_model, lime_explainer, x_test, y_test, 'decision_tree', start_test, end_test)
    # lime_explain(svm_model, lime_explainer, x_test, y_test, 'svm', start_test, end_test)
    # lime_explain(rf_model, lime_explainer, x_test, y_test, 'random_forest', start_test, end_test)
    # lime_explain(lr_model, lime_explainer, x_test, y_test, 'logistic_regression', start_test, end_test)

    # explanation_rf_1, predictions_instance_1 = explain_without_features(lime_explainer, instance,
    #                                                                    rf_model, explanation_rf,
    #                                                                    num_features=len(feature_names),
    #                                                                    features_to_remove=3)
    # save_explanation_to_file(explanation_rf_1, 'random_forest_1')
    # print('Difference:', predictions_instance[0] - predictions_instance_1[0])

    # ----- SHAP -----
    X100 = shap.utils.sample(x_test, 100)
    explainer = shap.Explainer(svm_model.predict, X100)


"""
    # Global feature importance
    explanations_svm, _ = lime_explain(svm_model, lime_explainer, x_test, y_test, 'svm', start_test, end_test, show=False,
                                       save_file=False)
    calculate_global_feature_importance(explanations_svm, feature_names, 'svm')

    explanations_rf, _ = lime_explain(rf_model, lime_explainer, x_test, y_test, 'rf', start_test, end_test, show=False,
                                      save_file=False)
    calculate_global_feature_importance(explanations_rf, feature_names, 'rf')

    # explanations_lr, _ = lime_explain(lr_model, lime_explainer, x_test, y_test, 'lr', start_test, end_test, show=False,
                                      save_file=False)
    # calculate_global_feature_importance(explanations_lr, feature_names, 'lr')
"""