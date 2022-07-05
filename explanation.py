import matplotlib.pyplot as plt
from lime import lime_tabular
from matplotlib import pyplot
from heapq import nsmallest
from data import load_data
from tensorflow import keras
import warnings
import pandas as pd
import shap
import numpy as np
import os
import pickle
import json


def load_model(file_name):
    with open(os.path.join('models', file_name), 'rb') as file:
        return pickle.load(file)


def show_explanation(explanation, title=""):
    explanation_figure = explanation.as_pyplot_figure()
    explanation_figure.set_size_inches(20, 18)
    explanation_figure.set_dpi(100)
    plt.title(title)
    plt.show()


def explain(explainer, instance_to_explain, model, num_features, print_out=True):
    if hasattr(model, 'predict_proba'):
        explanation = explainer.explain_instance(instance_to_explain, model.predict_proba, num_features=num_features)
        pred = model.predict_proba(instance_to_explain.values.astype(int).reshape(1, -1))
    else:
        explanation = explainer.explain_instance(instance_to_explain, model.predict, num_features=num_features)
        pred = model.predict(instance_to_explain.values.astype(int).reshape(1, -1))

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


def tree_global_explanation_static(tree_model):
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


def tree_global_feature_importance_to_file(clf, x_test, feature_names, start_index=0, end_index=10, n_top_features=3,
                                           file_name='tree'):
    node_indicator = clf.decision_path(x_test)
    n_nodes = clf.tree_.node_count
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    leaf_id = clf.apply(x_test)
    sample_ids = range(start_index, end_index)
    feature_presence = {f: 0 for f in feature_names}
    for sample_id in sample_ids:
        # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
        node_index = node_indicator.indices[
                     node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]
                     ]

        # print("Rules used to predict sample {id}:\n".format(id=sample_id))
        # print(f"Sample {id}:", x_test.iloc[sample_id])
        # node = 0  # start at root
        impurity_at_node = 1
        features_impurity_differences = {f: 0 for f in feature_names}
        for node_id in node_index:
            # continue to the next node if it is a leaf node
            feature_name = x_test.iloc[sample_id].keys()[feature[node_id]]  # the feature considered for the node split
            if leaf_id[sample_id] == node_id:
                continue
            """# check if value of the split feature for sample 0 is below threshold
            if x_test.iloc[sample_id, feature[node_id]] <= threshold[node_id]:
                threshold_sign = "<="
                # node = clf.tree_.children_left[node]
            else:
                threshold_sign = ">"
                # node = clf.tree_.children_right[node]"""
            if node_id != 0:
                diff_impurity = clf.tree_.impurity[node_id] - impurity_at_node
                features_impurity_differences[feature_name] += diff_impurity
            impurity_at_node = clf.tree_.impurity[node_id]

            """print(
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
            )"""
        """"# For a group of samples, we can determine the common nodes the samples go through
        common_nodes = node_indicator.toarray()[sample_ids].sum(axis=0) == len(sample_ids)
        # obtain node ids using position in array
        common_node_id = np.arange(n_nodes)[common_nodes]
        print(f"\nThe following samples {sample_ids} share the node(s) {common_node_id} in the tree.")
        print("This is {prop}% of all nodes.".format(prop=100 * len(common_node_id) / n_nodes))"""
        top_features_gini_importance = nsmallest(n_top_features, features_impurity_differences,
                                                 key=features_impurity_differences.get)
        # print(f"Top features are: {top_features_gini_importance}")
        for f in top_features_gini_importance:
            feature_presence[f] += 1
        base_path = os.path.join('output', 'feature_importance')
        file_path = os.path.join(base_path, file_name + '.json')
        mode = 'w' if os.path.exists(file_path) else 'x'
        with open(file_path, mode) as out:
            out.write(json.dumps(feature_presence))
    return feature_presence


def explain_logistic_regression_static(model, feature_names):
    df = pd.DataFrame({'coeff': model.coef_[0]},
                      index=feature_names)
    print(df)


def lime_explain(model, lime_explainer, x_test, y_test, model_name='lime', start_index=0, end_index=10, show=False, save_file=False):
    warnings.filterwarnings(action='ignore', category=UserWarning)
    explanations = []
    predictions = []
    for i in range(start_index, end_index):
        instance = x_test.iloc[i]
        real_class = 'Legit (0)' if y_test.iloc[i].item() == 0 else 'Phishing (1)'
        print("Instance " + str(i) + f" - Real class: {real_class}")
        folder_name = str(i)+"_"+str(y_test.iloc[i].item())
        explanation_model, prediction = explain(lime_explainer, instance, model, len(feature_names))
        if show:
            show_explanation(explanation_model, "Explanation " + model_name)
        if save_file:
            save_explanation_to_file(explanation_model, model_name, folder_name=folder_name)
        explanations.append(explanation_model.as_list())
        predictions.append(prediction)
    return explanations, predictions


def lime_global_feature_importance_to_file(explanations, feature_names, file_name, n_top_features=3):
    """
    Calcola percentuale di quanto ciascuna feature viene tenuta in conto durante le explanation di LIME
       Il "miglior" modello è quello con più eterogeneità nelle features
    """
    feature_presence = {f: 0 for f in feature_names}
    for e in explanations:
        for i in range(0, n_top_features):  # take the top N features
            top_feature = e[i][0].split(' ')[0]
            if top_feature[0].isdigit() or top_feature[1].isdigit():
                top_feature = e[i][0].split(' ')[2]
            feature_presence[top_feature] += 1  # increase the feature by one if it is in the top 3 features
    base_path = os.path.join('output', 'feature_importance', 'lime')
    file_path = os.path.join(base_path, file_name+'.json')
    mode = 'w' if os.path.exists(file_path) else 'x'
    with open(file_path, mode) as out:
        out.write(json.dumps(feature_presence))


def shap_global_feature_importance(model, masker, x_test, feature_names, file_name, seed, print_summary=True, n_top_features=3, nn=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if nn:
            explainer = shap.KernelExplainer(model.predict, masker, seed=seed)
            shap_values = explainer.shap_values(x_test.astype(int), nsamples=500)
            shap_values = shap_values[1]
        else:
            explainer = shap.Explainer(model.predict, masker, seed=seed)
            shap_values = explainer(x_test.astype(int), silent=True)

    if print_summary:
        shap.summary_plot(shap_values, x_test.astype(int))
    feature_presence = {f: 0 for f in feature_names}
    for x in shap_values:
        if not nn:
            x = x.values
        x = abs(x)
        indexes_sort = np.argsort(x)
        for i in range(0, n_top_features):  # take the top N features
            top_feature = feature_names[indexes_sort[-i]]  # the indexes are the feats (-i => arr sort in asc. order)
            feature_presence[top_feature] += 1  # increase the feature by one if it is in the top 3 features
    base_path = os.path.join('output', 'feature_importance', 'shap')
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
    mlp_model = keras.models.load_model('models/mlp')
    dnn_model = keras.models.load_model('models/dnn')

    start_test = 0
    end_test = len(y_test)


    masker_med = x_training.median().values.reshape((1, x_training.shape[1]))
    shap_global_feature_importance(rf_model, masker_med, x_test, feature_names, 'rf', seed)

    # DECISION TREE
    tree_global_explanation_static(dt_model)
    tree_global_feature_importance_to_file(clf=dt_model, x_test=x_test, feature_names=feature_names,
                                           start_index=start_test, end_index=end_test, n_top_features=3)

    # Logistic Regression
    explain_logistic_regression_static(lr_model, feature_names)

    # ---- LIME - Global feature importance -----
    lime_explainer = lime_tabular.LimeTabularExplainer(x_training.values, mode="classification",
                                                       class_names=['Legit', 'Phishing'],
                                                       feature_names=feature_names, random_state=seed)
    # DECISION TREE
    explanations_dt, _ = lime_explain(dt_model, lime_explainer, x_test, y_test, 'dt',
                                      start_test, end_test, show=False, save_file=False)
    lime_global_feature_importance_to_file(explanations_dt, feature_names, 'dt')

    # LOGISTIC REGRESSION
    explanations_lr, _ = lime_explain(lr_model, lime_explainer, x_test, y_test, 'lr',
                                      start_test, end_test, show=False, save_file=False)
    lime_global_feature_importance_to_file(explanations_lr, feature_names, 'lr')

    # SVM
    explanations_svm, _ = lime_explain(svm_model, lime_explainer, x_test, y_test, 'svm',
                                       start_test, end_test, show=False, save_file=False)
    lime_global_feature_importance_to_file(explanations_svm, feature_names, 'svm')

    # RANDOM FOREST
    explanations_rf, _ = lime_explain(rf_model, lime_explainer, x_test, y_test, 'rf',
                                      start_test, end_test, show=False, save_file=False)
    lime_global_feature_importance_to_file(explanations_rf, feature_names, 'rf')

    # Multi-Layer Perceptron
    explanations_mlp, _ = lime_explain(mlp_model, lime_explainer, x_test, y_test, 'mlp',
                                       start_test, end_test, show=False, save_file=False)
    lime_global_feature_importance_to_file(explanations_mlp, feature_names, 'mlp')

    # Deep Neural Network
    explanations_dnn, _ = lime_explain(dnn_model, lime_explainer, x_test, y_test, 'dnn',
                                       start_test, end_test, show=False, save_file=False)
    lime_global_feature_importance_to_file(explanations_dnn, feature_names, 'dnn')

    # ----- SHAP - Global feature importance -----
    """# Local explanation
    X_idx = 2    
    X100 = shap.utils.sample(x_training, 100)
    explainer = shap.KernelExplainer(svm_model.predict, X100, link='identity')
    shap_values = explainer.shap_values(X=x_test.iloc[X_idx:X_idx+1, :], nsamples=100)
    shap.summary_plot(shap_values=shap_values, features=feature_names)"""

    masker_med = x_training.median().values.reshape((1, x_training.shape[1]))

    # DECISION TREE
    shap_global_feature_importance(dt_model, masker_med, x_test, feature_names, 'dt', seed)

    # LOGISTIC REGRESSION
    shap_global_feature_importance(lr_model, masker_med, x_test, feature_names, 'lr', seed)

    # SVM
    shap_global_feature_importance(svm_model, masker_med, x_test, feature_names, 'svm', seed)

    # RANDOM FOREST
    shap_global_feature_importance(rf_model, masker_med, x_test, feature_names, 'rf', seed)

    # Multi-Layer Perceptron
    shap_global_feature_importance(mlp_model, masker_med, x_test, feature_names, 'mlp', seed, print_summary=False, nn=True)

    # Deep Neural Network
    shap_global_feature_importance(dnn_model, masker_med, x_test, feature_names, 'dnn', seed, print_summary=False, nn=True)

    # TODO EBM (last)

