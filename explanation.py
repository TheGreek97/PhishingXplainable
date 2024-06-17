import winsound

import matplotlib.pyplot as plt
from lime import lime_tabular
from matplotlib import pyplot
from heapq import nsmallest
from data import load_data_no_split, train_test_split
from tensorflow import keras
from training import displayConfusionMatrix, h_score_loss

import ebm
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


def compute_h_loss(explanation_list, shap=False):
    n_instances = len(explanation_list)
    n_features = len(explanation_list[0])
    features_sum = np.zeros(n_features)
    for e in explanation_list:
        for i in range(0, n_features):
            if shap:
                features_sum[i] += e[i]
            else:
                features_sum[i] += e[i][1]
    avg_importance = [feature / n_instances for feature in features_sum]

    loss = h_score_loss(avg_importance, alpha=0.5)
    return loss


def save_explanation_to_file(explanation, file_name, folder_name=None):
    base_path = os.path.join('output', 'explanations')
    if folder_name is not None:
        base_path = os.path.join(base_path, folder_name)
        if not os.path.exists(base_path):
            os.makedirs(base_path)
    explanation.save_to_file(os.path.join(base_path, file_name + ".html"))


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


def lime_explain_instance(instance_features, model, x_stats, feature_names, verbose=False):
    num_features = len(feature_names)
    lime_explainer = lime_tabular.LimeTabularExplainer(x_stats, mode="classification",
                                                       class_names=['Legit', 'Phishing'],
                                                       feature_names=feature_names, random_state=seed)
    if hasattr(model, 'predict_proba'):
        explanation = lime_explainer.explain_instance(instance_features, model.predict_proba, num_features=num_features)
    else:
        explanation = lime_explainer.explain_instance(instance_features, model.predict, num_features=num_features)
    if verbose:
        print(explanation.as_list())
    return explanation


def shap_explain_instance(instance_features, model, x_training, feature_names, verbose=False):
    X100 = shap.utils.sample(x_training, 100)
    explainer = shap.KernelExplainer(model.predict, X100, link='identity')
    shap_values = explainer.shap_values(X=instance_features, nsamples=100)
    if verbose:
        shap.summary_plot(shap_values=shap_values, features=feature_names)
    return shap_values


def lime_explain(model, lime_explainer, x_test, y_test, model_name='lime', start_index=0, end_index=10, show=False,
                 save_file=False, verbose=False):
    warnings.filterwarnings(action='ignore', category=UserWarning)
    explanations = []
    predictions = []
    for i in range(start_index, end_index):
        real_class = 'Legit (0)' if y_test.iloc[i].item() == 0 else 'Phishing (1)'
        print("Instance " + str(i) + f" - Real class: {real_class}")
        num_features = len(feature_names)
        if hasattr(model, 'predict_proba'):
            prediction = model.predict_proba(x_test.iloc[i].values.astype(int).reshape(1, -1))
            explanation = lime_explainer.explain_instance(x_test.iloc[i], model.predict_proba, num_features=num_features)
        else:
            prediction = model.predict(x_test.iloc[i].values.astype(int).reshape(1, -1))
            explanation = lime_explainer.explain_instance(x_test.iloc[i], model.predict, num_features=num_features)
        if verbose:
            print(f"Predictions: Legit = {prediction[0][0]}, Phishing = {prediction[0][1]}")
            print(explanation.as_list())
        if show:
            explanation_figure = explanation.as_pyplot_figure()
            explanation_figure.set_size_inches(20, 18)
            explanation_figure.set_dpi(100)
            plt.title("Explanation " + model_name)
            plt.show()
        if save_file:
            folder_name = str(i) + "_" + str(y_test.iloc[i].item())
            save_explanation_to_file(explanation, model_name, folder_name=folder_name)
        explanations.append(explanation.as_list())
        predictions.append(prediction)
    return explanations  # , predictions


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
    file_path = os.path.join(base_path, file_name + '.json')
    mode = 'w' if os.path.exists(file_path) else 'x'
    with open(file_path, mode) as out:
        out.write(json.dumps(feature_presence))


def shap_explain(model, masker, x_test, print_summary=True, nn=False):
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
    return shap_values


def shap_global_feature_importance_to_file(explanations, feature_names, file_name, n_top_features=3, nn=False):
    feature_presence = {f: 0 for f in feature_names}
    for x in explanations:
        if not nn:
            x = x.values
        x = abs(x)
        indexes_sort = np.argsort(x)
        for i in range(1, n_top_features + 1):  # take the top N features
            top_feature = feature_names[indexes_sort[-i]]  # the indexes are the feats (-i => arr sort in asc. order)
            feature_presence[top_feature] += 1  # increase the feature by one if it is in the top 3 features
    base_path = os.path.join('output', 'feature_importance', 'shap')
    file_path = os.path.join(base_path, file_name + '.json')
    mode = 'w' if os.path.exists(file_path) else 'x'
    with open(file_path, mode) as out:
        out.write(json.dumps(feature_presence))


if __name__ == "__main__":
    # Load data
    seed = 42

    X, y, feature_names = load_data_no_split()

    execute_decision_tree = False
    execute_logistic_regression = False
    execute_svm = False
    execute_random_forest = True
    execute_mlp = False
    execute_dnn = False
    execute_ebm = False

    start_test = 0
    end_test = 5  # len(y)

    # ---- LIME - Global feature importance -----
    x_training, _, _, _ = train_test_split(X, y, stratify=y, test_size=0.2, random_state=seed)
    lime_explainer = lime_tabular.LimeTabularExplainer(x_training.values, mode="classification",
                                                       class_names=['Legit', 'Phishing'],
                                                       feature_names=feature_names, random_state=seed)
    # ----- SHAP - Global feature importance -----
    """# Local explanation
    X_idx = 2    
    X100 = shap.utils.sample(x_training, 100)
    explainer = shap.KernelExplainer(svm_model.predict, X100, link='identity')
    shap_values = explainer.shap_values(X=X.iloc[X_idx:X_idx+1, :], nsamples=100)
    shap.summary_plot(shap_values=shap_values, features=feature_names)"""
    masker_med = x_training.median().values.reshape((1, x_training.shape[1]))

    # DECISION TREE
    if execute_decision_tree:
        dt_model = load_model('decision_tree.obj')
        tree_global_explanation_static(dt_model)
        tree_global_feature_importance_to_file(clf=dt_model, x_test=X, feature_names=feature_names,
                                               start_index=start_test, end_index=end_test, n_top_features=3)
        # LIME
        explanations = lime_explain(dt_model, lime_explainer, X, y, 'dt',
                                    start_test, end_test, show=False, save_file=False)
        lime_global_feature_importance_to_file(explanations, feature_names, 'dt')
        # SHAP
        explanations = shap_explain(dt_model, masker_med, X, print_summary=True, nn=False)
        shap_global_feature_importance_to_file(explanations, feature_names, 'dt', nn=False)

    # LOGISTIC REGRESSION
    if execute_logistic_regression:
        lr_model = load_model('logistic_regression.obj')
        explain_logistic_regression_static(lr_model, feature_names)
        # LIME
        explanations = lime_explain(lr_model, lime_explainer, X, y, 'lr',
                                    start_test, end_test, show=False, save_file=False)
        lime_global_feature_importance_to_file(explanations, feature_names, 'lr')
        # SHAP
        explanations = shap_explain(lr_model, masker_med, X, print_summary=True, nn=False)
        shap_global_feature_importance_to_file(explanations, feature_names, 'lr', nn=False)

    # SVM
    if execute_svm:
        svm_model = load_model('svm.obj')
        # LIME
        explanations = lime_explain(svm_model, lime_explainer, X, y, 'svm',
                                    start_test, end_test, show=False, save_file=False)
        lime_global_feature_importance_to_file(explanations, feature_names, 'svm')
        # SHAP
        explanations = shap_explain(svm_model, masker_med, X, print_summary=True, nn=False)
        shap_global_feature_importance_to_file(explanations, feature_names, 'svm', nn=False)

    # RANDOM FOREST
    if execute_random_forest:
        rf_model = load_model('random_forest.obj')
        # LIME
        explanations = lime_explain(rf_model, lime_explainer, X, y, 'rf',
                                    start_test, end_test, show=False, save_file=False)
        lime_global_feature_importance_to_file(explanations, feature_names, 'rf')
        h_score = 1 / compute_h_loss(explanations)
        print(f"H score LIME RF: {h_score}")
        # SHAP
        explanations = shap_explain(rf_model, masker_med, X, print_summary=True, nn=False)
        shap_global_feature_importance_to_file(explanations, feature_names, 'rf', nn=False)
        h_score = 1 / compute_h_loss(explanations.values, shap=True)
        print(f"H score SHAP RF: {h_score}")

    # Multi-Layer Perceptron
    if execute_mlp:
        mlp_model = keras.models.load_model('models/mlp')
        # LIME
        explanations = lime_explain(mlp_model, lime_explainer, X, y, 'mlp',
                                    start_test, end_test, show=False, save_file=False)
        lime_global_feature_importance_to_file(explanations, feature_names, 'mlp')
        # SHAP
        explanations = shap_explain(mlp_model, masker_med, X, print_summary=True, nn=True)
        shap_global_feature_importance_to_file(explanations, feature_names, 'mlp', nn=True)

    # Deep Neural Network
    if execute_dnn:
        dnn_model = keras.models.load_model('models/dnn')
        # LIME
        explanations = lime_explain(dnn_model, lime_explainer, X, y, 'dnn',
                                    start_test, end_test, show=False, save_file=False)
        lime_global_feature_importance_to_file(explanations, feature_names, 'dnn')
        # SHAP
        explanations = shap_explain(dnn_model, masker_med, X, print_summary=True, nn=True)
        shap_global_feature_importance_to_file(explanations, feature_names, 'dnn', nn=True)

    # ---- EBM -----
    if execute_ebm:
        ebm_model = load_model('ebm.obj')
        ebm_global_explanation = ebm_model.explain_global()
        ebm_local = ebm_model.explain_local(X.iloc[:4], y.iloc[:4])
        ebm_local.visualize().write_html('output/explanations/ebm/4.html')
        ebm_local_explanation = ebm.ebm_global_feature_importance(ebm_model, x_test=X, y_test=y,
                                                                  feature_names=feature_names)
    duration = 1000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)
