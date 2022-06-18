import pandas as pd
import numpy as np

import os
import pickle

import matplotlib.pyplot as plt
import sklearn.tree as tree
import sklearn.svm as svm

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from lime import lime_tabular


def load_data():
    dfs = []  # an empty list to store the data frames

    for path in ['datasets\\features\\enron', 'datasets\\features\\spam_assassin']:
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            data_ = pd.read_json(file_path, lines=True)  # read data frame from json file
            try:
                data_.iat[0, 16] = sum(data_.iat[0, 16].values())
                dfs.append(data_)  # append the data frame to the list
            except IndexError:
                print("Ignoring ", file)

    dataframe = pd.concat(dfs, ignore_index=True)  # concatenate all the data frames in the list.
    return dataframe


def stratifiedKFold(data_x, data_y, n_folds, seed):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    x_train_array = []
    x_test_array = []
    y_train_array = []
    y_test_array = []
    for train_index, test_index in skf.split(data_x, data_y):
        x_train_array.append(data_x.iloc[train_index])
        x_test_array.append(data_x.iloc[test_index])
        y_train_array.append(data_y.iloc[train_index])
        y_test_array.append(data_y.iloc[test_index])
    return x_train_array, x_test_array, y_train_array, y_test_array


def showTree(model):
    plt.figure(figsize=(200, 20))
    tree.plot_tree(model)
    print("Number of nodes:", model.tree_.node_count)
    print("Number of leaves:", model.get_n_leaves())
    plt.show()


def decisionTreeF1(x, y, model):
    predictions = model.predict(x)
    return f1_score(y_true=y, y_pred=predictions, average='weighted')


# Decision Tree Hyper-parameter tuning
def determineDTkFoldConfiguration(x_train, x_val, y_train, y_val, seed=0):
    best_criterion = 'gini'
    best_alpha = 0
    best_f1 = 0
    n_folds = len(x_train)
    for criterion in ['gini', 'entropy']:
        for alpha in np.arange(0, 0.5, 0.001):
            f1_scores = [0, 0, 0, 0, 0]
            print(f"Computing criterion={criterion}, alpha={alpha}")
            for k in range(0, n_folds):
                t = tree.DecisionTreeClassifier(criterion=criterion, ccp_alpha=alpha, random_state=seed)
                t.fit(x_train[k], y_train[k])
                f1_scores[k] = decisionTreeF1(x=x_val[k], y=y_val[k], model=t)
            avg_f1 = sum(f1_scores) / n_folds
            print(f"Average f1: {avg_f1}")
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_alpha = alpha
                best_criterion = criterion

    return best_alpha, best_criterion


# SVM Hyper-parameter tuning
def determineSVMkFoldConfiguration(x_train, x_val, y_train, y_val, seed=0):
    best_kernel = 'linear'
    best_c = 0.1
    best_gamma = 'scale'
    best_tol = 0.1
    best_degree = 3
    best_score = 0
    n_folds = len(x_train)
    for kernel in ["linear", "poly", "rbf", "sigmoid"]:
        poly_degrees = range(2, 7) if kernel == "poly" else [3]
        gammas = [0.1, 1, 'auto'] if kernel in {"rbf", "poly", "sigmoid"} else ['scale']
        for c in [0.1, 0.5, 1, 10]:
            for gamma in gammas:
                for tol in [0.1, 0.01, 0.005, 0.001]:
                    for degree in poly_degrees:
                        scores = [0, 0, 0, 0, 0]
                        print(f"Computing kernel={kernel}, c={c}, gamma={gamma}, tol={tol}, degree={degree}")
                        for k in range(0, n_folds):
                            model = make_pipeline(StandardScaler(), svm.SVC(gamma=gamma, tol=tol, coef0=c,
                                                                            degree=degree, kernel=kernel,
                                                                            random_state=seed))
                            model.fit(x_train[k], y_train[k].values.ravel())
                            scores[k] = model.score(x_val[k], y_val[k].values.ravel())
                        avg_score = sum(scores) / n_folds
                        print(avg_score)
                        if avg_score > best_score:
                            best_kernel = kernel
                            best_c = c
                            best_gamma = gamma
                            best_tol = tol
                            best_degree = degree
                            best_score = avg_score
    return {"kernel": best_kernel, "c": best_c, "gamma": best_gamma, "tol": best_tol, "degree": best_degree}


# Random Forest Hyper-parameter tuning
def determineRFkFoldConfiguration(x_train, x_val, y_train, y_val, seed=0):
    best_n = 10
    best_bootstrap = True
    best_max_depth = 3
    best_min_samples_split = 10
    best_max_leaf_nodes = 5
    best_min_samples_leaf = 10
    best_max_features = 3
    best_max_samples = 1
    best_score = 0
    n_folds = len(x_train)
    # total = 50k random forests
    for n in [10, 25, 50, 75, 100]:
        for bootstrap in [True, False]:
            max_samples_values = [0.1, 0.25, 0.5, 0.7] if bootstrap else [None]
            for max_depth in [3, 5, 7, 9]:  # A very deep tree can cause over-fitting
                for min_samples_split in [10, 25, 50, 100]:  # The min. observations in any node to split it
                    for max_leaf_nodes in [5, 15, 25, 35]:  # The max. leaves in a tree
                        for min_samples_leaf in [10, 25, 50, 100]:  # The min. samples in a leaf after a split
                            for max_features in [3, 7, 10, 14, 17]:
                                for max_samples in max_samples_values:
                                    print(f"Computing n={n}, max_depth={max_depth}, min_sample_split={min_samples_split}, "
                                          f"max_leaf_nodes={max_leaf_nodes}, min_samples_leaf={min_samples_leaf}, "
                                          f"max_features={max_features}, bootstrap={bootstrap}, max_samples={max_samples}")
                                    scores = [0, 0, 0, 0, 0]
                                    for k in range(0, n_folds):
                                        model = RandomForestClassifier(n_estimators=n, max_depth=max_depth, min_samples_split=min_samples_split,
                                                                       max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                                                                       max_features=max_features, bootstrap=bootstrap, max_samples=max_samples,
                                                                       ccp_alpha=0.002, criterion='entropy', random_state=seed)
                                        model.fit(x_train[k], y_train[k].values.ravel())
                                        scores[k] = model.score(x_val[k], y_val[k].values.ravel())
                                    avg_score = sum(scores) / n_folds
                                    print(avg_score)
                                    if avg_score > best_score:
                                        best_n = n
                                        best_bootstrap = bootstrap
                                        best_max_depth = max_depth
                                        best_min_samples_split = min_samples_split
                                        best_max_leaf_nodes = max_leaf_nodes
                                        best_min_samples_leaf = min_samples_leaf
                                        best_max_features = max_features
                                        best_max_samples = max_samples
                                        best_score = avg_score
    return {"n": best_n, "bootstrap": best_bootstrap, "max_depth": best_max_depth, "min_samples_split": best_min_samples_split,
            "max_leaf_nodes": best_max_leaf_nodes, "min_samples_leaf": best_min_samples_leaf, "max_features": best_max_features,
            "max_samples": best_max_samples}


def displayConfusionMatrix(y_ground_truth, y_predicted, title=""):
    cm = confusion_matrix(y_ground_truth, y_predicted)
    d = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Phishing"])
    d.plot()
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)

    data = load_data()
    feature_names = data.columns[1:]  # first is the class
    x_data = data.drop('class', axis='columns')
    y_data = data.iloc[:, :1]  # The class
    x_training, x_test, y_training, y_test = train_test_split(x_data, y_data, stratify=y_data, test_size=0.2)
    # print("Shape :", x_training.shape)
    # print(x_data.head())

    # - Stratified K-Fold
    n_folds = 5
    x_train_v, x_val, y_train_v, y_val = stratifiedKFold(data_x=x_training, data_y=y_training, n_folds=n_folds, seed=seed)

    # LIME
    explainer = lime_tabular.LimeTabularExplainer(x_training.values, mode="classification",
                                                  class_names=['Legit', 'Phishing'],
                                                  feature_names=feature_names)

    # --- DECISION TREE ----
    # best_alpha_tree, best_criterion_tree = determineDTkFoldConfiguration(x_train_v, x_val, y_train_v, y_val, seed)
    best_alpha_tree = 0.002
    best_criterion_tree = 'entropy'
    dt_model = tree.DecisionTreeClassifier(criterion=best_criterion_tree, ccp_alpha=best_alpha_tree, random_state=seed)
    dt_model.fit(x_training, y_training)
    # showTree(T)
    predictions_tree = dt_model.predict(x_test)
    displayConfusionMatrix(y_test, predictions_tree, "Decision Tree")
    with open(os.path.join('output', 'decision_tree.obj'), 'wb') as file:
        pickle.dump(dt_model, file)
    # print("Tree:", classification_report(y_test, predictions))

    # --- SVM ----
    # best_parameters_svm = determineSVMkFoldConfiguration(x_train_v, x_val, y_train_v, y_val, seed)
    best_parameters_svm = {"kernel": 'poly', "c": 1, "gamma": 0.1, "tol": 0.01, "degree": 5}
    svm_model = make_pipeline(StandardScaler(), svm.SVC(gamma=best_parameters_svm['gamma'], tol=best_parameters_svm['tol'],
                                                        coef0=best_parameters_svm['c'], degree=best_parameters_svm['degree'],
                                                        kernel=best_parameters_svm['kernel'], random_state=seed,
                                                        probability=True))
    svm_model.fit(x_training, y_training.values.ravel())
    with open(os.path.join('output', 'svm.obj'), 'wb') as file:
        pickle.dump(svm_model, file)
    predictions_svm = svm_model.predict(x_test)
    displayConfusionMatrix(y_test, predictions_svm, "SVM")

    # --- Random Forest ----
    # best_parameters_rf = determineRFkFoldConfiguration(x_train_v, x_val, y_train_v, y_val, seed)
    # print(best_parameters_rf)
    best_parameters_rf = {"n": 50, "bootstrap": False, "max_depth": 9, "min_samples_split": 25, "max_leaf_nodes": 35,
                          "min_samples_leaf": 10, "max_features": 10, "max_samples": None}

    rf_model = RandomForestClassifier(n_estimators=best_parameters_rf['n'], max_depth=best_parameters_rf['max_depth'],
                                      min_samples_split=best_parameters_rf['min_samples_split'], max_leaf_nodes=best_parameters_rf['max_leaf_nodes'],
                                      min_samples_leaf=best_parameters_rf['min_samples_leaf'], max_features=best_parameters_rf['max_features'],
                                      bootstrap=best_parameters_rf['bootstrap'], max_samples=best_parameters_rf['max_samples'],
                                      ccp_alpha=best_alpha_tree, criterion=best_criterion_tree, random_state=seed)
    rf_model.fit(x_training, y_training.values.ravel())
    with open(os.path.join('output', 'random_forest.obj'), 'wb') as file:
        pickle.dump(rf_model, file)
    predictions_rf = rf_model.predict(x_test)
    displayConfusionMatrix(y_test, predictions_rf, "RF")

    # LIME Explanations
    instance_to_explain = x_test.iloc[0]

    explanation_dt = explainer.explain_instance(instance_to_explain, dt_model.predict_proba,
                                                num_features=len(feature_names))
    explanation_figure = explanation_dt.as_pyplot_figure()
    explanation_figure.set_size_inches(20, 18)
    explanation_figure.set_dpi(100)
    plt.title("Explanation DT")
    plt.show()
    print(explanation_dt.as_list())

    explanation_svm = explainer.explain_instance(instance_to_explain, svm_model.predict_proba,
                                                 num_features=len(feature_names))
    explanation_figure = explanation_svm.as_pyplot_figure()
    explanation_figure.set_size_inches(20, 18)
    explanation_figure.set_dpi(100)
    plt.title("Explanation SVM")
    plt.show()
    print(explanation_svm.as_list())

    explanation_rf = explainer.explain_instance(instance_to_explain, rf_model.predict_proba,
                                                num_features=len(feature_names))
    explanation_figure = explanation_rf.as_pyplot_figure()
    explanation_figure.set_size_inches(20, 18)
    explanation_figure.set_dpi(100)
    plt.title("Explanation RF")
    plt.show()
    predictions_instance = rf_model.predict_proba(instance_to_explain.values.reshape(1, -1))
    print(f"Predictions: Legit = {predictions_instance[0]}%, Phishing = {predictions_instance[1]}%")
    print(explanation_rf.as_list())
