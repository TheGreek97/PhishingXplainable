import math

import numpy as np
import os.path
import pickle
import matplotlib.pyplot as plt

import sklearn.tree as tree
import sklearn.svm as svm
from sklearn import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, ParameterGrid, GridSearchCV, StratifiedKFold

from sklearn.inspection import permutation_importance
from sklearn.metrics import fbeta_score, make_scorer

import winsound
import tensorflow as tf
from tensorflow import keras
from keras import callbacks
import nn
from data import load_data, load_data_no_split


def h_score_loss(x_, alpha=0.5):
    """
    Calculates the heterogeneity score (H-score) loss of the array of feature importance x_
    Parameters
    ----------
    x_: array of feature importance
    alpha: coefficient (0 < alpha < 1) that defines the weight that features with value less than the mean have.
        With alpha=0.5 all features have the same weight, regardless if their value is below or above the mean.
        A value of alpha below 0.5 gives more weight to features with value above the mean.
    Returns
    -------
    result : H-score loss for the array of feature importance x_ in range [0, 1]
    """
    if not alpha > 0 and alpha < 1:
        raise ValueError("alpha must be between 0 and 1 (not included)")
    n = len(x_)
    mean = np.sum(x_)/n
    s = 0
    for x in x_:
        f = (1-alpha) * (x-mean) if x > 0 else alpha * (x-mean)
        s += np.abs(f)  # f * f
    loss = s/n
    std_loss = (1 / (1 + math.exp(-loss)) - 0.5) * 2  # apply shifted sigmoid function to put the loss in range [0,1]
    # score = 1 / std_loss if std_loss != 0 else 1000  # ensure we don't divide by 0; if loss = 0 -> very high score
    return std_loss


def custom_score(y_true, y_pred, feature_importance, verbose=False, het_weight=0.5):
    """
    The custom score which considers both the F1-Score and the heterogeneity metrics
    :param y_true: ground truth labels
    :param y_pred: predicted labels
    :param feature_importance: array containing the importance for each feature
        :param verbose: whether to print the f1-score and the h-score in the console
    :param het_weight: the weight to assign to the heterogeneity in the loss
    :return:
    """
    f1 = f1_score(y_true, y_pred)
    het_loss = h_score_loss(feature_importance, alpha=0.5)
    score = f1 - het_loss*het_weight
    if verbose:
        print(f"Score: {score}. F1: {f1}, H: {het_loss}")
    return score


def showTree(model, feature_names):
    plt.figure(figsize=(200, 20))
    tree.plot_tree(model, feature_names=feature_names, filled=True, rounded=True, class_names=["Legit", "Phishing"])
    print("Number of nodes:", model.tree_.node_count)
    print("Number of leaves:", model.get_n_leaves())
    plt.show()
    """output_filepath = os.path.join('output', 'decision_tree.dot')
    dot_data = tree.export_graphviz(model, out_file=None,
                                    feature_names=feature_names,
                                    class_names=["Legit", "Phishing"],
                                    filled=True, rounded=True,
                                    special_characters=True)
    # graph = pydotplus.graphviz.graph_from_dot_data(dot_data)
    # graph.write_png(output_filepath)
    # return graph"""


def computeBestModelConfig(X, y, model, param_space, n_fold=5, seed=0, verbose=2):
    # configure the cross-validation procedure
    cv = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    sampler = ParameterGrid(param_space)
    best_score, best_params = 0, {}
    inner_counter = 0
    for params in sampler:  # for each combination of parameters
        if verbose > 0:
            print(f"---- Configuration: {params} ")
        inner_scores = list()
        for ix_train, ix_val in cv.split(X, y):  # hyperparameter tuning
            inner_counter += 1
            temp_model = clone(model).set_params(**params)
            fitted_model = temp_model.fit(X.iloc[ix_train], y.iloc[ix_train].values.ravel())
            X_val, y_val = X.iloc[ix_val], y.iloc[ix_val].values.ravel()
            # get permutation importance (https://scikit-learn.org/stable/modules/permutation_importance.html)
            importance = permutation_importance(fitted_model, X_val, y_val, scoring='neg_mean_squared_error').importances_mean
            scorer = make_scorer(custom_score, greater_is_better=True, feature_importance=importance,
                                 verbose=verbose > 2, needs_proba=False)
            score = scorer(fitted_model, X_val, y_val)
            inner_scores.append(score)
            if verbose > 1:
                print(f"-- {inner_counter}/{n_fold * len(sampler)} - Score = {score}")
        avg_score = sum(inner_scores) / len(inner_scores)
        if avg_score > best_score:
            best_score = avg_score
            best_params = params
            if verbose > 1:
                print(f"Updated best model - score: {best_score} - {best_params}")
    if verbose > 0:
        print(f"Best configuration = {best_params}")
    return best_params


def computeBestModelConfigNested(X, y, model, param_space, split_ixs, n_fold_inner=5, seed=0, verbose=2):
    """
        Classifier Hyper-parameter tuning within a nested CV
    """
    outer_results = list()
    outer_loop = 0
    best_params = {}

    # split_ixs = list(split_ixs)[0:1]  # take only the first outer split
    for train_ix, test_ix in split_ixs:  # Outer CV
        outer_loop += 1
        if verbose > 0:
            print(f"Outer Fold: {outer_loop}")
        # split data
        X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
        y_train, y_test = y.iloc[train_ix].values.ravel(), y.iloc[test_ix].values.ravel()
        best_params = computeBestModelConfig(X_train, y_train, model, param_space, n_fold_inner, seed, verbose)

        # evaluate model on the hold out dataset
        best_model = clone(model).set_params(**best_params)
        best_model.fit(X_train, y_train)
        y_predicted = best_model.predict(X_test)
        f1 = f1_score(y_test, y_predicted)
        outer_results.append(f1)  # store the results
        # report progress
        if verbose > 0:
            displayConfusionMatrix(y_test, y_predicted)
            print(f'{outer_loop}> f1-score=%.3f - {best_params}' % f1)
    # summarize the estimated performance of the model
    if verbose > 0:
        print(f'Mean F1-Score: (mean = {np.mean(outer_results)}, std = {np.std(outer_results)})')
        print(f'Best configuration: {best_params})')
    return best_params


def test_model_cv(model, X, y, split_ixs, print_=False, name='Results'):
    metrics = []
    for train_ix, test_ix in split_ixs:
        x_train, y_train = X.iloc[train_ix], y.iloc[train_ix].values.ravel()
        x_test, y_test = X.iloc[test_ix], y.iloc[test_ix].values.ravel()

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics.append(
            classification_report(y_test, predictions, target_names=['Legit', 'Phishing'], output_dict=True))
        if print_:
            displayConfusionMatrix(y_test, predictions, name)
            print(name+": ", classification_report(y_test, predictions, target_names=['Legit', 'Phishing']))
    metrics = get_avg_metrics(metrics)
    return metrics


def displayConfusionMatrix(y_ground_truth, y_predicted, title="", save_file=False):
    cm = confusion_matrix(y_ground_truth, y_predicted)
    d = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Phishing"])
    d.plot()
    plt.title(title)
    plt.show()
    if save_file and title != "":
        output_path = os.path.join('output', 'conf_matrix', title)
        plt.savefig(output_path)


def get_avg_metrics(metrics):
    f1_scores = []
    accuracies = []
    precisions = []
    recalls = []
    npvs = []
    specificities = []
    n = len(metrics)
    n = n if n > 0 else 1
    for result in metrics:
        accuracies.append(result["accuracy"])
        f1_scores.append(result["Phishing"]["f1-score"])
        precisions.append(result["Phishing"]["precision"])
        recalls.append(result["Phishing"]["recall"])
        npvs.append(result["Legit"]["precision"])
        specificities.append(result["Legit"]["recall"])
    return {
        "accuracy": sum(accuracies) / n,
        "f1_score": sum(f1_scores) / n,
        "precision": sum(precisions) / n,
        "recall": sum(recalls) / n,
        "npv": sum(npvs) / n,
        "specificity": sum(specificities) / n,
    }


def saveModel(model, file_name):
    with open(os.path.join('models', file_name+'.obj'), 'wb') as write_file:
        pickle.dump(model, write_file)


def test_nn_cv(nn_model, X, y, split_ixs, class_weight, print_=False, name='Results'):
    metrics = []
    for train_ix, test_ix in split_ixs:
        x_train_nn, x_test_nn = X[train_ix, :], X[test_ix, :]
        y_train_nn, y_test_nn = y[train_ix], y[test_ix]
        model = keras.models.clone_model(nn_model)
        model = nn.fit_model(model, x_train_nn, y_train_nn, class_weight)
        # model.fit(x=x_train_nn, y=y_train_nn, epochs=200, verbose=2, callbacks=callbacks_list,
        #          shuffle=True, validation_split=0.0, class_weight=class_weight)
        predictions = model.predict(x_test_nn, verbose=1, use_multiprocessing=True, workers=12)
        predictions = np.argmax(predictions, axis=1)
        y_test_nn = np.argmax(y_test_nn, axis=1)
        metrics.append(classification_report(y_test_nn, predictions, target_names=['Legit', 'Phishing'],
                                             output_dict=True))
        if (print_):
            displayConfusionMatrix(y_test_nn, predictions, name)
            print(f"{name}: ", classification_report(y_test_nn, predictions, target_names=['Legit', 'Phishing']))
    return metrics


if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)  # set random state also for sklearn
    tf.random.set_seed(seed)

    execute_decision_tree = False
    execute_logistic_regression = False
    execute_svm = False
    execute_random_forest = False
    execute_ebm = False
    execute_mlp = True
    execute_dnn = False

    # LOAD THE DATA
    X, y, feature_names = load_data_no_split()
    X_train, y_train, _, _, _ = load_data(0.2, seed)  # Single split if we need it

    n_folds = 5
    cv_outer = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    test_cv_ixs = cv_outer.split(X, y)

    # --- DECISION TREE ----
    if execute_decision_tree:
        model = tree.DecisionTreeClassifier(random_state=seed)
        space = {
            'criterion': ['gini', 'entropy'],
            'ccp_alpha': np.arange(0, 0.5, 0.01),
            'min_samples_leaf': [7],
            'max_depth': [8]
        }
        # best_params_dt = computeBestModelConfig(X_train, y_train, model=model, param_space=space, seed=seed)
        best_params_dt = {'ccp_alpha': 0.0, 'criterion': 'entropy', 'max_depth': 8, 'min_samples_leaf': 7}
        dt_model = model.set_params(**best_params_dt)
        # Evaluate
        metrics_dt = test_model_cv(dt_model, X, y, test_cv_ixs, print_=True, name='DT')
        print(metrics_dt)
        dt_model.fit(X, y)
        saveModel(dt_model, 'decision_tree')
        # showTree(dt_model, feature_names)

    # --- LOGISTIC REGRESSION  ---
    if execute_logistic_regression:
        # print(best_parameters_lr)
        model = LogisticRegression(random_state=seed)
        # best_parameters_lr = {'solver': 'lbfgs', 'c': 10, 'penalty': 'l2', 'class_weights': {0: 1, 1: 2}}
        space = {
            'solver': ['newton-cg', 'lbfgs', 'liblinear'],
            'C': [100, 10, 1.0, 0.1, 0.01],
            'penalty': ['l2']
        }
        best_params_lr = computeBestModelConfig(X_train, y_train, model=model, param_space=space, seed=seed)
        lr_model = model.set_params(**best_params_lr)
        # Evaluate
        metrics_lr = test_model_cv(lr_model, X, y, test_cv_ixs, print_=True, name='LR')
        print(metrics_lr)
        lr_model.fit(X, y)  # Fit the final model on the whole dataset
        saveModel(lr_model, 'logistic_regression')

    # --- SVM ----
    if execute_svm:
        model = svm.SVC(probability=True, random_state=seed)
        space = {
            'C': [1, 10, 100],
            'gamma': [1, 0.1, 0.01],
            'degree': [3, 5, 7],
            'kernel': ['poly']
        }
        best_params_svm = computeBestModelConfig(X_train, y_train, model=model, param_space=space, seed=seed)
        # best_parameters_svm = {'C': 10, 'degree': 3, 'gamma': 1, 'kernel': 'poly'}

        svm_model = model.set_params(**best_params_svm)
        svm_model.fit(X, y.values.ravel())  # Fit the final model on the whole dataset
        svm_model = make_pipeline(StandardScaler(), svm_model)

        metrics_svm = test_model_cv(svm_model, X, y, test_cv_ixs, print_=True, name='SVM')
        print(metrics_svm)

        saveModel(svm_model, 'svm')

    # --- Random Forest ----
    if execute_random_forest:
        model = RandomForestClassifier(random_state=seed)
        space = {
            # 'class_weight': [{0: 1, 1: w} for w in [1, 2, 5, 10]],
            'n_estimators': [10, 50, 100],
            'max_features': [5, 10, 14, 18],
            'max_samples': [0.1, 0.25, 0.5]
        }
        best_params_rf = computeBestModelConfig(X_train, y_train, model=model, param_space=space, seed=seed)
        rf_model = model.set_params(**best_params_rf)
        metrics_rf = test_model_cv(rf_model, X, y, test_cv_ixs, print_=True, name='RF')
        print(metrics_rf)
        rf_model.fit(X, y.values.ravel())  # Fit the final model on the whole dataset
        saveModel(rf_model, 'random_forest')

    # ---- EBM -----
    if execute_ebm:
        # ebm_model = ebm.train(x_train=X_train, y_train=y_train, feature_names=feature_names, class_weight=1, seed=seed)
        metrics = list()
        from interpret.glassbox import ExplainableBoostingClassifier

        for train_ix, test_ix in test_cv_ixs:
            x_train_, y_train_ = X.iloc[train_ix], y.iloc[train_ix]
            x_test_, y_test_ = X.iloc[test_ix], y.iloc[test_ix]
            ebm_model = ExplainableBoostingClassifier(feature_names=feature_names, random_state=seed,
                                                      validation_size=0.2)
            ebm_model.fit(x_train_, y_train_)
            """
            ebm_model = ebm.train(x_train=X_train, y_train=y_train, feature_names=feature_names, class_weight=1,
                                  seed=seed)"""
            predictions = ebm_model.predict(x_test_)
            metrics.append(
                classification_report(y_test_, predictions, target_names=['Legit', 'Phishing'], output_dict=True))
            displayConfusionMatrix(y_test_, predictions, "EBM")
            print("EBM: ", classification_report(y_test_, predictions, target_names=['Legit', 'Phishing']))
        metrics_ebm = get_avg_metrics(metrics)
        print(metrics_ebm)
        # Fit the final model on the whole dataset
        # ebm_model = ebm.train(x_train=X, y_train=y, feature_names=feature_names, seed=seed)
        # saveModel(ebm_model, 'ebm')

    # # Neural Networks
    X_train_nn, y_train_nn = nn.format_x_data(X_train), nn.format_y_data(y_train)
    X_nn, y_nn = nn.format_x_data(X), nn.format_y_data(y)
    # --- Multi-Layer Perceptron ----
    if execute_mlp:
        mlp_model = nn.get_optimal_net(X_train, y_train, n_folds, deep=False)
        class_weights_mlp = {0: 1, 1: 2}
        test_nn_cv(mlp_model, X_nn, y_nn, test_cv_ixs, print_=True, class_weight=class_weights_mlp, name='MLP')
        # Fit to the whole dataset
        mlp_model = nn.fit_model(mlp_model, X_nn, y_nn, class_weights_mlp)
        mlp_model.save(os.path.join("models", "mlp"))

    # --- Deep Neural Network ----
    if execute_dnn:
        dnn_model = nn.get_optimal_net(X_train, y_train, n_folds, deep=True)
        class_weights_dnn = {0: 1, 1: 2}
        test_nn_cv(dnn_model, X_nn, y_nn, test_cv_ixs, print_=True, class_weight=class_weights_dnn, name='DNN')
        # Fit to the whole dataset
        dnn_model = nn.fit_model(dnn_model, X_nn, y_nn, class_weights_dnn)
        dnn_model.save(os.path.join("models", "dnn"))

    duration = 1000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)
