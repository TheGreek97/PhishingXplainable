import pandas as pd
import numpy as np
import os.path
import pickle
import matplotlib.pyplot as plt
import pydotplus
import graphviz

import sklearn.tree as tree
import sklearn.svm as svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.optimizers import Adam
from keras import callbacks
from keras.utils import np_utils
import keras_tuner as kt

import nn
from data import load_data, stratifiedKFold


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


def decisionTreeF1(x, y, model):
    predictions = model.predict(x)
    return f1_score(y_true=y, y_pred=predictions, average='weighted')


# Decision Tree Hyper-parameter tuning
def determineDTkFoldConfiguration(x_train, x_val, y_train, y_val, class_weights, seed=0):
    best_criterion = 'gini'
    best_alpha = 0
    best_f1 = 0
    min_samples_leaf = 7
    max_depth = 8
    n_folds = len(x_train)
    for criterion in ['gini', 'entropy']:
        for alpha in np.arange(0, 0.2, 0.001):
            f1_scores = [0, 0, 0, 0, 0]
            print(f"Computing criterion={criterion}, alpha={alpha}")
            for k in range(0, n_folds):
                t = tree.DecisionTreeClassifier(criterion=criterion,
                                                ccp_alpha=alpha,
                                                random_state=seed,
                                                class_weight=class_weights,
                                                min_samples_leaf=min_samples_leaf,
                                                max_depth=max_depth)
                t.fit(x_train[k], y_train[k])
                f1_scores[k] = decisionTreeF1(x=x_val[k], y=y_val[k], model=t)
            avg_f1 = sum(f1_scores) / n_folds
            print(f"Average f1: {avg_f1}")
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_alpha = alpha
                best_criterion = criterion

    return {"alpha": best_alpha, "criterion": best_criterion,
            "max_depth": max_depth, "min_samples_leaf": min_samples_leaf}


# Logistic Regression Hyper-parameter tuning
def determineLRkFoldConfiguration(x_train, x_val, y_train, y_val, class_weights, seed=0):
    best_solver = 'lbfgs'
    best_c = 10
    best_penalty = 'l2'
    best_score = 0
    n_folds = len(x_train)
    for solver in ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]:
        if solver in {'newton-cg', "lbfgs", "sag"}:
            penalties = ['l2', 'none']
        elif solver == "saga":
            penalties = ['elasticnet', 'l1', 'l2', 'none']
        else:  # liblinear
            penalties = ['l1', 'l2']
        for penalty in penalties:
            if penalty == 'elasticnet':
                l1_ratios = [0, 0.25, 0.5, 0.75, 1]
            else:
                l1_ratios = [None]
            if penalty == 'none':
                cs = [1]
            else:
                cs = [10, 1.0, 0.1, 0.01]
            for l1_ratio in l1_ratios:
                for c in cs:
                    scores = [0, 0, 0, 0, 0]
                    print(f"Computing solver={solver}, penalty={penalty}, C={c}")
                    for k in range(0, n_folds):
                        model = LogisticRegression(solver=solver, penalty=penalty, C=c, class_weight=class_weights,
                                                   random_state=seed, l1_ratio=l1_ratio, max_iter=500)
                        model.fit(x_train[k], y_train[k].values.ravel())
                        scores[k] = model.score(x_val[k], y_val[k].values.ravel())
                    avg_score = sum(scores) / n_folds
                    print(avg_score)
                    if avg_score > best_score:
                        best_solver = solver
                        best_c = c
                        best_penalty = penalty
                        best_score = avg_score
    return {"solver": best_solver, "c": best_c, "penalty": best_penalty}


def determineSVMkFoldConfiguration(x_train, x_val, y_train, y_val, class_weights, seed=0):
    """
    SVM Hyper-parameter tuning
    :param x_train: training set X
    :param x_val: validation set X
    :param y_train: training set Y
    :param y_val: validation set Y
    :param class_weights: class weights
    :param seed: random state
    :return: dictionary with the best parameters
    """
    best_kernel = 'linear'
    best_c = 0.1
    best_gamma = 'scale'
    best_tol = 0.1
    best_degree = 3
    best_score = 0
    n_folds = len(x_train)
    for kernel in ["linear", "poly", "rbf", "sigmoid"]:  # poly=240
        poly_degrees = range(2, 7) if kernel == "poly" else [3]
        gammas = [0.1, 1, 'auto'] if kernel in {"rbf", "poly", "sigmoid"} else ['scale']
        for c in [0.1, 0.5, 1, 10]:
            for gamma in gammas:
                for tol in [0.1, 0.01, 0.005, 0.001]:
                    for degree in poly_degrees:
                        scores = [0, 0, 0, 0, 0]
                        print(f"Computing kernel={kernel}, c={c}, gamma={gamma}, degree={degree}")
                        for k in range(0, n_folds):
                            model = make_pipeline(StandardScaler(), svm.SVC(kernel=kernel,
                                                                            degree=degree,
                                                                            gamma=gamma,
                                                                            coef0=c,
                                                                            tol=tol,
                                                                            class_weight=class_weights,
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
    return {"kernel": best_kernel, "c": best_c, "gamma": best_gamma, "degree": best_degree, "tol": best_tol}


def determineRFkFoldConfiguration(x_train, x_val, y_train, y_val, class_weights, seed=0):
    """
        Random Forest Hyper-parameter tuning
    """
    best_n = 10
    best_bootstrap = True
    best_max_depth = 3
    # best_min_samples_split = 10
    # best_max_leaf_nodes = 5
    # best_min_samples_leaf = 10
    best_max_features = 3
    best_max_samples = 1
    best_score = 0
    n_folds = len(x_train)
    # total = 50k random forests
    for n in [10, 25, 50, 75, 100]:
        for bootstrap in [True, False]:
            max_samples_values = [0.1, 0.25, 0.5, 0.7] if bootstrap else [None]
            for max_depth in [3, 5, 7, 9]:  # A very deep tree can cause over-fitting
                # for min_samples_split in [10, 25, 50, 100]:  # The min. observations in any node to split it
                    # for max_leaf_nodes in [5, 15, 25, 35]:  # The max. leaves in a tree
                        # for min_samples_leaf in [10, 25, 50, 100]:  # The min. samples in a leaf after a split
                for max_features in [3, 7, 10, 14, 17]:
                    for max_samples in max_samples_values:
                        print(f"Computing n={n}, max_depth={max_depth}, "
                              # f"min_sample_split={min_samples_split}, max_leaf_nodes={max_leaf_nodes}, min_samples_leaf={min_samples_leaf}, "
                              f"max_features={max_features}, bootstrap={bootstrap}, max_samples={max_samples}")
                        scores = [0, 0, 0, 0, 0]
                        for k in range(0, n_folds):
                            model = RandomForestClassifier(n_estimators=n,
                                                           max_depth=max_depth,
                                                           # min_samples_split=min_samples_split,
                                                           # max_leaf_nodes=max_leaf_nodes,
                                                           # min_samples_leaf=min_samples_leaf,
                                                           max_features=max_features,
                                                           bootstrap=bootstrap,
                                                           max_samples=max_samples,
                                                           ccp_alpha=0.002,
                                                           criterion='entropy',
                                                           class_weight=class_weights,
                                                           random_state=seed)
                            model.fit(x_train[k], y_train[k].values.ravel())
                            scores[k] = model.score(x_val[k], y_val[k].values.ravel())
                        avg_score = sum(scores) / n_folds
                        print(avg_score)
                        if avg_score > best_score:
                            best_n = n
                            best_bootstrap = bootstrap
                            best_max_depth = max_depth
                            # best_min_samples_split = min_samples_split
                            # best_max_leaf_nodes = max_leaf_nodes
                            # best_min_samples_leaf = min_samples_leaf
                            best_max_features = max_features
                            best_max_samples = max_samples
                            best_score = avg_score
    return {"n": best_n, "bootstrap": best_bootstrap, "max_depth": best_max_depth,
            # "min_samples_split": best_min_samples_split,
            # "max_leaf_nodes": best_max_leaf_nodes, "min_samples_leaf": best_min_samples_leaf,
            "max_features": best_max_features, "max_samples": best_max_samples}


def determineBoostkFoldConfiguration(x_train, x_val, y_train, y_val, model=None, seed=0):
    """
        AdaBoosting classifier Hyper-parameter tuning
    """
    best_n = 25
    best_learning_rate = 0.1
    best_score = 0
    n_folds = len(x_train)
    for n in [25, 50, 100, 200]:
        for learning_rate in [1, 2.5, 5]:
            print(f"Computing model={model} n_estimators={n}, leaarning rate={learning_rate}, ")
            scores = [0, 0, 0, 0, 0]
            for k in range(0, n_folds):
                model = AdaBoostClassifier(base_estimator=model, n_estimators=n,
                                           learning_rate=learning_rate, random_state=seed)
                model.fit(x_train[k], y_train[k].values.ravel())
                scores[k] = model.score(x_val[k], y_val[k].values.ravel())
            avg_score = sum(scores) / n_folds
            print(avg_score)
            if avg_score > best_score:
                best_n = n
                best_learning_rate = learning_rate
                best_score = avg_score
    return {"n": best_n, "learning_rate": best_learning_rate}


def displayConfusionMatrix(y_ground_truth, y_predicted, title=""):
    cm = confusion_matrix(y_ground_truth, y_predicted)
    d = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Phishing"])
    d.plot()
    plt.title(title)
    plt.show()


def saveModel(model, file_name):
    with open(os.path.join('models', file_name+'.obj'), 'wb') as write_file:
        pickle.dump(model, write_file)


if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)

    x_training, x_test, y_training, y_test, feature_names = load_data(test_size=0.2, seed=seed)
    # print("Shape :", x_training.shape)
    # print(x_data.head())

    # - Stratified K-Fold
    n_folds = 5
    x_train_v, x_val, y_train_v, y_val = stratifiedKFold(data_x=x_training, data_y=y_training, n_folds=n_folds, seed=seed)

    class_weights = {0: 1, 1: 5}
    """
    # --- DECISION TREE ----
    #    best_parameters_dt = determineDTkFoldConfiguration(x_train_v, x_val, y_train_v, y_val,
    #                                                       class_weights, seed)
    best_parameters_dt = {'alpha': 0.0, 'criterion': 'gini', 'max_depth': 8, 'min_samples_leaf': 7}
    dt_model = tree.DecisionTreeClassifier(criterion=best_parameters_dt["criterion"],
                                           ccp_alpha=best_parameters_dt["alpha"],
                                           random_state=seed,
                                           class_weight=class_weights,
                                           min_samples_leaf=best_parameters_dt["min_samples_leaf"],
                                           max_depth=best_parameters_dt["max_depth"])
    
    dt_model.fit(x_training, y_training)
    # showTree(dt_model, feature_names)
    predictions_tree = dt_model.predict(x_test)
    displayConfusionMatrix(y_test, predictions_tree, "DT")
    saveModel(dt_model, 'decision_tree')
    print("DT:", classification_report(y_test, predictions_tree, target_names=['Legit', 'Phishing']))
    
    # --- Logistic Regression ---
    # best_parameters_lr = determineLRkFoldConfiguration(x_train_v, x_val, y_train_v, y_val, class_weights, seed)
    # print(best_parameters_lr)
    best_parameters_lr = {'solver': 'liblinear', 'c': 1, 'penalty': 'l1'}
    lr_model = LogisticRegression(solver=best_parameters_lr["solver"], penalty=best_parameters_lr["penalty"],
                                  C=best_parameters_lr["c"], class_weight=class_weights, random_state=seed)
    lr_model.fit(x_training, y_training.values.ravel())
    predictions_lr = lr_model.predict(x_test)
    displayConfusionMatrix(y_test, predictions_lr, "LR")
    saveModel(lr_model, 'logistic_regression')
    print("LR:", classification_report(y_test, predictions_lr, target_names=['Legit', 'Phishing']))
    """
    """"
    # --- AdaBoost classifier ---
    best_parameters_boost = determineBoostkFoldConfiguration(x_train_v, x_val, y_train_v, y_val, seed=seed)
    print(best_parameters_boost)
    # best_parameters_boost =
    boost_model = AdaBoostClassifier(n_estimators=best_parameters_boost['n'],
                                     learning_rate=best_parameters_boost['learning_rate'],
                                     random_state=seed)
    boost_model.fit(x_training, y_training.values.ravel())
    predictions_boost = boost_model.predict(x_test)
    displayConfusionMatrix(y_test, predictions_boost, "AdaBoost")
    saveModel(boost_model, 'adaboost')
    print("AdaBoost:", classification_report(y_test, predictions_boost, target_names=['Legit', 'Phishing']))
    """
    """
    # --- SVM ----
    # best_parameters_svm = determineSVMkFoldConfiguration(x_train_v, x_val, y_train_v, y_val, class_weights, seed)
    # print(best_parameters_svm)
    best_parameters_svm = {'kernel': 'poly', 'c': 10, 'gamma': 'auto', 'degree': 5, 'tol': 0.1}  # {"kernel": 'poly', "c": 10, "gamma": 1, "degree": 3}
    svm_model = make_pipeline(StandardScaler(), svm.SVC(gamma=best_parameters_svm['gamma'],
                                                        # tol=best_parameters_svm['tol'],
                                                        coef0=best_parameters_svm['c'],
                                                        degree=best_parameters_svm['degree'],
                                                        kernel=best_parameters_svm['kernel'],
                                                        random_state=seed,
                                                        class_weight=class_weights,
                                                        probability=True))
    svm_model.fit(x_training, y_training.values.ravel())
    predictions_svm = svm_model.predict(x_test)
    displayConfusionMatrix(y_test, predictions_svm, "SVM")
    saveModel(svm_model, 'svm')
    print("SVM:", classification_report(y_test, predictions_svm, target_names=['Legit', 'Phishing']))

    # --- Random Forest ----
    # best_parameters_rf = determineRFkFoldConfiguration(x_train_v, x_val, y_train_v, y_val, class_weights, seed)
    # print(best_parameters_rf)
    best_parameters_rf = {'n': 100, 'bootstrap': False, 'max_depth': 9, 'max_features': 3, 'max_samples': None}  # ,  "min_samples_split": 25, "max_leaf_nodes": 35, "min_samples_leaf": 10}

    rf_model = RandomForestClassifier(n_estimators=best_parameters_rf['n'],
                                      max_depth=best_parameters_rf['max_depth'],
                                      # min_samples_split=best_parameters_rf['min_samples_split'],
                                      # max_leaf_nodes=best_parameters_rf['max_leaf_nodes'],
                                      # min_samples_leaf=best_parameters_rf['min_samples_leaf'],
                                      max_features=best_parameters_rf['max_features'],
                                      bootstrap=best_parameters_rf['bootstrap'],
                                      max_samples=best_parameters_rf['max_samples'],
                                      ccp_alpha=best_parameters_dt['alpha'],
                                      criterion=best_parameters_dt['criterion'],
                                      class_weight=class_weights,
                                      random_state=seed
                                      )
    rf_model.fit(x_training, y_training.values.ravel())
    predictions_rf = rf_model.predict(x_test)
    displayConfusionMatrix(y_test, predictions_rf, "RF")
    saveModel(rf_model, 'random_forest')
    print("RF:", classification_report(y_test, predictions_rf, target_names=['Legit', 'Phishing']))
"""
    # --- Multi-Layer Perceptron ----
    x_train_nn, y_train_nn, x_test_nn, _ = nn.get_data(x_training, y_training, x_test, y_test)

    mlp_model = nn.build_optimal_nn(x_train_nn, y_train_nn, deep=False)
    callbacks_list = [
        # min_delta: Minimum change in the monitored quantity to qualify as an improvement
        # patience: Number of epochs with no improvement after which training will be stopped
        # restore_best_weights: Whether to restore model weights from the epoch with the best value of val_loss
        callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, restore_best_weights=True)
    ]
    mlp_model.compile(loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    mlp_model.fit(x_train_nn, y_train_nn, epochs=100, verbose=2, callbacks=callbacks_list,
                 shuffle=True, validation_split=0.2, class_weight=class_weights)
    predictions_nn = mlp_model.predict(x_test_nn, verbose=1, use_multiprocessing=True, workers=12)
    predictions_nn = np.argmax(predictions_nn, axis=1)

    displayConfusionMatrix(y_test, predictions_nn, "MLP")
    mlp_model.save(os.path.join("models", "mlp"))
    print("MLP:", classification_report(y_test, predictions_nn, target_names=['Legit', 'Phishing']))

    # --- Deep NN ----
    dnn_model = nn.build_optimal_nn(x_train_nn, y_train_nn, deep=True)
    dnn_model.compile(loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    dnn_model.fit(x_train_nn, y_train_nn, epochs=100, verbose=2, callbacks=callbacks_list,
                 shuffle=True, validation_split=0.2, class_weight=class_weights)
    predictions_dnn = dnn_model.predict(x_test_nn, verbose=1, use_multiprocessing=True, workers=12)
    predictions_dnn = np.argmax(predictions_dnn, axis=1)

    displayConfusionMatrix(y_test, predictions_dnn, "DNN")
    dnn_model.save(os.path.join("models", "dnn"))
    print("DNN:", classification_report(y_test, predictions_dnn, target_names=['Legit', 'Phishing']))
