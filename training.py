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
import ebm
from data import load_data, stratifiedKFold, load_data_no_split


def h_score_loss(x_, alpha=0.5):
    """
    Calculates the loss on the heterogeneity score (H-score) of the array of feature importance x_
    Parameters
    ----------
    x_: array of feature importance
    alpha: coefficient (0 < alpha < 1) that defines the weight that features with value less than the mean have.
        With alpha=0.5 all features have the same weight, regardless if their value is below or above the mean.
        A value of alpha below 0.5 gives more weight to features with value above the mean.
    Returns
    -------
    result : H-score for the array of feature importance x_
    """
    if not alpha > 0 and alpha < 1:
        raise ValueError("alpha must be between 0 and 1 (not included)")
    n = len(x_)
    mean = np.sum(x_)/n
    s = 0
    for x in x_:
        f = (1-alpha) * (x-mean) if x > 0 else alpha * (x-mean)
        s += np.abs(f)  # f * f
    return s/n


def custom_loss(y_true, y_pred, feature_importance):
    acc = 1 - accuracy_score(y_true, y_pred)
    het = h_score_loss(feature_importance, alpha=0.5)
    return acc + het*30


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


# Decision Tree Hyper-parameter tuning
def determineDTkFoldConfiguration(x_train, x_val, y_train, y_val, seed=0):
    best_criterion = 'gini'
    best_alpha = 0
    best_f1 = 0
    best_weights = {0: 1, 1: 5}
    min_samples_leaf = 7
    max_depth = 8
    n_folds = len(x_train)
    for w in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50]:
        for criterion in ['gini', 'entropy']:
            for alpha in np.arange(0, 0.5, 0.01):
                f1_scores = [0, 0, 0, 0, 0]
                print(f"Computing criterion={criterion}, alpha={alpha}, class weights={w}")
                for k in range(0, n_folds):
                    t = tree.DecisionTreeClassifier(criterion=criterion,
                                                    ccp_alpha=alpha,
                                                    random_state=seed,
                                                    class_weight={0: 1, 1: w},
                                                    min_samples_leaf=min_samples_leaf,
                                                    max_depth=max_depth)
                    t.fit(x_train[k], y_train[k])
                    predictions = t.predict(x_val[k])
                    # f1_scores[k] = recall_score(y_true=y_val[k], y_pred=predictions, average='weighted')
                    f1_scores[k] = f1_score(y_true=y_val[k], y_pred=predictions, average='weighted')
                avg_f1 = sum(f1_scores) / n_folds
                print(f"Average f1: {avg_f1}")
                if avg_f1 > best_f1:
                    best_weights = {0: 1, 1: w}
                    best_f1 = avg_f1
                    best_alpha = alpha
                    best_criterion = criterion

    return {"alpha": best_alpha, "criterion": best_criterion, "class_weights": best_weights,
            "max_depth": max_depth, "min_samples_leaf": min_samples_leaf}


# Logistic Regression Hyper-parameter tuning
def determineLRkFoldConfiguration(x_train, x_val, y_train, y_val, seed=0):
    best_solver = 'lbfgs'
    best_c = 10
    best_penalty = 'l2'
    best_weights = {0: 1, 1: 5}
    best_score = 0
    n_folds = len(x_train)
    for w in [1, 2, 5, 10]:
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
                        print(f"Computing solver={solver}, penalty={penalty}, C={c}, class_weights={w}")
                        for k in range(0, n_folds):
                            model = LogisticRegression(solver=solver, penalty=penalty, C=c, class_weight={0: 1, 1: w},
                                                       random_state=seed, l1_ratio=l1_ratio, max_iter=500)
                            model.fit(x_train[k], y_train[k].values.ravel())
                            scores[k] = model.score(x_val[k], y_val[k].values.ravel())

                            results = permutation_importance(model, x_train[k], y_train[k].values.ravel(), scoring='neg_mean_squared_error')
                            importance = results.importances_mean  # TODO Compute H-Score from importances
                        avg_score = sum(scores) / n_folds
                        print(avg_score)
                        if avg_score > best_score:
                            best_weights = {0: 1, 1: w}
                            best_solver = solver
                            best_c = c
                            best_penalty = penalty
                            best_score = avg_score
    return {"solver": best_solver, "c": best_c, "penalty": best_penalty, "class_weights": best_weights}


def determineSVMkFoldConfiguration(x_train, x_val, y_train, y_val, seed=0):
    best_kernel = 'linear'
    best_c = 0.1
    best_gamma = 'scale'
    best_tol = 0.1
    best_degree = 3
    best_weights = {0: 1, 1: 5}
    best_score = 0
    n_folds = len(x_train)
    for w in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50]:
        for kernel in ["linear", "poly", "rbf", "sigmoid"]:  # poly=240
            poly_degrees = range(2, 7) if kernel == "poly" else [3]
            gammas = [0.1, 1, 'auto'] if kernel in {"rbf", "poly", "sigmoid"} else ['scale']
            for c in [0.1, 0.5, 1, 10]:
                for gamma in gammas:
                    for tol in [0.1, 0.01, 0.005, 0.001]:
                        for degree in poly_degrees:
                            scores = [0, 0, 0, 0, 0]
                            print(f"Computing kernel={kernel}, c={c}, gamma={gamma}, degree={degree}, class_weight={w}")
                            for k in range(0, n_folds):
                                model = make_pipeline(StandardScaler(), svm.SVC(kernel=kernel,
                                                                                degree=degree,
                                                                                gamma=gamma,
                                                                                coef0=c,
                                                                                tol=tol,
                                                                                class_weight={0: 1, 1: w},
                                                                                random_state=seed))
                                model.fit(x_train[k], y_train[k].values.ravel())
                                scores[k] = model.score(x_val[k], y_val[k].values.ravel())
                            avg_score = sum(scores) / n_folds
                            print(avg_score)
                            if avg_score > best_score:
                                best_kernel = kernel
                                best_weights = {0: 1, 1: w}
                                best_c = c
                                best_gamma = gamma
                                best_tol = tol
                                best_degree = degree
                                best_score = avg_score
    return {"kernel": best_kernel, "c": best_c, "gamma": best_gamma, "degree": best_degree,
            "tol": best_tol, "class_weights": best_weights}


def computeModelRF(X, y, seed=0):
    """
        Random Forest Hyper-parameter tuning
    """
    # configure the cross-validation procedure
    cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    # enumerate splits
    outer_results = list()
    # define the model
    model = RandomForestClassifier(random_state=seed)
    best_params = {'max_features': 5, 'max_samples': 0.1, 'n_estimators': 10}

    for train_ix, test_ix in cv_outer.split(X, y):
        # split data
        X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
        # configure the cross-validation procedure
        cv_inner = KFold(n_splits=5, shuffle=True, random_state=seed)
        # define search space
        space = {
            #'class_weight': [{0: 1, 1: w} for w in [1, 2, 5, 10]],
            'n_estimators': [10, 50, 100],
            'max_features': [5, 10, 14, 18],
            'max_samples': [0.1, 0.25, 0.5]
        }
        sampler = ParameterGrid(space)
        best_score = 1000
        best_model = model
        best_params = {}
        outer_loop = 0
        inner_loop = 0
        for params in sampler:
            outer_loop += 1
            for ix_train, ix_test in cv_inner.split(X, y):
                temp_model = clone(model).set_params(**params)
                fitted_model = temp_model.fit(X.iloc[ix_train], y.iloc[ix_train].values.ravel())
                # perform permutation importance (https://scikit-learn.org/stable/modules/permutation_importance.html)
                importance = permutation_importance(fitted_model, X, y, scoring='neg_mean_squared_error').importances_mean
                scorer = make_scorer(custom_loss, feature_importance=importance, needs_proba=False)
                score = scorer(fitted_model, X.iloc[ix_test], y.iloc[ix_test].values.ravel())
                # do something with the results
                if score < best_score:
                    best_model = temp_model
                    best_score = score
                    best_params = params
                inner_loop += 1
                print(f"--Fold: {outer_loop} - {inner_loop}--")

        # evaluate model on the hold out dataset
        y_predicted = best_model.predict(X_test)
        # evaluate the model
        acc = accuracy_score(y_test, y_predicted)
        # store the result
        outer_results.append(acc)
        # report progress
        print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, best_score, best_params))
    # summarize the estimated performance of the model
    # print(f'Accuracy: ({np.mean(outer_results)}, {np.std(outer_results)})')
    best_model = model.set_params(**best_params)
    best_model.fit(X, y.values.ravel())  # Fit the final model on the whole dataset
    return best_model


def determineRFkFoldConfiguration(x_train, x_val, y_train, y_val, class_weights, seed=0):
    """
        Random Forest Hyper-parameter tuning
    """
    rf = RandomForestClassifier()
    hyperparams = {
        'n_estimators' : [5, 25, 50, 75, 100],
        'max_depth' : [2, 12, 24, None]
    }
    cross_val = GridSearchCV(rf, hyperparams, cv=5)
    cross_val.fit(x_train, y_train.values.ravel())
    
    best_n = 10
    best_bootstrap = True
    best_max_depth = 3
    # best_min_samples_split = 10
    # best_max_leaf_nodes = 5
    # best_min_samples_leaf = 10
    best_max_features = 3
    best_max_samples = 1
    best_weights = {0: 1, 1: 5}
    best_score = 0
    n_folds = len(x_train)
    # total = 50k random forests
    for w in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50]:
        for n in [10, 25, 50, 75, 100]:
            for bootstrap in [True, False]:
                max_samples_values = [0.1, 0.25, 0.5, 0.7] if bootstrap else [None]
                for max_depth in [3, 5, 7, 9]:  # A very deep tree can cause over-fitting
                    # for min_samples_split in [10, 25, 50, 100]:  # The min. observations in any node to split it
                        # for max_leaf_nodes in [5, 15, 25, 35]:  # The max. leaves in a tree
                            # for min_samples_leaf in [10, 25, 50, 100]:  # The min. samples in a leaf after a split
                    for max_features in [3, 7, 10, 14, 17]:
                        for max_samples in max_samples_values:
                            print(f"Computing n={n}, max_depth={max_depth}, class_weight={w}, "
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
                                                               class_weight={0: 1, 1: w},
                                                               random_state=seed)
                                model.fit(x_train[k], y_train[k].values.ravel())
                                scores[k] = model.score(x_val[k], y_val[k].values.ravel())
                            avg_score = sum(scores) / n_folds
                            print(avg_score)
                            if avg_score > best_score:
                                best_n = n
                                best_bootstrap = bootstrap
                                best_max_depth = max_depth
                                best_weights = {0: 1, 1: w}
                                # best_min_samples_split = min_samples_split
                                # best_max_leaf_nodes = max_leaf_nodes
                                # best_min_samples_leaf = min_samples_leaf
                                best_max_features = max_features
                                best_max_samples = max_samples
                                best_score = avg_score
    return {"n": best_n, "bootstrap": best_bootstrap, "max_depth": best_max_depth,
            # "min_samples_split": best_min_samples_split,
            # "max_leaf_nodes": best_max_leaf_nodes, "min_samples_leaf": best_min_samples_leaf,
            'alpha': 0.0, 'criterion': 'entropy',
            "max_features": best_max_features, "max_samples": best_max_samples, "class_weights": best_weights}


def displayConfusionMatrix(y_ground_truth, y_predicted, title="", save_file=False):
    cm = confusion_matrix(y_ground_truth, y_predicted)
    d = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Phishing"])
    d.plot()
    plt.title(title)
    plt.show()


def test_model(model, X_train, y_train, X_test, y_test, n_folds, print_=False, name='Results'):
    metrics = []
    for k in range(0, n_folds):
        y_train[k] = np.ravel(y_train[k])
        model.fit(X_train[k], y_train[k])
        predictions = model.predict(X_test[k])
        metrics.append(
            classification_report(y_test[k], predictions, target_names=['Legit', 'Phishing'], output_dict=True))
        if print_:
            displayConfusionMatrix(y_test[k], predictions, name)
            print(name+": ", classification_report(y_test[k], predictions, target_names=['Legit', 'Phishing']))
    metrics = get_avg_metrics(metrics, n_folds)
    return metrics


def get_avg_metrics(metrics, n=5):
    f1_scores = []
    accuracies = []
    precisions = []
    recalls = []
    npvs = []
    specificities = []
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


if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)

    execute_decision_tree = False
    execute_logistic_regression = False
    execute_svm = False
    execute_random_forest = True
    execute_mlp = False
    execute_dnn = False
    execute_ebm = False

    # LOAD THE DATA
    X, y, feature_names = load_data_no_split()

    # - Stratified K-Fold for validation
    x_training, x_test, y_training, y_test, _ = load_data(test_size=0.2, seed=seed)
    n_folds = 10
    x_train_v, x_val, y_train_v, y_val = stratifiedKFold(data_x=x_training, data_y=y_training, n_folds=n_folds, seed=seed)

    """# - Stratified K-Fold for testing
    x_training, x_test, y_training, y_test = stratifiedKFold(data_x=X, data_y=y, n_folds=n_folds, seed=seed)
    """
    # --- DECISION TREE ----
    if execute_decision_tree:
        # best_parameters_dt = determineDTkFoldConfiguration(x_train_v, x_val, y_train_v, y_val, seed)
        # print(best_parameters_dt)
        best_parameters_dt = {'alpha': 0.0, 'criterion': 'entropy', 'class_weights': {0: 1, 1: 1}, 'max_depth': 8, 'min_samples_leaf': 7} # best f1 score
        dt_model = tree.DecisionTreeClassifier(criterion=best_parameters_dt["criterion"],
                                               ccp_alpha=best_parameters_dt["alpha"],
                                               random_state=seed,
                                               class_weight=best_parameters_dt["class_weights"],
                                               min_samples_leaf=best_parameters_dt["min_samples_leaf"],
                                               max_depth=best_parameters_dt["max_depth"])

        metrics_dt = test_model(dt_model, x_training, y_training, x_test, y_test, n_folds, print_=False, name='DT')
        print(metrics_dt)
        dt_model.fit(X, y)
        # showTree(dt_model, feature_names)
        saveModel(dt_model, 'decision_tree')

    # --- LOGISTIC REGRESSION  ---
    if execute_logistic_regression:
        # best_parameters_lr = determineLRkFoldConfiguration(x_train_v, x_val, y_train_v, y_val, seed)
        # print(best_parameters_lr)
        best_parameters_lr = {'solver': 'lbfgs', 'c': 10, 'penalty': 'l2', 'class_weights': {0: 1, 1: 2}}
        lr_model = LogisticRegression(solver=best_parameters_lr["solver"], penalty=best_parameters_lr["penalty"],
                                      C=best_parameters_lr["c"], class_weight=best_parameters_lr["class_weights"],
                                      random_state=seed)
        metrics_lr = test_model(lr_model, x_training, y_training, x_test, y_test, n_folds, print_=False, name='LR')
        print(metrics_lr)
        lr_model.fit(X, y)
        saveModel(lr_model, 'logistic_regression')

    # --- SVM ----
    if execute_svm:
        # best_parameters_svm = determineSVMkFoldConfiguration(x_train_v, x_val, y_train_v, y_val, seed)
        # print(best_parameters_svm)
        best_parameters_svm = {'kernel': 'poly', 'c': 10, 'gamma': 1, 'degree': 2, 'tol': 0.1, 'class_weights': {0: 1, 1: 1}}
        svm_model = make_pipeline(StandardScaler(), svm.SVC(gamma=best_parameters_svm['gamma'],
                                                            # tol=best_parameters_svm['tol'],
                                                            coef0=best_parameters_svm['c'],
                                                            degree=best_parameters_svm['degree'],
                                                            kernel=best_parameters_svm['kernel'],
                                                            random_state=seed,
                                                            class_weight=best_parameters_svm["class_weights"],
                                                            probability=True))
        metrics_svm = test_model(svm_model, x_training, y_training, x_test, y_test, n_folds, print_=False, name='SVM')
        print(metrics_svm)
        svm_model.fit(X, y)
        saveModel(svm_model, 'svm')

    # --- Random Forest ----
    if execute_random_forest:
        # best_parameters_rf = determineRFkFoldConfiguration(x_train_v, x_val, y_train_v, y_val, seed)
        # print(best_parameters_rf)

        best_parameters_rf = {'alpha': 0.0, 'criterion': 'entropy', 'n': 75, 'bootstrap': True, 'max_depth': 9, 'max_features': 10, 'max_samples': 0.7, 'class_weights': {0: 1, 1: 1}}

        rf_model = RandomForestClassifier(n_estimators=best_parameters_rf['n'],
                                          max_depth=best_parameters_rf['max_depth'],
                                          # min_samples_split=best_parameters_rf['min_samples_split'],
                                          # max_leaf_nodes=best_parameters_rf['max_leaf_nodes'],
                                          # min_samples_leaf=best_parameters_rf['min_samples_leaf'],
                                          max_features=best_parameters_rf['max_features'],
                                          bootstrap=best_parameters_rf['bootstrap'],
                                          max_samples=best_parameters_rf['max_samples'],
                                          ccp_alpha=best_parameters_rf['alpha'],
                                          criterion=best_parameters_rf['criterion'],
                                          class_weight=best_parameters_rf["class_weights"],
                                          random_state=seed
                                          )
        metrics_rf = test_model(rf_model, x_training, y_training, x_test, y_test, n_folds, print_=False, name='RF')
        print(metrics_rf)
        rf_model.fit(X, y.values.ravel())
        """
        rf_model = computeModelRF(X, y, seed)
        saveModel(rf_model, 'random_forest')
        """
    # ---- EBM -----
    if execute_ebm:
        metrics = []
        for k in range(0, n_folds):
            ebm_model = ebm.train(x_train=x_training[k], y_train=y_training[k], feature_names=feature_names, seed=seed)
            # metrics_ebm = test_model(ebm_model, x_training, y_training, x_test, y_test, n_folds, print_=True, name='EBM')
            ebm_predictions = ebm_model.predict(x_test[k])
            metrics.append(
                classification_report(y_test[k], ebm_predictions, target_names=['Legit', 'Phishing'], output_dict=True))
            displayConfusionMatrix(y_test[k], ebm_predictions, 'EBM')
            print("EBM: ", classification_report(y_test[k], ebm_predictions, target_names=['Legit', 'Phishing']))
        metrics = get_avg_metrics(metrics, n_folds)
        print(metrics)
        ebm_model = ebm.train(x_train=X, y_train=y, feature_names=feature_names, seed=seed)
        saveModel(ebm_model, 'ebm')

    # # Neural Networks
    x_train_nn = nn.format_x_data(x_training)
    y_train_nn = nn.format_y_data(y_training)
    x_test_nn = nn.format_x_data(x_test)
    y_test_nn = nn.format_y_data(y_test)
    callbacks_list = [
        # min_delta: Minimum change in the monitored quantity to qualify as an improvement
        # patience: Number of epochs with no improvement after which training will be stopped
        # restore_best_weights: Whether to restore model weights from the epoch with the best value of val_loss
        callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, restore_best_weights=True)
    ]
    # Whole dataset (used for the final training)
    X_total_nn = nn.format_x_data(X)  # = nn.format_x_data([X])[0]
    y_total_nn = nn.format_y_data(y)

    # --- Multi-Layer Perceptron ----
    if execute_mlp:
        mlp_model, class_weights_mlp = nn.build_optimal_nn(x_train_v, x_val, y_train_v, y_val, deep=False)
        # class_weights_mlp = {0: 1, 1: 3}
        print("Class weights: ", class_weights_mlp)
        mlp_model.compile(loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy', tf.keras.metrics.Precision(),
                                                                           tf.keras.metrics.Recall()])
        """best_w = 1
        best_w_recall = 1
        best_score = 0
        best_score_recall = 0
        precisions = []
        recalls = []
        for w in weights:
            class_weights_mlp = {0: 1, 1: w}
            mlp_model.fit(x_train_nn, y_train_nn, epochs=100, verbose=2, callbacks=callbacks_list,
                          shuffle=True, validation_split=0.2, class_weight=class_weights_mlp)
            score = mlp_model.evaluate(x_test_nn, y_test_nn)
            precisions.append(score[2])
            recalls.append(score[3])
            if score[2] > best_score:
                best_w = {0: 1, 1: w}
                best_score = score[2]
            if score[3] > best_score_recall:
                best_w_recall = {0: 1, 1: w}
                best_score_recall = score[3]
        plt.xlabel("Weight of class Phishing")
        plt.ylabel("Precision")
        plt.plot(weights, precisions)
        plt.show()
        print("Best weight for precision: ", best_w)
        plt.xlabel("Weight of class Phishing")
        plt.ylabel("Recall")
        plt.plot(weights, recalls)
        plt.show()
        print("Best weight for recall: ", best_w_recall)"""
        mlp_model.compile(loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
        mlp_metrics = []
        for k in range(0, n_folds):
            mlp_model.fit(x_train_nn[k], y_train_nn[k], epochs=100, verbose=2, callbacks=callbacks_list,
                          shuffle=True, validation_split=0.2, class_weight=class_weights_mlp)

            predictions_mlp = mlp_model.predict(x_test_nn[k], verbose=1, use_multiprocessing=True, workers=12)
            predictions_mlp = np.argmax(predictions_mlp, axis=1)
            mlp_metrics.append(
                classification_report(y_test[k], predictions_mlp, target_names=['Legit', 'Phishing'], output_dict=True))
            displayConfusionMatrix(y_test[k], predictions_mlp, "MLP")
            print("MLP: ", classification_report(y_test[k], predictions_mlp, target_names=['Legit', 'Phishing']))
        metrics = get_avg_metrics(mlp_metrics, n_folds)
        print(metrics)
        mlp_model.fit(X_total_nn, y_total_nn, epochs=200, verbose=2, callbacks=callbacks_list,
                      shuffle=True, validation_split=0.2, class_weight=class_weights_mlp)

        mlp_model.save(os.path.join("models", "mlp"))

    # --- Deep Neural Network ----
    if execute_dnn:
        dnn_model, _ = nn.build_optimal_nn(x_train_v, x_val, y_train_v, y_val, deep=True)
        dnn_model.compile(loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
        precisions = []
        recalls = []
        class_weights_dnn = {0: 1, 1: 4}
        """for w in weights:
            class_weights_dnn = {0: 1, 1: w}
            dnn_model.fit(x_train_nn, y_train_nn, epochs=100, verbose=2, callbacks=callbacks_list,
                          shuffle=True, validation_split=0.2, class_weight=class_weights_dnn)
            score = dnn_model.evaluate(x_test_nn, y_test_nn)
            precisions.append(score[2])
            recalls.append(score[3])
            if score[2] > best_score:
                best_w = {0: 1, 1: w}
                best_score = score[2]
            if score[3] > best_score_recall:
                best_w_recall = {0: 1, 1: w}
                best_score_recall = score[3]
        plt.xlabel("Weight of class Phishing")
        plt.ylabel("Precision")
        plt.plot(weights, precisions)
        plt.show()
        print ("Best weight for precision: ", best_w)
        plt.xlabel("Weight of class Phishing")
        plt.ylabel("Recall")
        plt.plot(weights, recalls)
        plt.show()
        print ("Best weight for recall: ", best_w_recall)"""
        dnn_metrics = []
        # K-fold Cross Validation
        for k in range(0, n_folds):
            dnn_model.fit(x_train_nn[k], y_train_nn[k], epochs=100, verbose=2, callbacks=callbacks_list,
                          shuffle=True, validation_split=0.2, class_weight=class_weights_dnn)

            predictions_dnn = dnn_model.predict(x_test_nn[k], verbose=1, use_multiprocessing=True, workers=12)
            predictions_dnn = np.argmax(predictions_dnn, axis=1)
            dnn_metrics.append(classification_report(y_test[k], predictions_dnn, target_names=['Legit', 'Phishing'],
                                                     output_dict=True))
            displayConfusionMatrix(y_test[k], predictions_dnn, "DNN")
            print("DNN: ", classification_report(y_test[k], predictions_dnn, target_names=['Legit', 'Phishing']))
        metrics = get_avg_metrics(dnn_metrics, n_folds)
        print(metrics)
        dnn_model.fit(X_total_nn, y_total_nn, epochs=200, verbose=2, callbacks=callbacks_list,
                      shuffle=True, validation_split=0.2, class_weight=class_weights_dnn)

        dnn_model.save(os.path.join("models", "dnn"))

    duration = 1000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)
