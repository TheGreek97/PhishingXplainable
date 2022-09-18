import numpy as np
import os.path
import pickle
import matplotlib.pyplot as plt

import sklearn.tree as tree
import sklearn.svm as svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from keras import callbacks
import nn
import ebm
from data import load_data, stratifiedKFold, load_data_no_split

"""
def het_score(x):
    n_samples = 100
    alpha_het = 0.5
    sum = 0
    for i in range(0,n_samples):

    return sum/n_samples
"""

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
    for w in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50]:
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
            "max_features": best_max_features, "max_samples": best_max_samples, "class_weights": best_weights}

"""
def determineBoostkFoldConfiguration(x_train, x_val, y_train, y_val, model=None, seed=0):
    # AdaBoosting classifier Hyper-parameter tuning
    best_n = 25
    best_learning_rate = 0.1
    best_score = 0
    n_folds = len(x_train)
    for n in [25, 50, 100, 200]:
        for learning_rate in [1, 2.5, 5]:
            print(f"Computing model={model} n_estimators={n}, learning rate={learning_rate}, class_weight={w}")
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
"""


def displayConfusionMatrix(y_ground_truth, y_predicted, title="", save_file=False):
    cm = confusion_matrix(y_ground_truth, y_predicted)
    d = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Phishing"])
    d.plot()
    plt.title(title)
    plt.show()


def test_model(model, X_train, y_train, X_test, y_test, n_folds, print_=False, name='Results'):
    metrics = []
    for k in range(0, n_folds):
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

    # LOAD THE DATA
    X, y, feature_names = load_data_no_split()
    # print("Shape :", X.shape)
    # print(X.head())

    # - Stratified K-Fold for validation
    x_train_v, _, y_train_v, _, _ = load_data(test_size=0.2, seed=seed)
    n_folds = 10
    x_train_v, x_val, y_train_v, y_val = stratifiedKFold(data_x=x_train_v, data_y=y_train_v, n_folds=n_folds, seed=seed)

    # - Stratified K-Fold for testing
    x_training, x_test, y_training, y_test = stratifiedKFold(data_x=X, data_y=y, n_folds=n_folds, seed=seed)

    # --- DECISION TREE ----
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
    # best_parameters_rf = determineRFkFoldConfiguration(x_train_v, x_val, y_train_v, y_val, seed)
    # print(best_parameters_rf)
    best_parameters_rf = {'n': 75, 'bootstrap': True, 'max_depth': 9, 'max_features': 10, 'max_samples': 0.7, 'class_weights': {0: 1, 1: 5}}

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
                                      class_weight=best_parameters_dt["class_weights"],
                                      random_state=seed
                                      )
    metrics_rf = test_model(rf_model, x_training, y_training, x_test, y_test, n_folds, print_=False, name='RF')
    print(metrics_rf)
    rf_model.fit(X, y)
    saveModel(rf_model, 'random_forest')

# --- Multi-Layer Perceptron ----
    x_train_nn, y_train_nn, x_test_nn, y_test_nn = nn.get_data(x_training, y_training, x_test, y_test)
    callbacks_list = [
        # min_delta: Minimum change in the monitored quantity to qualify as an improvement
        # patience: Number of epochs with no improvement after which training will be stopped
        # restore_best_weights: Whether to restore model weights from the epoch with the best value of val_loss
        callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, restore_best_weights=True)
    ]

    mlp_model, class_weights_mlp = nn.build_optimal_nn(x_train_v, x_val, y_train_v, y_val, deep=False)
    print("Class weights: ", class_weights_mlp)  # best = {0: 1, 1: 3}
    mlp_model.compile(loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy', tf.keras.metrics.Precision(),
                                                                       tf.keras.metrics.Recall()])
    weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50]
    class_weights_mlp = {0: 1, 1: 3}
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
    X_nn, y_nn, _, _ = nn.get_data(X, y, x_test, y_test)
    mlp_model.fit(X_nn, y_nn, epochs=200, verbose=2, callbacks=callbacks_list,
                  shuffle=True, validation_split=0.2, class_weight=class_weights_mlp)

    mlp_model.save(os.path.join("models", "mlp"))

    # --- Deep NN ----
    dnn_model, _ = nn.build_optimal_nn(x_train_v, x_val, y_train_v, y_val, deep=True)
    dnn_model.compile(loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy', tf.keras.metrics.Precision(),
                                                                       tf.keras.metrics.Recall()])
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
    dnn_model.compile(loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    dnn_metrics = []
    for k in range(0, n_folds):
        dnn_model.fit(x_train_nn[k], y_train_nn[k], epochs=100, verbose=2, callbacks=callbacks_list,
                      shuffle=True, validation_split=0.2, class_weight=class_weights_mlp)

        predictions_dnn = dnn_model.predict(x_test_nn[k], verbose=1, use_multiprocessing=True, workers=12)
        predictions_dnn = np.argmax(predictions_dnn, axis=1)
        dnn_metrics.append(
            classification_report(y_test[k], predictions_dnn, target_names=['Legit', 'Phishing'], output_dict=True))
        displayConfusionMatrix(y_test[k], predictions_dnn, "DNN")
        print("DNN: ", classification_report(y_test[k], predictions_dnn, target_names=['Legit', 'Phishing']))
    metrics = get_avg_metrics(dnn_metrics, n_folds)
    print(metrics)
    dnn_model.fit(X_nn, y_nn, epochs=200, verbose=2, callbacks=callbacks_list,
                  shuffle=True, validation_split=0.2, class_weight=class_weights_mlp)

    dnn_model.save(os.path.join("models", "dnn"))

    # ---- EBM -----
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

    """
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
