import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import sklearn.tree as Tree
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn import preprocessing


def load(path):
    return pd.read_csv(path)


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


def decisionTreeLearner(X, y, seed, criterion='gini', ccp_alpha=0.01):
    tree_learner = Tree.DecisionTreeClassifier(criterion=criterion, ccp_alpha=ccp_alpha, random_state=seed)
    tree_learner.fit(X, y)
    return tree_learner


def showTree(T):
    plt.figure(figsize=(200, 20))
    Tree.plot_tree(T)
    print("Number of nodes:", T.tree_.node_count)
    print("Number of leaves:", T.get_n_leaves())
    plt.show()


def decisionTreeF1(x, y, tree):
    predictions = tree.predict(x)
    return f1_score(y_true=y, y_pred=predictions, average='weighted')


def determineDecisionTreekFoldConfiguration(x_train, x_val, y_train, y_val, seed=0):
    best_criterion = 'gini'
    best_alpha = 0
    best_f1 = 0
    n_folds = len(x_train)
    for criterion in ['gini', 'entropy']:
        for alpha in np.arange(0, 0.5, 0.001):
            f1_scores = [0, 0, 0, 0, 0]
            print(f"Computing criterion={criterion}, alpha={alpha}")
            for k in range(0, n_folds):
                tree = decisionTreeLearner(x_train[k], y_train[k], seed=seed, criterion=criterion, ccp_alpha=alpha)
                f1_scores[k] = decisionTreeF1(x=x_val[k], y=y_val[k], tree=tree)
            avg_f1 = sum(f1_scores) / n_folds
            print(f"Average f1: {avg_f1}")
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_alpha = alpha
                best_criterion = criterion

    return best_f1, best_alpha, best_criterion


if __name__ == '__main__':
    training = load('trainDdosLabelNumeric.csv')
    seed = 42
    np.random.seed(seed)

    print("Shape :", training.shape)
    cols = list(training.columns.values)
    independentList = cols[0:training.shape[1] - 1]
    x = training.loc[:, independentList]
    y = training['Label']  # The class

    # - Stratified K-Fold
    n_folds = 5
    x_train, x_val, y_train, y_val = stratifiedKFold(data_x=x, data_y=y, n_folds=n_folds, seed=seed)

    # - Compute tree with an arbitrary value for criterion and ccp_alpha
    T = decisionTreeLearner(x, y, seed=0, criterion='entropy', ccp_alpha=0.01)
    showTree(T)

    # - Compute the best values for cpp_alpha and criterion
    #best_f1, best_alpha, best_criterion = determineDecisionTreekFoldConfiguration(x_train, x_val, y_train, y_val, seed)
    #print(f"Best f1: {best_f1}")
    #print(best_alpha, best_criterion)

    best_alpha = 0.001
    best_criterion = 'entropy'

    tree = decisionTreeLearner(x, y, seed, best_criterion, best_alpha)

    # --Testing
    test = load('testDdosLabelNumeric.csv')
    x_test = test.loc[:, independentList]
    y_test = test['Label']  # The class

    predictions = tree.predict(x_test)
    cf_mat = confusion_matrix(y_test, predictions)
    display = ConfusionMatrixDisplay(cf_mat)
    display.plot()
    plt.show()
    print(classification_report(y_test, predictions))
