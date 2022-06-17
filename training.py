import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.tree as Tree
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


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
    seed = 42
    np.random.seed(seed)

    data = load_data()
    # training = data.sample(frac=0.8, random_state=42)
    # testing = data.drop(training.index)
    x_data = data.drop('class', axis='columns')
    y_data = data.iloc[:, :1]  # The class
    x_training, x_test, y_training, y_test = train_test_split(x_data, y_data, stratify=y_data, test_size=0.2)

    print("Shape :", x_training.shape)
    print(x_data.head())
    # - Stratified K-Fold
    # n_folds = 5
    # x_train_v, x_val, y_train_v, y_val = stratifiedKFold(data_x=x_training, data_y=y_training, n_folds=n_folds, seed=seed)

    # - Compute the best values for cpp_alpha and criterion
    # best_f1, best_alpha, best_criterion = determineDecisionTreekFoldConfiguration(x_train_v, x_val, y_train_v, y_val, seed)
    # print(f"Best f1: {best_f1}")
    # print(best_alpha, best_criterion)

    best_alpha = 0.002
    best_criterion = 'entropy'

    tree = decisionTreeLearner(x_training, y_training, seed, best_criterion, best_alpha)
    showTree(tree)

    # --Testing
    predictions = tree.predict(x_test)
    cf_mat = confusion_matrix(y_test, predictions)
    display = ConfusionMatrixDisplay(cf_mat, display_labels=["Legit", "Phishing"])
    display.plot()
    plt.show()
    print(classification_report(y_test, predictions))
