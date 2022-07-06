from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import f1_score
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def train(x_train, y_train, feature_names, seed):
    best_score = 0
    best_w = 1  # best w is 1
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train, test_size=0.2, random_state=seed)
    f1_scores = []
    weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50]
    """for w in weights:
        ebm = ExplainableBoostingClassifier(feature_names=feature_names, random_state=seed, validation_size=0.2)
        sample_weights = np.zeros((len(y_train)))
        for i, y in enumerate(y_train.values):
            sample_weights[i] = 1 if y[0] == 0 else w
        ebm.fit(x_train, y_train, sample_weight=sample_weights)
        y_pred = ebm.predict(x_val)
        score = f1_score(y_val.values.astype(int).ravel(), y_pred.astype(int), labels=['Non-phishing', 'Phishing'])
        f1_scores.append(score)
        if score > best_score:
            best_score = score
            best_w = w
    plt.xlabel("Weight of class Phishing")
    plt.ylabel("F1 score")
    plt.plot(weights, f1_scores)
    plt.show()
    print(best_w, best_score)"""
    sample_weights = np.zeros((len(y_train)))
    for i, y in enumerate(y_train.values):
        sample_weights[i] = 1 if y[0] == 0 else best_w
    ebm_model = ExplainableBoostingClassifier(feature_names=feature_names, random_state=seed, validation_size=0.2)
    ebm_model.fit(x_train, y_train, sample_weight=sample_weights)

    return ebm_model


def ebm_global_feature_importance(model, x_test, y_test, feature_names, n_top_features=3):
    feature_presence = {f: 0 for f in feature_names}
    length = len(x_test)
    ebm_local = model.explain_local(x_test.iloc[:length], y_test.iloc[:length])
    tmp = 0
    for i in range(0, length):
        explanation = ebm_local.data(i)  # ["names"] and ["scores"]
        features = {explanation["names"][i]: explanation["scores"][i] for i in range(0, len(explanation["names"]))}
        x = [abs(features[f]) for f in feature_names]
        indexes_sort = np.argsort(x)
        tmp += features['url_shortened']
        for j in range(1, n_top_features+1):  # take the top N features
            top_feature = feature_names[indexes_sort[-j]]  # the indexes are the features (-j => arr sort in asc. order)
            feature_presence[top_feature] += 1  # increase the feature by one if it is in the top 3 features

    base_path = os.path.join('output', 'feature_importance')
    file_path = os.path.join(base_path,  'ebm.json')
    mode = 'w' if os.path.exists(file_path) else 'x'
    with open(file_path, mode) as out:
        out.write(json.dumps(feature_presence))
