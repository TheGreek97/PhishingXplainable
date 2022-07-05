from interpret.glassbox import ExplainableBoostingClassifier
import numpy as np
import os
import json


def train(x_train, y_train, feature_names, seed):
    ebm = ExplainableBoostingClassifier(feature_names=feature_names, random_state=seed)
    sample_weights = np.zeros((len(y_train)))
    for i, y in enumerate(y_train):
        sample_weights[i] = 1 if y == 0 else 5
    ebm.fit(x_train, y_train, sample_weight=sample_weights)
    return ebm


def global_explanation(model):
    ebm_global = model.explain_global()
    return ebm_global


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
