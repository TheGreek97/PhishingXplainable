from data import load_data_no_split, train_test_split
from feature_computation import extract_features
import os
import pickle
import sys
import numpy as np
from lime import lime_tabular
from explanation import lime_explain_instance, shap_explain_instance


def load_model(file_name):
    with open(os.path.join('models', file_name), 'rb') as file:
        return pickle.load(file)


if __name__ == "__main__":
    # Load data
    seed = 42
    n_top_features = 3

    X, y, feature_names = load_data_no_split()
    explanation = None

    mail_file_basepath = "."
    mail_file_name = os.path.join(mail_file_basepath, sys.argv[0])

    model_name = sys.argv[1] + ".obj"
    is_nn = model_name in ["nn", "dnn", "mlp"]

    explanation_method = sys.argv[2]

    with open(mail_file_name, 'r') as mail_file:
        mail_features = extract_features(mail_file)  # X
        ml_model = load_model(model_name)
        classification_result = ml_model.predict(mail_features)  # y

        # ---- EBM -----
        if model_name == "ebm":
            # if model is ebm, it has the embedded method explain_local
            ebm_local = ml_model.explain_local(mail_features, classification_result)
            explanation = ebm_local.data(0)
            features = {explanation["names"][i]: explanation["scores"][i] for i in range(0, len(explanation["names"]))}
            x = [abs(features[f]) for f in feature_names]

        else:  # LIME or SHAP
            # Needed by the LIME and SHAP explainers
            x_training, _, _, _ = train_test_split(X, y, stratify=y, test_size=0.2, random_state=seed)

            if explanation_method == "lime":
                explanation = lime_explain_instance(instance_features=mail_features, model=ml_model,
                                                    feature_names=feature_names, x_stats=x_training.values)
            else:
                explanation = shap_explain_instance(instance_features=mail_features, model=ml_model,
                                                    feature_names=feature_names, x_training=x_training)
    print(explanation)

