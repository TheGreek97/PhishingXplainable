import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def load_data(datasets, test_size=0.2, seed=0):
    x_data, y_data, feature_names = load_data_no_split(datasets)
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, stratify=y_data, test_size=test_size, random_state=seed)
    return X_train, y_train, X_test, y_test, feature_names


def load_data_no_split(datasets):
    dfs = []  # an empty list to store the data frames
    for path in datasets:
        for f in os.listdir(path):
            file_path = os.path.join(path, f)
            data_ = pd.read_json(file_path, lines=True)  # read data frame from json file
            try:
                # url_spec_chars = sum of all individual chars
                data_.at[0, 'url_spec_chars'] = sum(data_.at[0, 'url_spec_chars'].values())
                # 99,999,999 is too low for default values => transform it to 999,999,999
                data_.at[0, 'url_ranking'] = 999999999 if data_.at[0, 'url_ranking'] == 99999999 else data_.at[0, 'url_ranking']
                dfs.append(data_)  # append the data frame to the list
            except IndexError:
                print("Ignoring ", f)

    dataframe = pd.concat(dfs, ignore_index=False)  # concatenate all the data frames in the list.
    feature_names = dataframe.columns[1:]  # first is the class
    categorical_feature = "domain_geolocation"
    numeric_features = [col for col in feature_names if col != categorical_feature]
    x_data = dataframe.drop('class', axis='columns')
    y_data = dataframe.iloc[:, :1]  # The class

    # Fill eventual NaN values
    for col in x_data.columns:
        x_data[col].fillna(0)

    # Define transformers for preprocessing
    numeric_transformer = Pipeline(steps=[
        ('standard_scaler', StandardScaler()),  # Standard Scaling
        ('normalizer', MinMaxScaler())  # Normalization Scaling
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
    ])

    # Combine transformers into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, [categorical_feature])
        ]
    )

    # Preprocess the training data
    x_data = preprocessor.fit_transform(x_data)
    x_data = pd.DataFrame(x_data)
    return x_data, y_data, feature_names
