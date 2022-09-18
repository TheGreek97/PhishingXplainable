import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def load_data(test_size=0.2, seed=0):
    x_data, y_data, feature_names = load_data_no_split()

    x_tr, y_tr, x_tst, y_tst = train_test_split(x_data, y_data, stratify=y_data, test_size=test_size, random_state=seed)
    return x_tr, y_tr, x_tst, y_tst, feature_names


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


def load_data_no_split():
    dfs = []  # an empty list to store the data frames
    for path in [os.path.join('datasets', 'features', 'enron'), os.path.join('datasets', 'features', 'spam_assassin')]:
        for f in os.listdir(path):
            file_path = os.path.join(path, f)
            data_ = pd.read_json(file_path, lines=True)  # read data frame from json file
            try:
                # url_spec_chars = sum of all individual chars
                data_.at[0, 'url_spec_chars'] = sum(data_.at[0, 'url_spec_chars'].values())
                # 99,999,999 is too low => transform it to 999,999,999
                data_.at[0, 'url_ranking'] = 999999999 if data_.at[0, 'url_ranking'] == 99999999 else data_.at[0, 'url_ranking']
                dfs.append(data_)  # append the data frame to the list
            except IndexError:
                print("Ignoring ", f)

    dataframe = pd.concat(dfs, ignore_index=False)  # concatenate all the data frames in the list.
    feature_names = dataframe.columns[1:]  # first is the class
    x_data = dataframe.drop('class', axis='columns')
    y_data = dataframe.iloc[:, :1]  # The class

    # Fill eventual NaN values
    for col in x_data.columns:
        x_data[col].fillna(0)
    std_scaler = StandardScaler()
    normalize_scaler = MinMaxScaler()
    columns = x_data.columns
    df_scaled = normalize_scaler.fit_transform(x_data.to_numpy())
    df_scaled = std_scaler.fit_transform(df_scaled)
    x_data = pd.DataFrame(df_scaled, columns=columns)

    return x_data, y_data, feature_names