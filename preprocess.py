import os
import pandas
import feature_computation as fe
import json
import math


def enron_dataset():
    base_path = 'datasets/raw/enron/'
    for dataset in ['emails-phishing.csv', 'emails-enron.csv']:
        df = pandas.read_csv(os.path.join(base_path, dataset), encoding='utf-8')
        # print(df.columns)
        df = df.reset_index()  # make sure indexes pair with number of rows
        item_class = 1 if (dataset == 'emails-phishing.csv') else 0
        filename_class = "phishing_" if (item_class == 1) else "legit_"
        print("DATASET: ", dataset)
        for index, row in df.iterrows():
            mail = row['Message']
            if not pandas.isnull(mail) and mail != "" and index < 5:
                filename = filename_class + str(index) + ".json"
                print(filename)
                features = fe.extract_features(mail)
                if features:
                    features["class"] = item_class  # PHISHING
                    feature_path = 'datasets\\features'
                    try:
                        with open(os.path.join(feature_path, filename), 'x') as output:
                            output.write(json.dumps(features))
                    except FileExistsError:
                        with open(os.path.join(feature_path, filename), 'w') as output:
                            output.write(json.dumps(features))


def spam_assassin_dataset():
    base_path = 'datasets/raw/SpamAssassin/'
    for folder in ['easy_ham', 'hard_ham']:
        dataset_path = os.path.join(base_path, folder)
        mails = os.listdir(dataset_path)
        # for m in mails:
        #   file_path = os.path.join(dataset_path, m)

        file_path = os.path.join(dataset_path, mails[0])
        with open(file_path, mode='r') as m:
            mail = m.read()
            # TODO separate mails within a single file (divided by -------------------------)
            features = fe.extract_features(mail)
            feature_path = 'datasets/features'
            # with open(os.path.join(feature_path, m), 'w') as output:
            #    output.write(features)


if __name__ == '__main__':
    enron_dataset()
