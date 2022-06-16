import os
import pandas
import feature_computation as fe
import json
import mailparser


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
            if not pandas.isnull(mail) and mail != "" and 150 < index:
                filename = filename_class + str(index) + ".json"
                print(filename)
                features = fe.extract_features(mail)
                if features:
                    features["class"] = item_class  # PHISHING
                    feature_path = 'datasets/features/enron'
                    file_path = os.path.join(feature_path, filename)
                    write_feature_file(file_path, features)


def spam_assassin_dataset(folder):
    base_path = 'datasets/raw/SpamAssassin/'
    print("DATASET: ", folder)
    dataset_path = os.path.join(base_path, folder)
    mails = os.listdir(dataset_path)
    for i, m in enumerate(mails):
        filename = folder+"_legit_" + str(i) + ".json"
        print(filename)
        file_path = os.path.join(dataset_path, m)
        mail = mailparser.parse_from_file(file_path)
        # print(mail.body)
        features = fe.extract_features(mail.body)
        if features:
            features["class"] = 0  # LEGIT
            feature_path = 'datasets/features/spam_assassin'
            file_path = os.path.join(feature_path, filename)
            write_feature_file(file_path, features)


def write_feature_file(file_path, features):
    try:
        with open(file_path, 'x') as output:
            output.write(json.dumps(features, sort_keys=True, default=str))
    except FileExistsError:
        with open(file_path, 'w') as output:
            output.write(json.dumps(features, sort_keys=True, default=str))


if __name__ == '__main__':
    # enron_dataset()
    # spam_assassin_dataset('easy_ham')
    spam_assassin_dataset('hard_ham')
