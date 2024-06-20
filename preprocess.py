import os
import pandas
import feature_computation as fe
import json
import mailparser


def preprocess_csv_datasets():
    base_path = os.path.join("datasets", "csv")
    for dataset in ['phishing', 'legit']:
        df = pandas.read_csv(os.path.join(base_path, dataset + ".csv"), encoding='utf-8')
        df = df.reset_index()  # make sure indexes pair with number of rows
        for index, mail in df.iterrows():
            filename = str(index) + ".json"
            file_path = os.path.join('datasets', 'features', dataset, filename)
            if not os.path.isfile(file_path):
                print(f"Processing email {dataset}\\{filename}")
                features = fe.extract_features(mail)
                if features:
                    features["class"] = 1 if (dataset == 'phishing') else 0
                    write_feature_file(file_path, features)


def enron_dataset(start_index=0):
    base_path = 'datasets/raw/enron/'
    for dataset in ['emails-phishing.csv', 'emails-enron.csv']:
        df = pandas.read_csv(os.path.join(base_path, dataset), encoding='utf-8')
        df = df.reset_index()  # make sure indexes pair with number of rows
        item_class = 1 if (dataset == 'emails-phishing.csv') else 0
        filename_class = "phishing_" if (item_class == 1) else "legit_"
        print("DATASET: ", dataset)
        for index, mail in df.iterrows():
            if index >= start_index:
                if not pandas.isnull(mail['Message']) and mail['Message'] != "":
                    filename = filename_class + str(index) + ".json"
                    print(filename)
                    features = fe.extract_features(mail)
                    if features:
                        features["class"] = item_class  # PHISHING
                        feature_path = 'datasets/features/enron'
                        file_path = os.path.join(feature_path, filename)
                        write_feature_file(file_path, features)


def spam_assassin_dataset(folder, start_index=0):
    base_path = 'datasets/raw/SpamAssassin/'
    print("DATASET: ", folder)
    dataset_path = os.path.join(base_path, folder)
    mails = os.listdir(dataset_path)
    for i, m in enumerate(mails):
        if i >= start_index:
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
    if not os.path.isfile(file_path):  # if file does not exist, create it
        with open(file_path, 'x') as output:
            output.write(json.dumps(features, sort_keys=True, default=str))
    else:  # otherwise overwrite it
        with open(file_path, 'w') as output:
            output.write(json.dumps(features, sort_keys=True, default=str))


if __name__ == '__main__':
    preprocess_csv_datasets()
    # spam_assassin_dataset('easy_ham')
    # spam_assassin_dataset('hard_ham')


"""
# Change only the links_present feature
def dataset_links(folder):
    base_path = 'datasets/raw/SpamAssassin/'
    print("DATASET: ", folder)
    dataset_path = os.path.join(base_path, folder)
    mails = os.listdir(dataset_path)
    for i, m in enumerate(mails):
        if i < 862:
            filename = folder+"_legit_" + str(i) + ".json"
            print(filename)
            file_path = os.path.join(dataset_path, m)
            mail = mailparser.parse_from_file(file_path)
            # print(mail.body)
            links = fe.compute_links_in_mail_feature(mail.body)
            if links:
                feature_path = 'datasets/features/spam_assassin'
                file_path = os.path.join(feature_path, filename)
                with open(file_path, 'r') as json_file:
                    features = json.load(json_file)
                    features["links_present_mail"] = links
                    write_feature_file(file_path, features)
def compute_ip_addr_features(folder='', dataset=None):
    if dataset:
        base_path = 'datasets/raw/enron/'
        for dataset in ['emails-phishing.csv', 'emails-enron.csv']:
            dataset_path = os.path.join(base_path, dataset)
            df = pandas.read_csv(os.path.join(base_path, dataset), encoding='utf-8')
            df = df.reset_index()  # make sure indexes pair with number of rows
            item_class = 1 if (dataset == 'emails-phishing.csv') else 0
            filename_class = "phishing_" if (item_class == 1) else "legit_"
            print("DATASET: ", dataset)
            for index, row in df.iterrows():
                mail = row['Message']
                if not pandas.isnull(mail) and mail != "":
                    filename = filename_class + str(index) + ".json"
                    print(filename)
                    # print(mail.body)
                    ip = fe.compute_ip_addr_feature(mail)
                    if ip:
                        feature_path = 'datasets/features/enron'
                        file_path = os.path.join(feature_path, filename)
                        with open(file_path, 'r') as json_file:
                            features = json.load(json_file)
                            features["url_ip_address"] = ip
                            write_feature_file(file_path, features)
    else:
        base_path = 'datasets/raw/SpamAssassin/'
        print("DATASET: ", folder)
        dataset_path = os.path.join(base_path, folder)
        mails = os.listdir(dataset_path)
        for i, m in enumerate(mails):
            filename = folder + "_legit_" + str(i) + ".json"
            print(filename)
            file_path = os.path.join(dataset_path, m)
            mail = mailparser.parse_from_file(file_path)
            # print(mail.body)
            ip = fe.compute_ip_addr_feature(mail.body)
            if ip:
                feature_path = 'datasets/features/spam_assassin'
                file_path = os.path.join(feature_path, filename)
                with open(file_path, 'r') as json_file:
                    features = json.load(json_file)
                    features["url_ip_address"] = ip
                    write_feature_file(file_path, features)
"""
