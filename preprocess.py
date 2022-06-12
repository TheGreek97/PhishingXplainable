import os
import pandas
import feature_extraction as fe
import json

def enron_dataset():
    base_path = 'datasets/raw/enron/'
    for dataset in ['emails-phishing.csv', 'emails-enron.csv']:
        df = pandas.read_csv(os.path.join(base_path, dataset), encoding='utf-8', encoding_errors='ignore')
        # print(df.columns)
        df = df.reset_index()  # make sure indexes pair with number of rows
        print (df.iloc[251])
        """
        for index, row in df.iterrows():
            mail = row['Message']
            features = fe.extract_features(mail)
            features["class"] = 1  # PHISHING
            feature_path = 'datasets\\features'
            try:
                with open(os.path.join(feature_path, "phishing_" + str(index)) + ".json", 'x') as output:
                    output.write(json.dumps(features))
            except FileExistsError:
                with open(os.path.join(feature_path, "phishing_" + str(index)) + ".json", 'w') as output:
                    output.write(json.dumps(features))"""

        """with open(os.path.join(base_path, dataset), mode='r', encoding='utf-8', errors='ignore') as csv_file:
            mails = csv.reader(csv_file)
            for mail in mails:
                mail = mail[5]  # read csv: [Date,FromName,FromAddress,To,Subject,Message]
        """


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
            features = extract_features(mail)
            feature_path = 'datasets/features'
            # with open(os.path.join(feature_path, m), 'w') as output:
            #    output.write(features)


if __name__ == '__main__':
    enron_dataset()
