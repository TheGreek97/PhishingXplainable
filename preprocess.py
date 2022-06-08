import os
import itertools as it


def enron_dataset():
    base_path = 'datasets/raw/enron/'
    for f in ['emails-phishing.txt', 'emails-enron.txt']:
        file_path = os.path.join(base_path, f)

        with open(file_path, mode='rb') as m:
            mail = m.read()
            mail = mail.decode('utf-8') # TODO
            print (mail)

            mails = mail.split("\n\nFrom")
            print (mails[0])
            feature_path = 'datasets/features'
            # with open(os.path.join(feature_path, m), 'w') as output:
            #    output.write(features)


if __name__ == '__main__':
    enron_dataset()