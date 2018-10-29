import pandas as pd
import numpy as np

def load_data(train_path,train_label_path, test_path,test_label_path, stoplist_path):
    train =[]
    test =[]
    word_label = []
    vocabulary=[]
    counter=-1
    feature=[]
    print("Preprocesing data...")

    # Get data into memory as list of lines
    with open(train_path, 'r') as file:
        train_data = file.read().splitlines()
    with open(test_path, 'r') as file:
        test_data = file.read().splitlines()
    with open(stoplist_path, 'r') as file:
        stoplist = file.read().splitlines()
    with open(train_label_path, 'r') as file:
        train_label = file.read().splitlines()
    with open(test_label_path, 'r') as file:
        test_label = file.read().splitlines()

    #Convert the data into list of words
    for line in train_data:
        train_refined=[]
        words = line.split()
        for word in words:
            if word not in stoplist:
                train_refined.append(word)
        train.append(train_refined)

        for line in train:
            for word in line:
                if word not in vocabulary:
                    vocabulary.append(word)
    vocabulary=sorted(vocabulary)
    for line in train:
        zeros = ('0' * len(vocabulary))
        for word in line:
            index=(vocabulary.index(word))
            zeros = list(zeros)
            zeros[index] = '1'
            zeros = "".join(zeros)
        feature.append(zeros)

    #Convert the data into list of words
    for line in test_data:
        test_refined=[]
        words = line.split()
        for word in words:
            if word not in stoplist:
                test_refined.append(word)
        test.append(test_refined)
    return train, test, train_label,test_label, feature, vocabulary
