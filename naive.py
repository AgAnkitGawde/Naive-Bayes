import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_naive import *
from sklearn.naive_bayes import BernoulliNB ,GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression

def preprocess_data():
    path_to_train = "traindata.txt"
    path_to_train_label = "trainlabels.txt"
    path_to_test = "testdata.txt"
    path_to_test_label = "testlabels.txt"
    path_to_stoplist = "stoplist.txt"
    # Load the data
    print("Loading data...")
    train_data, test_data, train_label,test_label, features, vocabulary = load_data(path_to_train, path_to_train_label, path_to_test,path_to_test_label, path_to_stoplist)
    # Standardize the data
    return train_data, test_data, train_label,test_label, features, vocabulary


if __name__ == "__main__":

    X_train, X_test,Y_train,Y_test, features, vocabulary= preprocess_data()
    print("\nData sucessfully loaded.")
    Y_test_predict=[]
    Y_train_predict=[]
    error=0
    Probability_x1_y1=[]
    Probability_x1_y0=[]
    Probability_y1_vocabulary =[]
    Probability_y0_vocabulary =[]
    N_1 = Y_train.count('1')
    N_0 = Y_train.count('0')
    Probability_y1 = X_train[:N_1]
    Probability_y0 = X_train[N_1:]

    for i in range(0 ,len(vocabulary)):
        count = 0
        counter = 0
        word = vocabulary[i]
        #For counting words when y = 1
        for j in range(0,N_1):
            count+=X_train[j].count(word)
        #For counting words when y = 0
        for k in range(N_1 + 1 , len(X_train)):
            counter+= X_train[k].count(word)
        Probability_y1_vocabulary.append((count+1)/(N_1+2))
        Probability_y0_vocabulary.append((counter+1)/(N_0+2))
    #print(Probability_y1_vocabulary)
    for word in X_train:
        product_y1=1
        product_y0=1
        y_predict1=[]
        y_predict0=[]
        for i in range(0,len(vocabulary)):
            if vocabulary[i] in word:
                y_predict1.append(Probability_y1_vocabulary[i])
                y_predict0.append(Probability_y0_vocabulary[i])
            else:
                y_predict1.append(1-Probability_y1_vocabulary[i])
                y_predict0.append(1-Probability_y0_vocabulary[i])
        for num in y_predict1:
            product_y1 *= num
        for num in y_predict0:
            product_y0 *=num
        if product_y0 > product_y1:
            Y_train_predict.append('0')
        else:
            Y_train_predict.append('1')
    for i in range(0,len(Y_train)):
        if Y_train_predict[i] != Y_train[i]:
            error += 1
    accuracy = ((1 - error / len(Y_train)) * 100 )
    print("Training accuracy is")
    print(accuracy)
    #print(Probability_y1_vocabulary)

    error=0
    x_train = np.zeros(shape=(len(X_train),len(vocabulary)))
    x_test = np.zeros(shape=(len(X_test),len(vocabulary)))
    for word in X_test:
        product_y1=1
        product_y0=1
        y_predict1=[]
        y_predict0=[]
        for i in range(0,len(vocabulary)):
            if vocabulary[i] in word:
                y_predict1.append(Probability_y1_vocabulary[i])
                y_predict0.append(Probability_y0_vocabulary[i])
            else:
                y_predict1.append(1-Probability_y1_vocabulary[i])
                y_predict0.append(1-Probability_y0_vocabulary[i])
        for num in y_predict1:
            product_y1 *= num
        for num in y_predict0:
            product_y0 *=num
        if product_y0 > product_y1:
            Y_test_predict.append('0')
        else:
            Y_test_predict.append('1')
    k=0
    for word in X_train:
        for j in range(0,len(vocabulary)):
            if vocabulary[j] in word:
                x_train[k][j]=1
        k +=1
    k=0
    for word in X_test:
        for j in range(0,len(vocabulary)):
            if vocabulary[j] in word:
                x_test[k][j]=1
        k +=1
    for i in range(0,len(Y_test)):
        if Y_test_predict[i] != Y_test[i]:
            error += 1
    accuracy = ((1 - error / len(Y_test)) * 100 )
    print("Testing accuracy is")
    print(accuracy)

    print("\nNaive Bayes with scikit")
    bnb = BernoulliNB()
    model = bnb.fit(x_train,Y_train)
    y_pred = model.predict(x_train)
    print("Training accuracy is")
    print(np.mean(y_pred==Y_train)*100)
    print("Testing accuracy is")
    y_pred = model.predict(x_test)
    print(np.mean(y_pred==Y_test)*100)
    print("\nLogisticRegression")
    print("Training accuracy")
    clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(x_train, Y_train)
    y_pred=clf.predict(x_train)
    print(np.mean(y_pred==Y_train)*100)

    print("Testing accuracy")
    y_pred=clf.predict(x_test)
    print(np.mean(y_pred==Y_test)*100)
    '''
    for i in range(0,len(Y_test)):
        if y_pred[i] != Y_test[i]:
            error += 1
    accuracy = ((1 - error / len(Y_test)) * 100 )
    print(accuracy)
    Probability_y1=( N_1 / len(Y_train))
    for i in range(0,len(X_train)):
        if Y_train[i] == '1':
            Probability_x1_y1.append(len(X_train[i])/N_1)
        else:
            Probability_x1_y0.append(len(X_train[i])/N_0)
    print(Probability_x1_y1)
    print(Probability_x1_y0)
    '''
