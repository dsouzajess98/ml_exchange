import pandas as pd
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
import json
import joblib
import numpy as np


class Classification2():
    def __init__(self, train_data, attributes):
        # heart = pd.read_csv('/home/jessica/ml_exchange_project/data/SAHeart.csv', sep=',', header=0)
        # print(heart.head())
        #
        # y = heart.iloc[:,9]
        # X = heart.iloc[:,:9]
        #
        # LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr',max_iter=1000).fit(X, y)
        # print(LR.predict(X.iloc[460:, :]))
        # print(round(LR.score(X, y), 4))

        # diabetes = pd.read_csv('/home/jessica/ml_exchange_project/data/election_dataset.csv', sep=',', header=0)
        diabetes = pd.DataFrame(train_data)
        print(diabetes.head())

        y = diabetes.iloc[:,attributes]
        X = diabetes.iloc[:,:attributes]

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # print(X_train.shape, y_train.shape)
        # print(X_test.shape, y_test.shape)

        # lm = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr',max_iter=1000)
        # model = lm.fit(X_train, y_train)
        # predictions = lm.predict(X_test)
        #
        # print("Score:", model.score(X_test, y_test))

        self.mlp = MLPClassifier(hidden_layer_sizes=(4, 4, 4), activation='relu', solver='adam', max_iter=5000)
        self.mlp.fit(X, y)

        predict_train = self.mlp.predict(X)

        print(confusion_matrix(y, predict_train))
        print(classification_report(y, predict_train))

        print("Training accuracy:", self.mlp.score(X, y))
        print("Weights: ", self.mlp.coefs_)
        print("Biases: ", self.mlp.intercepts_)
        self.score = self.mlp.score(X, y)

    def weights_and_biases(self):
        coef = []
        intercepts = []
        for x in self.mlp.coefs_:
            coef.append(x.tolist())

        for x in self.mlp.intercepts_:
            intercepts.append(x.tolist())

        c=[]
        i =[]
        removeNestings(coef, c)
        removeNestings(intercepts, i)

        coef = [int(i*1000) for i in c]
        intercepts = [int(i*1000) for i in i]
        json_str = {"weights": coef, "biases": intercepts, "accuracy": self.score}
        filename = 'finalized_model_2.sav'
        joblib.dump(self.mlp, filename)
        return json.dumps(json_str)


def removeNestings(l, output):
    for i in l:
        if type(i) == list:
            removeNestings(i, output)
        else:
            output.append(i)