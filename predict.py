__author__ = 'kdq'

import pandas as pd
import numpy as np
import re
import csv
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model, neighbors, svm
import matplotlib.pyplot as plt
port = {
    'S':1,
    'C':2,
    'Q':3
}
ids = []
diff_title = {}
diff_age = {}
title = {
    'Mr':1,
    'Mrs':2,
    'Miss':3,
    'Master':4
}
female_ratio, male_ratio = 0, 0
pclass_survived_ratio = [.62962962963, .472826086957, .242362525458]

def get_title(title):
    x = title.split(',')[1]
    x = x.split('.')[0]
    x = x.strip()
    diff_title[x] = 1
    return x

def new_title(title):
    if title == 'Mr' or title == 'Capt' or title == 'Don' or title == 'Dr' or title == 'Jonkheer' or title == 'Major' or \
                    title == 'Rev' or title == 'Sir' or title == 'Col':
        return 'Mr'
    elif title == 'Mrs' or title == 'Lady' or title == 'Mme' or title == 'Ms' or title == 'the Countess':
        return 'Mrs'
    elif title == 'Miss' or title == 'Mlle':
        return 'Miss'
    else:
        return 'Master'

def changeAge(x):
    if x <= 13:
        return 0
    elif x <19:
        return 1
    elif x < 49:
        return 2
    else:
        return 3

def test_data(train_data):
    data = pd.read_csv(train_data, header=0)
    global female_ratio, male_ratio
    female_survived = len(data[(data['Sex'] == 'female') & (data['Survived'] == 1)])
    female_total = len(data[data['Sex'] == 'female'])
    female_ratio = female_survived * 1.0 / female_total
    male_survived = len(data[(data['Sex'] == 'male') * (data['Survived'] == 1)])
    male_total = len(data[data['Sex'] == 'male'])
    male_ratio = male_survived * 1.0 / male_total
    print male_ratio, male_survived, male_total
    print female_ratio, female_survived, female_total
    for i in range(1,4):
        total =len(data[data.Pclass == i])
        num = len(data[(data.Pclass == i) & (data.Survived == 1)])
        print 'PClass {} survived ratio is {}'.format(i, num * 1.0 / total)

def process_data(test_data):
    data = pd.read_csv(test_data, header = 0)
    global ids, female_ratio, male_ratio
    ids = data.values[:,0]
    data.Embarked[data.Embarked.isnull()] = data.Embarked.dropna().mode().values

    # data.Age[data.Age.isnull()] = data.Age.dropna().mean()
    data['sex_survived'] = data.Sex.map(lambda x:female_ratio if x =='female' else male_ratio)
    # data['is_child'] = data.Age.map(lambda x: 1 if x <= 13 else 0)
    data['AgeCat'] = data.Age.map(lambda x:changeAge(x))
    data['title'] = data.Name.map(lambda x:get_title(x))
    data.title = data.title.map(lambda x:new_title(x))
    # data['pclass_survived'] = data.Pclass.map(lambda x:pclass_survived_ratio[x - 1])
    global diff_age
    diff_age['Mr'] = data[(data['title'] == 'Mr')].Age.dropna().mean()
    diff_age['Miss'] = data[(data['title'] == 'Miss')].Age.dropna().mean()
    diff_age['Mrs'] = data[(data['title'] == 'Mrs')].Age.dropna().mean()
    diff_age['Master'] = data[(data['title'] == 'Master')].Age.dropna().mean()
    # print diff_age
    data.loc[(data.title == 'Mr') & (data.Age.isnull()), 'Age'] = diff_age['Mr']
    data.loc[(data.title == 'Miss') & (data.Age.isnull()), 'Age'] = diff_age['Miss']
    data.loc[(data.title == 'Mrs') & (data.Age.isnull()), 'Age'] = diff_age['Mrs']
    data.loc[(data.title == 'Master') & (data.Age.isnull()), 'Age'] = diff_age['Master']
    data.title = data.title.map(lambda x:title[x])
    data.Sex = data.Sex.map(lambda x: 1 if x == 'male' else 0)
    data['Family_Size'] = data.SibSp + data.Parch + 1
    for i in range(1,4):
        print data[data.Pclass == i].Fare.dropna().mean()
        data.loc[(data.Pclass == i) & (data.Fare.isnull()), 'Fare'] = data[(data.Pclass == i)].Fare.dropna().mean()
    data['FPP'] = data.Fare / data.Family_Size
    global port
    data.Embarked = data.Embarked.map(lambda x:port[x])
    data['is_alone'] = data.Family_Size.map(lambda x:0 if x == 1 else 1)
    data = data.drop(['Name', 'PassengerId', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1)
    # print len(data.columns)
    # print data.values[0::,1] * data.values[0::,2]
    x = len(data.columns)
    for i in range(x):
        for j in range(x):
            if str(data.columns[i]) == 'Survived' or str(data.columns[j]) == 'Survived':
                continue
            name = str(data.columns[i]) + str(data.columns[j])
            data[name] = data.values[0::, i] * data.values[0::, j]
    # data.Fare[data.Fare.isnull()] = data.Fare.dropna().mean()

    # print data.head(20)
    # print data.info()
    return data.values

def WriteFile(name, out):
    predictions_file = open(name, "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId","Survived"])
    global ids
    open_file_object.writerows(zip(ids, out))
    predictions_file.close()

def RandomForest(train_data, test_data):
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_data[0::, 1::], train_data[0::,0])
    out = forest.predict(test_data).astype(int)
    WriteFile('./result/randomforest.csv', out)

def Linear(train_data, test_data):
    linear = linear_model.LinearRegression()
    linear.fit(train_data[0::, 1::] , train_data[0::, 0])
    out = linear.predict(test_data).astype(int)
    WriteFile('./result/linearegressio.csv', out)

def Knn(train_data, test_data):
    knn = neighbors.KNeighborsClassifier()
    knn.fit(train_data[0::, 1::], train_data[0::, 0])
    out = knn.predict(test_data).astype(int)
    WriteFile('./result/Knn.csv', out)

def Svm(train_data, test_data, mode):
    svm_model = svm.SVC(kernel = mode)
    svm_model.fit(train_data[0::, 1::], train_data[0::, 0])
    out = svm_model.predict(test_data).astype(int)
    WriteFile('./result/Svm{}.csv'.format(mode), out)

if __name__ == '__main__':
    # get_title('Ahlin, Mrs. Johan (Johanna Persdotter Larsson)')
    test_data('./data/train.csv')
    print 'training...........'
    train_data = process_data('./data/train.csv')
    test_data = process_data('./data/test.csv')
    print 'predict...........'
    RandomForest(train_data, test_data)
    print 'done............'
    # Linear(train_data, test_data)
    # Knn(train_data, test_data)
    # Svm(train_data, test_data, 'linear')
    # Svm(train_data, test_data, 'rbf')
    # for k in diff_title.keys():
    #     print k

    # # print time.time()
    # # # print train_data.describe()

    # score = []
    # for tree in range(1,200):
    #     model = RandomForestClassifier(n_estimators=tree)
    #     model.fit(train_data[0::,1::], train_data[0::, 0])
    #     score.append(model.score(train_data[0::, 1::] , train_data[0::, 0]))
    # plt.plot(range(1,200), score, '-r')
    # plt.xlabel('Tree_Num')
    # plt.ylabel('Score')
    # plt.title('Tree_Num & Score Pic')
    # plt.show()
    # plt.savefig('model.png')