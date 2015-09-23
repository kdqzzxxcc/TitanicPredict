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
label = []

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
    # print male_ratio, male_survived, male_total
    # print female_ratio, female_survived, female_total
    for i in range(1,4):
        total =len(data[data.Pclass == i])
        num = len(data[(data.Pclass == i) & (data.Survived == 1)])
        print 'PClass {} survived ratio is {}'.format(i, num * 1.0 / total)

def process_train_data(test_data):
    data = pd.read_csv(test_data, header = 0)
    global ids, female_ratio, male_ratio
    ids = data.values[:,0]
    data.Embarked[data.Embarked.isnull()] = data.Embarked.dropna().mode().values
    # sex_survived:cal the ratio of each sex
    data['sex_survived'] = data.Sex.map(lambda x:female_ratio if x =='female' else male_ratio)
    # AgeCat:map different Age to different class
    data['AgeCat'] = data.Age.map(lambda x:changeAge(x))
    # title: split title
    data['title'] = data.Name.map(lambda x:get_title(x))
    # title: union title
    data.title = data.title.map(lambda x:new_title(x))
    # process missing age : use the mean num of each title
    global diff_age
    diff_age['Mr'] = data[(data['title'] == 'Mr')].Age.dropna().mean()
    diff_age['Miss'] = data[(data['title'] == 'Miss')].Age.dropna().mean()
    diff_age['Mrs'] = data[(data['title'] == 'Mrs')].Age.dropna().mean()
    diff_age['Master'] = data[(data['title'] == 'Master')].Age.dropna().mean()
    data.loc[(data.title == 'Mr') & (data.Age.isnull()), 'Age'] = diff_age['Mr']
    data.loc[(data.title == 'Miss') & (data.Age.isnull()), 'Age'] = diff_age['Miss']
    data.loc[(data.title == 'Mrs') & (data.Age.isnull()), 'Age'] = diff_age['Mrs']
    data.loc[(data.title == 'Master') & (data.Age.isnull()), 'Age'] = diff_age['Master']
    # title : map string to int
    data.title = data.title.map(lambda x:title[x])
    # sex : map string to int
    data.Sex = data.Sex.map(lambda x: 1 if x == 'male' else 0)
    # family_size
    data['Family_Size'] = data.SibSp + data.Parch + 1
    # process missing Fara : use the mean num of each pclass
    for i in range(1,4):
        data.loc[(data.Pclass == i) & (data.Fare.isnull()), 'Fare'] = data[(data.Pclass == i)].Fare.dropna().mean()
    # FPP:Fara per person, total Fare div Family_Size
    data['FPP'] = data.Fare / data.Family_Size
    global port
    # Embarked:map string to int
    data.Embarked = data.Embarked.map(lambda x:port[x])
    # is_alone:
    data['is_alone'] = data.Family_Size.map(lambda x:0 if x == 1 else 1)
    data = data.drop(['Name', 'PassengerId', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1)
    train_data_x = data.values[0::, 1::]
    train_data_y = data.values[0::, 0]
    data = data.drop(['Survived'], axis = 1)
    global label
    label = data.keys()
    return train_data_x, train_data_y

def process_test_data(test_data):
    data = pd.read_csv(test_data, header = 0)
    global ids, female_ratio, male_ratio
    ids = data.values[:,0]
    data.Embarked[data.Embarked.isnull()] = data.Embarked.dropna().mode().values
    # sex_survived:cal the ratio of each sex
    data['sex_survived'] = data.Sex.map(lambda x:female_ratio if x =='female' else male_ratio)
    # AgeCat:map different Age to different class
    data['AgeCat'] = data.Age.map(lambda x:changeAge(x))
    # title: split title
    data['title'] = data.Name.map(lambda x:get_title(x))
    # title: union title
    data.title = data.title.map(lambda x:new_title(x))
    # process missing age : use the mean num of each title
    global diff_age
    diff_age['Mr'] = data[(data['title'] == 'Mr')].Age.dropna().mean()
    diff_age['Miss'] = data[(data['title'] == 'Miss')].Age.dropna().mean()
    diff_age['Mrs'] = data[(data['title'] == 'Mrs')].Age.dropna().mean()
    diff_age['Master'] = data[(data['title'] == 'Master')].Age.dropna().mean()
    data.loc[(data.title == 'Mr') & (data.Age.isnull()), 'Age'] = diff_age['Mr']
    data.loc[(data.title == 'Miss') & (data.Age.isnull()), 'Age'] = diff_age['Miss']
    data.loc[(data.title == 'Mrs') & (data.Age.isnull()), 'Age'] = diff_age['Mrs']
    data.loc[(data.title == 'Master') & (data.Age.isnull()), 'Age'] = diff_age['Master']
    # title : map string to int
    data.title = data.title.map(lambda x:title[x])
    # sex : map string to int
    data.Sex = data.Sex.map(lambda x: 1 if x == 'male' else 0)
    # family_size
    data['Family_Size'] = data.SibSp + data.Parch + 1
    # process missing Fara : use the mean num of each pclass
    for i in range(1,4):
        data.loc[(data.Pclass == i) & (data.Fare.isnull()), 'Fare'] = data[(data.Pclass == i)].Fare.dropna().mean()
    # FPP:Fara per person, total Fare div Family_Size
    data['FPP'] = data.Fare / data.Family_Size
    global port
    # Embarked:map string to int
    data.Embarked = data.Embarked.map(lambda x:port[x])
    # is_alone:
    data['is_alone'] = data.Family_Size.map(lambda x:0 if x == 1 else 1)
    data = data.drop(['Name', 'PassengerId', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1)
    return data.values

def WriteFile(name, out):
    predictions_file = open(name, "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId","Survived"])
    global ids
    open_file_object.writerows(zip(ids, out))
    predictions_file.close()


def RandomForest(train_data_x, train_data_y, test_data):
    forest = RandomForestClassifier(n_estimators=1000)
    forest = forest.fit(train_data_x, train_data_y)
    out = forest.predict(test_data).astype(int)
    # WriteFile('./result/randomforest.csv', out)
    return out

def Linear(train_data_x, train_data_y, test_data):
    linear = linear_model.LinearRegression()
    linear.fit(train_data_x, train_data_y)
    out = linear.predict(test_data).astype(int)
    # WriteFile('./result/linearegressio.csv', out)
    return out

def Knn(train_data_x, train_data_y, test_data):
    knn = neighbors.KNeighborsClassifier()
    knn.fit(train_data_x, train_data_y)
    out = knn.predict(test_data).astype(int)
    # WriteFile('./result/Knn.csv', out)
    return out

def Svm(train_data_x, train_data_y, test_data, mode):
    svm_model = svm.SVC(kernel = mode)
    svm_model.fit(train_data_x, train_data_y)
    out = svm_model.predict(test_data).astype(int)
    # WriteFile('./result/Svm{}.csv'.format(mode), out)
    return out

if __name__ == '__main__':
    # get_title('Ahlin, Mrs. Johan (Johanna Persdotter Larsson)')
    test_data('./data/train.csv')
    print 'training...........'
    train_data_x, train_data_y = process_train_data('./data/train.csv')
    test_data = process_test_data('./data/test.csv')
    print 'predict...........'
    out = RandomForest(train_data_x, train_data_y, test_data)
    WriteFile('./result/RandomForest.csv', out)
    print 'done............'


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
