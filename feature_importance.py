__author__ = 'kdq'

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd
import numpy as np
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

def process_train_data(test_data):
    data = pd.read_csv(test_data, header = 0)
    global ids, female_ratio, male_ratio
    ids = data.values[:,0]
    data.loc[(data.Embarked.isnull()), 'Embarked'] = data.Embarked.dropna().mode().values
    # data.Embarked[data.Embarked.isnull()] = data.Embarked.dropna().mode().values
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
    train_data_y = data['Survived']
    data = data.drop(['Survived', 'Name', 'PassengerId', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1)
    global label
    label = data.keys()
    return data.values, train_data_y.values

def process_feature_importance(train_data_x, train_data_y) :
    global label
    forest = RandomForestClassifier(n_estimators=1000)
    forest = forest.fit(train_data_x, train_data_y)
    feature_importance = forest.feature_importances_
    print feature_importance
    print label
    feature_importance = 100. * feature_importance / feature_importance.max()
    index = np.argsort(feature_importance)[::-1]
    print len(label), len(feature_importance)
    for i in range(len(feature_importance)):
        print 'feature',label[index[i]], index[i], feature_importance[index[i]]

    importances_sort = sorted(zip(map(lambda x:feature_importance[x], index),map(lambda x:label[x], index)), key=lambda x:x[0], reverse=False)
    plt.barh(range(len(feature_importance)), map(lambda x:x[0], importances_sort))
    plt.xlabel('Ratio')
    plt.title('Feature Importances')
    plt.yticks(range(len(feature_importance)), map(lambda x:x[1], importances_sort))
    plt.show()

if __name__ == '__main__':
    train_data_x, train_data_y = process_train_data('./data/train.csv')
    process_feature_importance(train_data_x, train_data_y)