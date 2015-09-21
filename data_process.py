# coding=utf8
__author__ = 'kdq'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def test_data(train_data):
    data = pd.read_csv(train_data, header=0)
    female_survived = len(data[(data['Sex'] == 'female') & (data['Survived'] == 1)])
    female_total = len(data[data['Sex'] == 'female'])
    female_ratio = female_survived * 1.0 / female_total
    male_survived = len(data[(data['Sex'] == 'male') * (data['Survived'] == 1)])
    male_total = len(data[data['Sex'] == 'male'])
    male_ratio = male_survived * 1.0 / male_total
    age = data.loc[data.Age.isnull() == 0, 'Age']
    age1 = data.loc[(data.Age.isnull() == 0) & (data.Survived == 1) , 'Age']
    age_list = np.zeros(100)
    age_survived = np.zeros(100)
    sum = 0
    for i in age:
        age_list[i] += 1
    for i in age1:
        age_survived[i] += 1
    # 处理age_survived / age_list = nan ,即age_list = 0的情况
    for i in age:
        age_list[i] += 0.1

    x = list()
    for i in range(1,4):
        total =len(data[data.Pclass == i])
        num = len(data[(data.Pclass == i) & (data.Survived == 1)])
        x.append(num * 1.0 / total)
        # print 'PClass {} survived ratio is {}'.format(i, num * 1.0 / total)

    print data.info()
    # 绘制不同Pclass的幸存概率
    plt.figure(1)
    plt.plot(range(1,4), x)
    plt.xlabel('Pclass')
    plt.ylabel('Survived Ratio')
    plt.title('Survived Ratio - Pclass')

    # 绘制男女幸存数量图
    plt.figure(2)
    plt.bar([1,2], [male_total, female_total], alpha = .5,color='b')
    plt.bar([1,2] , [male_survived, female_survived], alpha = .5,color='r')
    # plt.plot([1,2], [male_ratio, female_ratio], alpha= .5,color = 'g')
    plt.xlabel('Male or Female')
    plt.ylabel('Num Survived & Total')

    # 绘制各年龄幸存数量图
    plt.figure(3)
    # plt.plot(range(1, 101), age_list, 'r-')
    plt.bar(range(1, 101), age_list, color='r')
    # plt.plot(range(1, 101), age_survived, 'g-')
    plt.bar(range(1, 101), age_survived, color='g')
    plt.xlabel('Age')
    plt.ylabel('Num')
    plt.title('Age-Survived')

    # 绘制各年龄幸存比例图
    plt.figure(4)
    plt.plot(range(1, 101, ) ,age_survived / age_list, 'r')
    plt.xlabel('Age')
    plt.ylabel('Survived Ratio')
    plt.title('Survived Ratio - Age')
    plt.show()

if __name__ == '__main__':
    test_data('./data/train.csv')
