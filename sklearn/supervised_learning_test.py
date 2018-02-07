# -*- coding: utf-8 -*-
# @Time    : 2018/2/7
# @Author  : yipwinghong
# @Email   : yipwinghong@outlook.com
# @File    : supervised_learning_test.py
# @Software: PyCharm

import os

# 数据处理
import pandas as pd
import numpy as np

# 图像绘制
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

# 学习模型
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

# 集成学习算法
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.gaussian_process import GaussianProcessClassifier      # 高斯过程分类器

from sklearn.linear_model import PassiveAggressiveClassifier        # 广义线性分类器
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier                  # K近邻分类器

from sklearn.naive_bayes import GaussianNB                          # 朴素贝叶斯分类器

from sklearn.neural_network import MLPClassifier                    # 神经网络分类器

from sklearn.tree import DecisionTreeClassifier                     # 决策树分类器
from sklearn.tree import ExtraTreeClassifier

from sklearn.svm import SVC                                         # 支持向量机分类器
from sklearn.svm import LinearSVC


path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = pd.read_csv(os.path.join(path, "data/class_data.csv"), header=0)

feature, target = data[['X', 'Y']], data['CLASS']

training_feature, test_feature, training_target, test_target = train_test_split(
    feature, target, test_size=.3
)

cm_color = ListedColormap(["red", "blue"])
plt.scatter(
    data['X'], data['Y'],
    c=data['CLASS'], cmap=cm_color
)
# plt.show()

# 比较多种分类器
models = {
    'AdaBoost': AdaBoostClassifier(),
    'Bagging': BaggingClassifier(),
    'ExtraTrees': ExtraTreesClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'RandomForest': RandomForestClassifier(),
    'GaussianProcess': GaussianProcessClassifier(),
    'PassiveAggressive': PassiveAggressiveClassifier(),
    'Ridge': RidgeClassifier(),
    'SGD': SGDClassifier(),
    'KNeighbors': KNeighborsClassifier(),
    'GaussianNB': GaussianNB(),
    'MLP': MLPClassifier(),
    'DecisionTree': DecisionTreeClassifier(),
    'ExtraTree': ExtraTreeClassifier(),
    'SVC': SVC(),
    'LinearSVC': LinearSVC()
}


# for name, model in models.items():
#     model.fit(training_feature, training_target)
#     prediction = model.predict(test_feature)
#     score = accuracy_score(test_target, prediction)
#     print("{0}\t{1}".format(name, score))

# 绘制热力图
i = 1
cm = plt.cm.Reds
cm_color = ListedColormap(['red', 'yellow'])

# 栅格化
x_min, x_max = data['X'].min() - .5, data['X'].max() + .5
y_min, y_max = data['Y'].min() - .5, data['Y'].max() + .5
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, .1), np.arange(y_min, y_max, .1)
)

for name, model in models.items():
    ax = plt.subplot(4, 4, i)
    model.fit(training_feature, training_target)
    prediction = model.predict(test_feature)
    score = accuracy_score(test_target, prediction)

    if hasattr(model, "decision_function"):
        z = model.decision_function(
            np.c_[xx.ravel(), yy.ravel()]
        )
        print("decision_function", model)
    else:
        z = model.predict_proba(
            np.c_[xx.ravel(), yy.ravel()]
        )[:, 1]

    # 绘制决策边界热力图
    z = z.reshape(xx.shape)
    ax.contourf(xx, yy, z, cmap=cm, alpha=.6)

    # 绘制训练集和测试集
    ax.scatter(training_feature['X'], training_feature['Y'], c=training_target, cmap=cm_color)
    ax.scatter(test_feature['X'], test_feature['Y'], c=test_target, cmap=cm_color, edgecolors='black')

    # 图形样式设定
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title('%s | %.2f' % (name, score))

    i += 1

plt.show()