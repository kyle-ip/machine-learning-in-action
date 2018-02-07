# -*- coding: utf-8 -*-
# @Time    : 2018/2/7
# @Author  : yipwinghong
# @Email   : yipwinghong@outlook.com
# @File    : digits.py          支持向量机分类测试
# @Software: PyCharm

import os
import pandas as pd
from sklearn import cross_validation
from sklearn.svm import LinearSVC                   # 线性支持向量机
from sklearn.linear_model import Perceptron         # 感知机

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 线性分类测试
df = pd.read_csv(os.path.join(path, "data/svm_test_data.csv"), header=0)
feature, target = df[["x", "y"]], df["class"]       # 提取样本点特征和分类

training_feature, test_feature, training_target, test_target = cross_validation.train_test_split(
    feature, target, test_size=0.7
)

model1 = Perceptron()
model1.fit(training_feature, training_target)
score1 = model1.score(test_feature, test_target)

model2 = LinearSVC()
model2.fit(training_feature, training_target)
score2 = model2.score(test_feature, test_target)

# 