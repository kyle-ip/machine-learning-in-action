# -*- coding: utf-8 -*-
# @Time    : 2018/2/7
# @Author  : yipwinghong
# @Email   : yipwinghong@outlook.com
# @File    : digits.py      支持向量机分类测试（手写数字识别）
# @Software: PyCharm

import os
import pandas as pd
from sklearn import cross_validation
from sklearn.svm import LinearSVC, SVC              # 支持向量机模型
from sklearn.linear_model import Perceptron         # 感知机模型
from sklearn.metrics import accuracy_score          # 性能评估
from sklearn import datasets                        # 数据集
import matplotlib.pyplot as plt


path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 线性分类测试
# df = pd.read_csv(os.path.join(path, "data/svm_test_data.csv"), header=0)
# feature, target = df[["x", "y"]], df["class"]       # 提取样本点特征和分类

feature, target = datasets.make_classification(     # 随机生成二分类数据集
    n_samples=500,                                  # 500个样本
    n_features=2,                                   # 2个特征（等于n_informative + n_redundant + n_repeated）
    n_redundant=0,                                  # 0个冗余特征
    n_informative=1,                                # 1个多信息特征
    n_clusters_per_class=1,                         # 1个簇/类别
    n_classes=2                                     # 2个类别输出
)

training_feature, test_feature, training_target, test_target = cross_validation.train_test_split(
    feature, target, test_size=0.7
)

model1 = Perceptron()
model1.fit(training_feature, training_target)
score1 = model1.score(test_feature, test_target)

model2 = LinearSVC()
model2.fit(training_feature, training_target)
score2 = model2.score(test_feature, test_target)


# 手写数字识别测试
digits = datasets.load_digits()


for i, image in enumerate(digits.images[:5]):
    plt.subplot(2, 5, i + 1)                    # 选取前5个手写数字打印灰度图
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
# plt.show()

feature, target = digits.data, digits.target
training_feature, test_feature, training_target, test_target = cross_validation.train_test_split(
    feature, target, test_size=0.3
)       # 随机划分数据集，70%作为训练集，30%作为测试集

model = SVC(gamma=0.001)
model.fit(training_feature, training_target)
prediction = model.predict(test_feature)
score = accuracy_score(test_target, prediction)
print(score)
