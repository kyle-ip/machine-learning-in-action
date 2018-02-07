# -*- coding: utf-8 -*-
# @Time    : 2018/2/7
# @Author  : yipwinghong
# @Email   : yipwinghong@outlook.com
# @File    : iris.py    鸢尾花决策树分类测试
# @Software: PyCharm

from sklearn import datasets                                # 数据集
from sklearn import cross_validation                        # 划分数据集的方法
from sklearn.tree import DecisionTreeClassifier             # 决策树分类器
from sklearn.metrics import accuracy_score                  # 评估计算方法查看预测结果的准确度

iris = datasets.load_iris()                                 # 鸢尾花数据集
feature = iris.data                                         # 特征（Sepal length, Sepal width, Petal length, Petal width）
target = iris.target                                        # 分类标签（共三类：0, 1, 2）

# print(target)

training_feature, test_feature, training_target, test_target = cross_validation.train_test_split(
    feature, target, test_size=0.3
)                                                           # 按比例随机划分数据集与测试集（random_state=42指定混乱程度）

model = DecisionTreeClassifier()                       # 创建分类器
model.fit(training_feature, training_target)           # 使用训练集训练模型
res = model.predict(test_feature)                      # 使用模型预测测试集
score = accuracy_score(res, test_target)                    # 比较预测结果与实际分类，得出准确率


print(score)
