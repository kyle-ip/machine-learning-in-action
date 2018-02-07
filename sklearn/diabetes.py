# -*- coding: utf-8 -*-
# @Time    : 2018/2/7
# @Author  : yipwinghong
# @Email   : yipwinghong@outlook.com
# @File    : diabetes.py    糖尿病线性回归测试（广义线性回归模型）
# @Software: PyCharm

"""
    其他回归模型：
        岭回归         采用带罚项的残差平方和损失函数
        Lasso回归      采用待L1范数的罚项平方损失函数
        贝叶斯岭回归
        随机梯度下降回归
        鲁棒回归
        ...
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import datasets
from sklearn import cross_validation                    # 划分数据集

# 导入数据集
diabetes = datasets.load_diabetes()
feature, target = diabetes.data[:, np.newaxis, 2], diabetes.target               # 选取其中一个特征
training_feature, test_feature, training_target, test_target = cross_validation.train_test_split(
    feature, target, test_size=0.3, random_state=56
)                                                       # 划分测试集和训练集

# 训练模型
model = linear_model.LinearRegression()                 # 线性回归模型（最小二乘回归，即使用平方损失函数）
model.fit(training_feature, training_target)            # 拟合样本点（特征向量）
w, b = model.coef_, model.intercept_                    # 拟合直线的权值和偏置（y(x) = wx + b）

# 绘制图像
plt.scatter(training_feature, training_target, color="black")                       # 训练集散点
plt.scatter(test_feature, test_target, color="red")                                 # 测试集散点
plt.plot(test_feature, model.predict(test_feature), color="blue", linewidth=3)      # 拟合直线
plt.legend(("Fit line", "Train Set", "Test Set"), loc="lower right")                # 图例
plt.title("LinearRegression Example")                                               # 标题
plt.xticks(())                                                                      # 不显示刻度
plt.yticks(())
plt.show()
