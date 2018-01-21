import operator
import os
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
from numpy import *

"""
    K近邻：
        1、训练集
        2、距离度量Lp（如欧氏距离L2）
        3、k值（k值大，模型简单，估计误差小，近似误差大，当k=样本空间数时）
        4、分类规则（如多数表决，即经验风险最小化）
    对于任何一个新的输入实例它所属的类唯一地确定：
        1、寻找近邻点（线性扫描：效率低，kd树：适合训练实例数远大于空间维数的情况）
        2、多数表决确定分类
"""


def classify0(inX, dataSet, labels, k):
    """
    k-近邻算法分类器（线性扫描）
    :param inX:         输入向量
    :param dataSet:     训练集
    :param labels:      标签向量
    :param k:           k值
    :return:
    """

    # 对属性值作距离计算
    dataSetSize = dataSet.shape[0]                      # 取数据集行数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet     # 以输入向量创建dataSetSize行、重复1次矩阵（把inX纵向复制dataSetSize-1次组成矩阵）
    distances = ((diffMat ** 2).sum(axis=1)) ** 0.5     # 求输入向量与训练集各点的距离（sum对行求和，同行每列相加化为一列，axis=0则对列求和）
    sortedDistIndicies = argsort(distances)             # 对距离从小到大排序，返回距离排序下标的数组（2, 3, 1, 0）

    classCount = Counter(
        [labels[sortedDistIndicies[i]] for i in range(k)]
    )                                                   # 统计距离排名前k的已知数据点的类别出现次数，并取出现频率最高的预测分类

    return classCount.most_common()[0][0]


def createDataSet():
    """ 创建示例数据集 """

    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def file2matrix(filename):
    """ 读取文件到特征矩阵 """

    lines = open(filename).readlines()

    numberOfLines = len(lines)                  # 取文件行数
    returnMat = zeros((numberOfLines, 3))       # 取文件每行三个字段构造矩阵（初始化为0）
    classLabelVector = []                       # 返回数据点对应的标签

    for index, line in enumerate(lines, 0):
        listFromLine = line.strip().split('\t')
        returnMat[index, :] = listFromLine[0: 3]        # 取每行前三个字段赋到矩阵
        classLabelVector.append(listFromLine[-1])       # 最后一个字段作为标签

    return returnMat, classLabelVector


def autoNorm(dataSet):
    """
    归一化
    :param dataSet:     数据集
    :return:
    """

    minVals, maxVals = dataSet.min(0), dataSet.max(0)   # 选取每列最大、最小值组成一行
    ranges = maxVals - minVals                          # 每列极差组成的行
    m = dataSet.shape[0]
    normDataSet = (dataSet - tile(minVals, (m, 1))) / tile(ranges, (m, 1))

    return normDataSet, ranges, minVals


def datingClassTest(filename):
    """
    分类器测试
    :param filename:
    :return:
    """

    datingDataMat, datingLabels = file2matrix(filename)     # 读取数据集并作归一化处理
    normMat, ranges, minVals = autoNorm(datingDataMat)

    m = normMat.shape[0]
    hoRatio = 0.10                                          # 取其中10%作为测试集
    numTestVecs = int(m * hoRatio)

    trainingMat = normMat[numTestVecs: m, :]
    trainingLabels = datingLabels[numTestVecs: m]

    errorCount = 0                                          # 测试数据，记录分来错误数
    for i in range(numTestVecs):
        classifierResult = classify0(                       # 从测试集选出输入向量、训练集、标签向量，输入到分类器
            normMat[i, :], trainingMat, trainingLabels, 3
        )
        print("the classifier came back with: {0}, the real answer is: {1}".format(
            classifierResult, datingLabels[i]
        ))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0

    return errorCount / float(numTestVecs)


def classifyPerson(filename):
    """
    约会网站预测
    :param filename:
    :return:
    """

    percentTats = input("percentage of time spent playing video games? ")
    ffMiles = input("frequent flier miles earned per year? ")
    iceCream = input("liters of ice cream consumed per year? ")
    datingDataMat, datingLabels = file2matrix(filename)
    normMat, ranges, minVals = autoNorm(datingDataMat)

    inArr = array([ffMiles, percentTats, iceCream]).astype(float)
    classifierResult = classify0(
        (inArr - minVals)/ranges, normMat, datingLabels, 3
    )

    print("You will probably like this person: " + classifierResult)


def img2vector(filename):
    """
    把图像文件转化为向量
    :param filename:
    :return:
    """
    returnVect = zeros((1, 1024))
    lines = open(filename).readlines()
    for i in range(32):                                     # 读取文件前32行
        for j in range(32):                                 # 把每行头31个字符存储在数组
            returnVect[0, 32 * i + j] = int(lines[i][j])

    return returnVect


def handwritingClassTest(path, k=3):
    """
    手写字体识别
    数据集存放由手写字体转化而来的0-1矩阵，
    分类器读取训练集数据并逐行存放文件数据（为长度为1024的数组）到特征矩阵，同时记录对应的label
    当测试时读取测试集文件为一个数组（输入向量）并与输入到分类器
    :param path:    文件根目录
    :return:
    """
    hwLabels = []

    # 读取训练集
    trainingFileList = os.listdir(os.path.join(path, 'trainingDigits'))
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        # fileStr = fileNameStr.replace(".txt", "")
        hwLabels.append(int(fileNameStr.split('_')[0]))                     # 从文件名解析分类数字
        trainingMat[i, :] = img2vector('trainingDigits/' + fileNameStr)     # 把文件字符读取到矩阵

    # 读取测试集
    testFileList = os.listdir(os.path.join(path, 'testDigits'))
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumStr = int(fileNameStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/' + fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, k)

        # print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0

    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))


if __name__ == "__main__":

    filename = "D:/machinelearninginaction/Ch02/datingTestSet2.txt"

    # group, labels = createDataSet()
    # print(classify0([0, 0], group, labels, 3))

    # datingDataMat, datingLabels = file2matrix(filename)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)       # 参数：1行1列1块
    # ax.scatter(
    #     datingDataMat[:, 1], datingDataMat[:, 2],
    #     15.0 * array(datingLabels), 15.0 * array(datingLabels),
    # )
    # plt.grid(True)
    # plt.show()
    # plt.title("K-")
    # plt.xlabel('fly')
    # plt.ylabel('consume')

    # autoNorm(datingDataMat)
    # datingClassTest(filename)

    # filename = "D:/machinelearninginaction/Ch02/datingTestSet.txt"
    # classifyPerson(filename)

    # filename = "D:/machinelearninginaction/Ch02/testDigits/0_13.txt"
    # print(img2vector(filename))

    path = "D:/machinelearninginaction/Ch02/"
    handwritingClassTest(path, 2)