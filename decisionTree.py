from collections import Counter, defaultdict
from math import log
import os
import pickle
import operator

from Ch03.treePlotter import *

"""
    决策树：内部节点表示一个特征/属性，叶节点表示一个类
        从根到叶每条路径构建一条规则，路径和规则集合互斥且完备
        表示给定特征条件下类的条件概率分布（P(Y|X)：X特征下类为Y的概率）
    复杂度：叶节点的个数，复杂度越高拟合程度越高（分类条件越苛刻、分类越细）
    生成：考虑局部（当前选取信息增益最大的节点）最优（贪心算法，ID3、C4.5）
    剪枝：考虑全局最优（避免分类过度，过拟合，CART）
    
"""


def createDataSet():
    """ 创建示例数据集 """

    dataSet = [
        [1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'],
        [0, 1, 'no'], [0, 1, 'no']
    ]                                                       # 前两项为属性值，最后一项为分类（正例和反例）
    labels = ['no surfacing', 'flippers']                   # 特征：属性值对应的label
    return dataSet, labels


def calcShannonEnt(dataSet):
    """ 计算香农熵：即信息的期望，表示数据分布的混乱程度，与数据取值无关 """

    labelCounts = Counter([i[-1] for i in dataSet])
    shannonEnt = 0.0                                        # 统计正例和反例个数
    for _, v in labelCounts.items():
        prob = float(v) / len(dataSet)                      # 信息：选择该分类的概率的负对数（概率越大，香农熵越小，混合数据越少）
        shannonEnt += prob * log(prob, 2)
    return -shannonEnt


def splitDataSet(dataSet, axis, value):
    """
    划分数据集：取数据集中第axis项特征值为value的项组成子集（不含该特征）
    :param dataSet:     数据集
    :param axis:        划分数据集的特征
    :param value:       特征的属性值
    :return:
    """
    return [
        featVec[:axis] + featVec[axis + 1:]
        for featVec in dataSet
        if featVec[axis] == value
    ]


def chooseBestFeatureToSplit(dataSet):
    """ 求划分后信息增益最大的特征 """

    baseEntropy = calcShannonEnt(dataSet)                   # 原始香农熵
    infoGainList = []                                       # 记录最大划分信息增益（熵或数据无序度减少的程度）及其特征编号

    for i in range(len(dataSet[0]) - 1):                    # 逐列取特征，其中最后一列为分类（正例和反例）
        featList = {example[i] for example in dataSet}      # 存放一列属性值的列表
        newEntropy = 0.0
        for value in featList:                              # 依当前特征、逐个属性值划分数据集
            subDataSet = splitDataSet(dataSet, i, value)    # 划分香农熵 = sum(划分子集的概率 * 子集熵)，即条件熵
            newEntropy += len(subDataSet) / float(len(dataSet)) * calcShannonEnt(subDataSet)
        # infoGainList.append((baseEntropy - newEntropy, i))    # 信息增益
        infoGainList.append((newEntropy / baseEntropy, i))      # 信息增益比

    return sorted(infoGainList, key=lambda x: x[0], reverse=True)[0][1]


def majorityCnt(classList):
    """ 多数表决决定叶子节点分类 """
    return Counter(classList).most_common(1)[0]             # classCount[vote] = classCount.setdefault(vote, 0) + 1


def createTree(dataSet, labels):
    """
    创建决策树（以当前最佳特征为根，向下逐层用特征节点构造子树）
    :param dataSet:     数据集
    :param labels:      特征labels列表
    :return:
    """

    classList = [example[-1] for example in dataSet]        # 取当前数据集的所有正例和反例信息

    if classList.count(classList[0]) == len(classList):     # 类别完全相同，停止划分
        return classList[0]

    if len(dataSet[0]) == 1:                                # 遍历完所有特征，返回出现最多的类别
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)            # 选取最佳特征（信息增益最大的）
    bestFeatLabel = labels[bestFeat]

    del(labels[bestFeat])                                   # 每次递归调用都删除当前最佳特征的标签，以免子树中重复出现该特征
    featValues = {i[bestFeat] for i in dataSet}             # 取该特征下的所有属性值

    myTree = {bestFeatLabel: {}}
    for value in featValues:                                # 遍历所有属性值来创建子集，并对子集递归调用创建子树
        myTree[bestFeatLabel][value] = createTree(
             (dataSet, bestFeat, value), labels[:]
        )

    return myTree


def retrieveTree(i):
    """ 测试决策树 """

    listOfTrees = [
        {
            "no surfacing": {
                0: "no",
                1: {
                    "flippers": {
                        0: "no",
                        1: "yes"}
                }
            }
        },
        {
            "no surfacing": {
                0: "no",
                1: {
                    "flippers": {
                        0: {
                            "head": {
                                0: "no",
                                1: "yes"
                            }
                        },
                        1: "no"}
                }
            }
        }
    ]

    return listOfTrees[i]


def classify(inputTree, featLabels, testVec):
    """ 使用决策树实现分类 """

    firstStr = list(inputTree.keys())[0]        # 获取特征及其下子树和叶节点
    secondDict = inputTree[firstStr]

    key = testVec[featLabels.index(firstStr)]   # 取该特征的在标签列表中的下标，并通过标签下标取测试向量的属性值
    valueOfFeat = secondDict[key]               # 在树中取该属性值对应的决策

    return classify(valueOfFeat, featLabels, testVec) \
        if isinstance(valueOfFeat, dict) else valueOfFeat


def storeTree(inputTree, filename):
    """ 存储决策树：毋须每次使用都重新创建 """

    with open(filename, 'w') as f:
        pickle.dumps(inputTree, f)


def grabTree(filename):
    """ 读取决策树 """

    return pickle.load(open(filename))


def lensesClassTest(path):
    """ 隐形眼镜分类测试 """

    with open(os.path.join(path, "lenses.txt")) as f:
        lenses = [line.strip().split("\t") for line in f.readlines()]
    lensesLabels = ["age", "prescript", "astigmatic", "tearRate"]
    lensesTree = createTree(lenses, lensesLabels)

    return lensesTree


if __name__ == "__main__":
    myDat, labels = createDataSet()

    # print(calcShannonEnt(myDat))
    # print(splitDataSet(myDat, 0, 1))
    # print(chooseBestFeatureToSplit(myDat))
    # print(createTree(myDat, labels))

    # getNumLeafs({'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}})
    # getTreeDepth({'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}})
    # print(getTreeDepth(retrieveTree(0)))

    # mytree = retrieveTree(0)
    # print(classify(mytree, labels, [1, 0]))

    tree = lensesClassTest("D:/machinelearninginaction/Ch03")
    createPlot(tree)