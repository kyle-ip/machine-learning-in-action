"""
    Logistic回归

    优点：计算代价不高，易于理解和实现
    缺点：容易欠拟合，分类精度可能不高
    适用数据类型：数值型和标称型数据

"""

from numpy import *
import matplotlib.pyplot as plt


def loadDataSet():
    """ 创建示例数据集 """

    dataMat, labelMat = [], []
    for line in open('testSet.txt').readlines():            # 3列（表示两个特征）、100行的数据集
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])     # 第一个补位
        labelMat.append(int(lineArr[2]))
    return array(dataMat), array(labelMat)


def sigmoid(inX):
    """
    sigmoid函数 传入参数为向量内积（输入向量*权值向量）
    :param inX: 
    :return: 
    """

    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels, alpha=0.001, maxCycles=150):
    """
    梯度上升法：每次更新回归系数需要遍历整个数据集，批处理算法
    :param dataMatIn: 
    :param classLabels:
    :param alpha:       学习率
    :return: 
    """

    dataMatrix = mat(dataMatIn)                     # 转换100 * 3矩阵：3个特征（2个可用），100行数据
    labelMat = mat(classLabels).transpose()         # 转换100 * 1矩阵：转置
    m, n = shape(dataMatrix)                        # 获取矩阵行数、列数
    weights = ones((n, 1))                                      # 把权值初始化为3 * 1矩阵
    for _ in range(maxCycles):                                  # TODO 根据偏差调整权值，迭代计算500次
        error = labelMat - sigmoid(dataMatrix * weights)        # 计算预测结果与实际值的偏差
        weights += alpha * dataMatrix.transpose() * error       # 梯度上升迭代计算权值（行列转置）
    return weights                                              # 把权值矩阵转化为数组


def plotBestFit(weights):
    """ 绘制图像测试 """

    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n, _ = shape(dataArr)

    xcord1, xcord2 = [], []                     # 两组列表，分别存放0类、1类的x、y坐标
    ycord1, ycord2 = [], []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])

    fig = plt.figure()                          # 配置图像参数
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stocGradAscent0(dataMatrix, classLabels, alpha=0.01):
    """
    随机梯度上升法：增量更新（每次仅用一个样本点、即一行更新回归系数），在线学习算法
    :param dataMatrix: 
    :param classLabels: 
    :return: 
    """

    m, n = shape(dataMatrix)
    weights = ones(n)
    for i in range(m):
        error = classLabels[i] - sigmoid(sum(dataMatrix[i] * weights))
        weights += alpha * error * dataMatrix[i]            # 根据偏差调整权值
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=30):
    """
    改进的随机梯度上升法：1，多次迭代；2，随机抽取；3，根据迭代次数修改学习率
    :param dataMatrix: 
    :param classLabels: 
    :param numIter:         迭代次数
    :return: 
    """

    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):                    # 迭代计算numIter次
        dataIndex = list(range(m))
        random.shuffle(dataIndex)               # 打乱样本随机抽取
        for i in dataIndex:                     # 逐行样本插入训练
            alpha = 4 / (1.0 + j + i) + 0.0001  # 学习率随迭代次数和样本数增加而减小（当j<<max(i)时，alpha就不是严格下降的）
            error = classLabels[i] - sigmoid(sum(dataMatrix[i] * weights))
            weights += + alpha * error * dataMatrix[i]
    return weights


def classifyVector(inX, weights):
    """ 分类器函数 """
    return 1 if sigmoid(sum(inX * weights)) > 0.5 else 0


# 测试：从疝气病症预测病马死亡率
def colicTest():
    """ 预测病马死亡率 """
    trainingSet = []
    trainingLabels = []
    for line in open('horseColicTraining.txt').readlines():
        currLine = line.strip().split('\t')
        lineArr = [float(currLine[i]) for i in range(21)]
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))

    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)    # 随机梯度下降计算logist回归函数权值
    errorCount, numTestVec = 0, 0.0
    for line in open('horseColicTest.txt').readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = [float(currLine[i]) for i in range(21)]
        if classifyVector(array(lineArr), trainWeights) != int(currLine[-1]):        # 统计分类失败次数
            errorCount += 1
    errorRate = errorCount / numTestVec
    print("the error rate of this source is: {0}".format(errorRate))
    return errorRate


def multiTest():
    """ 统计多次测试的失败率 """

    numTests, errorSum = 10, 0.0
    errorSum = sum([colicTest() for _ in range(numTests)])
    print("after %d iterations the average error rate is: {0}".format(numTests, errorSum / float(numTests)))


if __name__ == "__main__":

    # dataArr, labelMat = loadDataSet()
    # w = stocGradAscent1(dataArr, labelMat)

    # w = gradAscent(dataArr, labelMat)
    # plotBestFit(w)
    multiTest()