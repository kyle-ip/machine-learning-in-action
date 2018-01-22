from numpy import *
import matplotlib.pyplot as plt


def loadDataSet():
    """ 创建示例数据集 """

    dataMat, labelMat = [], []
    for line in open('testSet.txt').readlines():        # 3列、100行的数据集
        lineArr = line.strip().split()
        dataMat.append(
            [1.0, float(lineArr[0]), float(lineArr[1])]
        )
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    """ sigmoid函数 """

    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):
    """
    梯度上升法：每次更新回归系数需要遍历整个数据集，批处理算法
    :param dataMatIn: 
    :param classLabels: 
    :return: 
    """

    alpha = 0.001

    dataMatrix = mat(dataMatIn)                 # 转换100 * 3矩阵：3个特征（2个可用），100行数据
    labelMat = mat(classLabels).transpose()     # 转换100 * 1矩阵：转置
    m, n = shape(dataMatrix)                    # 获取矩阵行数、列数

    weights = ones((n, 1))                      # 把权值初始化为3 * 1矩阵

    # TODO 求梯度gradient = dataMatrix.transpose() * (labelMat - sigmoid(dataMatrix * weights))
    for _ in range(500):                        # 迭代计算500次
        h = sigmoid(dataMatrix * weights)       # 把训练矩阵*权值矩阵结果（100*3 x 3*1）置入sigmoid函数
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error  # 梯度上升迭代计算权值

    return weights.getA()                       # 把权值矩阵转化为数组


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


def stocGradAscent0(dataMatrix, classLabels):
    """
    随机梯度上升法：增量更新（每次仅用一个样本点更新回归系数），在线学习算法
    :param dataMatrix: 
    :param classLabels: 
    :return: 
    """

    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)       # initialize to all ones
    # TODO 求梯度gradient = (classLabels[i] - h) * (classLabels[i] - sigmoid(sum(dataMatrix[i] * weights)))
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))       # 逐行取训练矩阵为向量，和权值向量的点积置入sigmoid函数
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights



def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)  # initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.0001  # alpha decreases with iteration, does not
            randIndex = int(random.uniform(0, len(dataIndex)))  # go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])
    return weights


dataArr, labelMat = loadDataSet()
w = stocGradAscent0(dataArr, labelMat)

# w = gradAscent(dataArr, labelMat)
plotBestFit(w)


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = [];
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0;
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print
    "the error rate of this test is: %f" % errorRate
    return errorRate


def multiTest():
    numTests = 10;
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print
    "after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests))


# if __name__ == "__main__":
    # dataArr, labelMat = loadDataSet()
    # gradAscent(dataArr, labelMat)