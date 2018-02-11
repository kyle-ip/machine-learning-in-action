"""
    支持向量机：
        分类器目标是找到把数据点划分到两边的超平面，且该超平面到最近的数据点（支持向量）的距离最大化
        表示1️⃣充分大的确信度对训练数据分类（对相距超平面最近的点有足够大的确信度分开），使模型有较高泛化能力
    超平面：w * x + b = 0
    函数间隔：yi(w * xi + b)，参数w、b成比例改变时超平面不变
    几何间隔：γ = 函数间隔 / ||w||，表示实例点到超平面带符号的距离（不受w、b成比例改变而改变）

    最大间隔分离超平面：最大化超平面关于训练集的几何间隔γ，约束条件为超平面关于每个训练样本点的几何间隔至少为γ
        max γ
        s.t. yi(w * xi + b)

    优点：泛化错误率低，计算开销不大，结果易解释
    缺点：对参数调节和核函数的选择敏感，原始分类器不加修改仅适用于处理二类问题
    适用数据类型：数值型和标称型数据
"""

from numpy import *
import os
from time import sleep

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def loadDataSet(filename="testSet.txt"):
    """ 创建示例数据集 """

    fileName = os.path.join(path, filename)
    dataMat, labelMat = [], []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))                      # 分类标签为-1和1
    return dataMat, labelMat


def selectJrand(i, m):
    """ 从[0, m]中选取不等于i的随机数 """
    j = i
    while j == i:
        j = random.randint(0, m)
    return j


def clipAlpha(aj, H, L):
    """ 把aj限制在[L, H]范围内 """

    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """
    简化版SMO算法
    :param dataMatIn:
    :param classLabels:
    :param C:               惩罚参数（使间隔尽量大、同时误分类点尽可能少）
    :param toler:           容错率（实际与预测分类的误差需在[-toler, toler]范围内）
    :param maxIter:         取消前最大迭代次数
    :return:
    """

    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))                                     # 原优化问题中的拉格朗日乘子
    iter = 0                                                        # 迭代次数
    while iter < maxIter:
        alphaPairsChanged = 0                                       # alpha修改标记：alpha没有修改则迭代次数+1
        for i in range(m):                                          # 逐行样本处理

            # 代入决策函数预测类型（公式见《统计学习方法》P124）
            k_i = dataMatrix * dataMatrix[i, :].T                     # 内积（可被替换成正定核函数）
            fXi = float(multiply(alphas, labelMat).T * k_i) + b       # 决策函数（multiply：对应元素相乘）
            Ei = fXi - float(labelMat[i])                             # 预测与实际误差值（误差大可对该数据实例的alpha值优化）

            # 优化：超出容错率、且alpha的值不等于C或0，选为第一个alpha值
            if (labelMat[i] * Ei < -toler and alphas[i] < C) or (labelMat[i] * Ei > toler and alphas[i] > 0):

                # 选取第二个alpha2，预测类型、计算误差（选取规则见《机器学习》P125、《统计学习方法》P128）
                j = selectJrand(i, m)                                 # [i, m]范围内随机选取
                k_j = dataMatrix * dataMatrix[j, :].T
                fXj = float(multiply(alphas, labelMat).T * k_j) + b
                Ej = fXj - float(labelMat[j])

                # 求alpha最优修改量（公式见《统计学习方法》P126）
                alphaIold = alphas[i].copy()                    # 保留旧alpha值
                alphaJold = alphas[j].copy()
                if labelMat[i] != labelMat[j]:
                    L, H = max(0, alphas[j] - alphas[i]), min(C, C + alphas[j] - alphas[i])
                else:
                    L, H = max(0, alphas[j] + alphas[i] - C), min(C, alphas[j] + alphas[i])
                if L == H:                                      # L、H表示第二个alpha值的范围
                    print("L == H")
                    continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                      dataMatrix[i, :] * dataMatrix[i, :].T - \
                      dataMatrix[j, :] * dataMatrix[j, :].T     # alpha最优修改量（公式见《统计学习方法》P127）
                if eta >= 0:
                    print("eta >= 0")
                    continue

                # 优化两个alpha的值：alphas[i]与alphas[j]朝相反方向更新
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta      # 修改第二个alpha值，发生轻微改变则跳过循环
                alphas[j] = clipAlpha(alphas[j], H, L)
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print("j not moving enough")
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])

                # 每次完成两个alpha的优化后，都重新计算阈值b
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T

                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                alphaPairsChanged += 1
                print("iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged))
        iter = iter + 1 if alphaPairsChanged == 0 else 0
        print("iteration number: %d" % iter)
    return b, alphas


def kernelTrans(X, A, kTup):    # calc the kernel or transform data to a higher dimensional space
    """ 核转换函数（把原公式中的内积替换成核函数） """

    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':        # 线性核函数
        K = X * A.T
    elif kTup[0] == 'rbf':      # 径向基核函数（高斯版本）
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T        # kTup[1]：确定到达率的参数
        K = exp(K / (-1 * kTup[1] ** 2))        # 对应元素相除（在Matlab中是求逆）
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))   # 第一列为有效标记位
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def calcEk(oS, k):
    """ 计算实际与预测误差值 """
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    return fXk - float(oS.labelMat[k])


def selectJ(i, oS, Ei):
    """
    选取第二个alpha值（保证每次优化采用最大步长，即误差值之差最大化）
    :param i:       第一个alpha值的索引
    :param oS:      数据集结构
    :param Ei:      第一个alpha值对应实际与预测的误差值
    :return: 
    """
    maxK = -1                           # 第二个alpha值的索引
    maxDeltaE = 0                       # 最大的误差值之差
    Ej = 0                              # 索引j对应的误差值
    oS.eCache[i] = [1, Ei]                              # 把索引j对应误差值放入缓存，标记为
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]     # 有效的误差值对应的索引值
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i:                  # 索引相同则跳过，不计算误差值
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:      # 误差值之差较大：更新三个参数
                maxK, maxDeltaE, Ej = k, deltaE, Ek
        return maxK, Ej
    else:                               # 缓存中没有有效的误差值，则随机选取一个并计算误差值（见第76行）
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):
    """ 计算误差值并加入缓存（当所有alpha被更新时） """
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i, oS):
    """ Platt版alpha值优化 """
    Ei = calcEk(oS, i)

    # 超出容错率，优化
    if (oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C) or (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0):

        # 选取第二个alpha2，预测类型、计算误差
        j, Ej = selectJ(i, oS, Ei)

        # 求alpha最优修改量（公式见《统计学习方法》P126）
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]    # changed for kernel
        if eta >= 0:
            print("eta>=0")
            return 0

        # 优化两个alpha的值：alphas[i]与alphas[j]朝相反方向更新
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)                                     # aplha被修改：更新误差缓存
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)

        # 每次完成两个alpha的优化后，都重新计算阈值b
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i]-alphaIold)*oS.K[i, i] - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i]-alphaIold)*oS.K[i, j] - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    """ Platt版SMO算法 """

    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    iter = 0                    # 迭代次数
    flag = True
    alphaPairsChanged = 0       # 优化次数
    while iter < maxIter and (alphaPairsChanged > 0 or flag):      # 迭代次数达到最大值或整个集合都未对alpha优化时退出
        alphaPairsChanged = 0
        if flag:                    # 遍历所有样本
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:                       # go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if flag:
            flag = False
        elif alphaPairsChanged == 0:
            flag = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas


def calcWs(alphas, dataArr, classLabels):
    """ 计算超平面权值（公式见《统计学习方法》P130） """

    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


def SVM():
    """ 支持向量机分类测试 """
    dataArr, labelArr = loadDataSet()
    b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
    for i in range(100):
        if alphas[i] > 0.0:
            print(dataArr[i], labelArr[i])

    dataMat = mat(dataArr)
    ws = calcWs(alphas, dataArr, labelArr)
    res = dataMat[0] * mat(ws) + b


def testRbf(k1=1.3):
    """ 径向基测试函数 """

    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]                 # 获取支持向量
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % shape(sVs)[0])
    m, n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))

    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat=mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the source error rate is: %f" % (float(errorCount) / m))


def img2vector(filename):
    """ 把图像文件转化为向量 """

    returnVect = zeros((1, 1024))
    lines = open(filename).readlines()
    for i in range(32):                         # 读取文件前32行
        for j in range(32):                     # 把每行头31个字符存储在数组
            returnVect[0, 32 * i + j] = int(lines[i][j])

    return returnVect


def loadImages(dirName):
    """ 加载图片到训练集 """

    hwLabels = []
    trainingFileList = os.listdir(dirName)                              # 读取训练集图片
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNumStr = int(fileNameStr.split('.')[0].split('_')[0])      # 从文件名中获取数字实际值
        if classNumStr == 9:                                            # 二分类测试：只取1和9
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels    


# 测试：手写字体识别（SVM）
def testDigits(kTup=('rbf', 10)):

    dataArr, labelArr = loadImages('trainingDigits')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat=mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A > 0)[0]
    sVs=datMat[svInd] 
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % shape(sVs)[0])
    m, n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr, labelArr = loadImages('testDigits')
    errorCount = 0
    datMat=mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]):
            errorCount += 1
    print("the source error rate is: %f" % (float(errorCount)/m))

if __name__ == "__main__":

    # simple smo test
    pass
