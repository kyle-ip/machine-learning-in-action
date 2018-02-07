import re

from numpy import *

import feedparser

"""
    朴素贝叶斯：假设每个特征同等重要（特征之间相互独立）
    实现方式：基于伯努利模型（只考虑是否出现）或基于多项式模型（考虑出现次数）
        p(ci|x, y)      后验概率：已知坐标(x, y)，其属于类ci的概率
        p(ci)           先验概率：类c1的概率
        p(ci|x, y) = p(x, y|ci)p(ci) / p(x, y)
"""


def loadDataSet():
    """ 创建示例数据集 """

    postingList=[
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],     # 1
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],              # 1
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']            # 1
    ]
    classVec = [0, 1, 0, 1, 0, 1]       # 1表示侮辱性文字，0表示正常言论
    return postingList, classVec


def createVocabList(dataSet):
    """ 词汇表去重排序 """
    return sorted(list({str(item) for line in dataSet for item in line}))


def setOfWords2Vec(vocabList, inputSet):
    """
    词袋模型
    :param vocabList:   去重排序后的单词表
    :param inputSet:    输入评论
    :return:            评论的单词在单词表中出现的位置及其出现的次数
    """

    returnVec = [0] * len(vocabList)        # 在单词表中标记0为该位置的单词没有在输入的评论中出现的次数

    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    """
    朴素贝叶斯分类算法
    :param trainMatrix:     训练集
    :param trainCategory:   训练集分类标签
    :return:
    """

    numTrainDocs = len(trainMatrix)                         # 输入的评论总行数
    numWords = len(trainMatrix[0])                          # 单词表长度（sum(trainCategory)表示测试数据矩阵中侮辱性评论的总行数）
    pAbusive = sum(trainCategory) / float(numTrainDocs)     # 先验概率：评论输入侮辱类的概率

    p0Num, p1Num = ones(numWords), ones(numWords)           # 两个列表分别记录单词表中侮辱性/非侮辱性单词出现个数
    p0Denom, p1Denom = 2.0, 2.0                             # 总单词数（Laplace Smoothing）：样本总数 + 类的个数
    for i in range(numTrainDocs):                           # 逐行评论判断
        if trainCategory[i] == 1:                           # 如当前行评论存在侮辱性文字，则单词表在中该行评论出现的单词 + 1
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])                  # 统计侮辱性评论单词总个数
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])                  # 统计非侮辱性评论单词总个数
    p1Vect, p0Vect = p1Num / p1Denom, p0Num / p0Denom       # 条件概率向量（即p(wi|c1)和p(wi|c0)，见《统计学习方法》P51）

    return log(p0Vect), log(p1Vect), pAbusive               # 返回取对数后的概率（避免下溢出）


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    朴素贝叶斯分类器
    :param vec2Classify:
    :param p0Vec:       条件概率向量
    :param p1Vec:
    :param pClass1:     先验概率
    :return:
    """
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    # 分为正例及反例的概率
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0


def testingNB():
    """ 侮辱性评论分类测试 """

    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = [setOfWords2Vec(myVocabList, postinDoc) for postinDoc in listOPosts]
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))

    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))         # 当前评论在单词表中出现的位置和次数
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


# 示例：使用朴素贝叶斯过滤垃圾邮件
def textParse(bigString):
    """ 切分文本为单词列表 """
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 


def spamTest(path):
    """ 垃圾邮件分类测试 """

    docList, classList = [], []
    for i in range(1, 26):                          # 导入并解析文本文件
        filename = os.path.join(path, "spam/{0}.txt".format(i))

        wordList = textParse(open(filename).read())
        docList.append(wordList)                    # 按行保存邮件的单词
        classList.append(1)

        filename = os.path.join(path, "ham/{0}.txt".format(i))
        wordList = textParse(open(filename).read())
        docList.append(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)            # 构造词汇表

    totalErrorCount = 0
    for i in range(100):                    # 交叉验证：在整体数据集中随机抽取部分为测试集，剩余为训练集，迭代100次取平均值
        trainingSet = list(range(50))

        testSet = []                                    # 构造测试集
        for i in range(10):
            randIndex = random.randint(0, len(trainingSet))
            testSet.append(trainingSet[randIndex])
            del(trainingSet[randIndex])

        trainMat, trainClasses = [], []                 # 构造训练集（邮件词汇及其出现在词汇表中的位置和次数）及其分类列表
        for docIndex in trainingSet:
            trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
            trainClasses.append(classList[docIndex])
        p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))

        errorCount = 0                                  # 对测试集分类
        for docIndex in testSet:                        # 逐行取测试邮件，转化为词袋并分类
            wordVector = setOfWords2Vec(vocabList, docList[docIndex])
            res = classifyNB(array(wordVector), p0V, p1V, pSpam)
            if res != classList[docIndex]:
                errorCount += 1
                print("classification error", docList[docIndex])

        totalErrorCount += float(errorCount)
    print('the error rate is: ', totalErrorCount / 1000)


# 示例：使用朴素贝叶斯分类器从个人广告中获取区域倾向

def calcMostFreq(vocabList, fullText):
    """ 选取出现频率最高的前30个元素 """

    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)

    return sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)[:30]


def localWords(feed1, feed0):
    """ RSS原分类器及高频词去除 """

    docList, classList, fullText = [], [], []
    minLen = min((len(feed1['entries']), len(feed0['entries'])))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)                # 创建词汇表
    top30Words = calcMostFreq(vocabList, fullText)      # 移除出现频次最高的前30个词
    print(top30Words)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])

    # 还需要去除停词：https://www.ranks.nl/stopwords
    trainingSet = list(range(2 * minLen))               # 构造训练集和测试集
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat, trainClasses = [], []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:            # 对测试集分类
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount)/len(testSet))
    return vocabList, p0V, p1V


def getTopWords(ny, sf):
    """ 展示最具表征性的词汇 """

    vocabList, p0V, p1V = localWords(ny, sf)
    topNY, topSF = [], []
    for i in range(len(p0V)):
        if p0V[i] > -6.0 :
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0 :
            topNY.append((vocabList[i], p1V[i]))

    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])

if __name__ == "__main__":
    listOPosts, listClasses = loadDataSet()
    myVocalbList = createVocabList(listOPosts)
    res = setOfWords2Vec(myVocalbList, listOPosts[0])

    trainMat = [setOfWords2Vec(myVocalbList, postinDoc) for postinDoc in listOPosts]

    # print(myVocalbList)
    # p0V, p1V, pAb = trainNB0(trainMat, listClasses)

    # 垃圾邮件过滤测试
    # spamTest("D:/machinelearninginaction/Ch04/email")

    ny = feedparser.parse("http://newyork.craigslist.org/stp/index.rss")    # NY 记为1类
    sf = feedparser.parse("http://sfbay.craigslist.org/stp/index.rss")      # SF 记为0类
    localWords(ny, sf)
    getTopWords(ny, sf)