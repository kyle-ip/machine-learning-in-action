import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def getNumLeafs(myTree, numLeafs=0):
    """ 求决策树的叶节点数 """

    son = myTree[list(myTree.keys())[0]]        # 取当前树的第一个子树
    for _, v in son.items():                    # 当前节点类型为dict，则继续求其子树的叶节点，否则叶节数+1
        numLeafs += getNumLeafs(v) if isinstance(v, dict) else 1

    return numLeafs


def getTreeDepth(myTree, maxDepth=0):
    """ 求决策树的深度 """

    son = myTree[list(myTree.keys())[0]]
    for _, v in son.items():                    # 当前节点类型为dict，则深度+1，否则为1，每个子树计算深度后与最大深度比较
        curDepth = getTreeDepth(v) + 1 if isinstance(v, dict) else 1
        if curDepth > maxDepth:
            maxDepth = curDepth
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(
        nodeTxt, xy=parentPt,  xycoords='axes fraction',
        xytext=centerPt, textcoords='axes fraction',
        va="center", ha="center", bbox=nodeType, arrowprops=arrow_args
    )


def plotMidText(cntrPt, parentPt, txtString):
    """ 在父子节点之间添加文本信息 """
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):    # if the first key tells you what feat was split on
    """ 打印决策树 """

    numLeafs = getNumLeafs(myTree)          # this determines the x width of this tree
    firstStr = list(myTree.keys())[0]       # the text label for this node should be this

    cntrPt = (
        plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW,
        plotTree.yOff
    )

    plotMidText(cntrPt, parentPt, nodeTxt)              # 标记子节点属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]

    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD   # 减少y偏移

    for k, v in secondDict.items():
        if isinstance(v, dict):                 # 递归打印子树
            plotTree(v, cntrPt, str(k))
        else:                                   # 打印叶节点
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(v, (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(k))

    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

# if you do get a dictonary you know it's a tree, and the first element will be another dict


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    # no ticks
    # createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()

# def createPlot():
#    fig = plt.figure(1, facecolor='white')
#    fig.clf()
#    createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
#    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
#    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
#    plt.show()


# createPlot(thisTree)
