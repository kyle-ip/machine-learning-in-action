from numpy import *


def classfy(dataSet, classList, learningRate=1, numIter=20):
    trainSetSize, dim = len(dataSet), len(dataSet[0])
    w, b, errorRate = zeros((dim)), 0, 1
    for _ in range(numIter):                        # 迭代训练数据
        flag = True                                 # 修正标记：当遍历完所有样本都无须修改则表示完成训练
        errorCount = 0.0
        for i in range(trainSetSize):               # 逐行数据带入判别式，判断预测分类与实际分类是否相符，直到每行都分类正确
            res = classList[i] * (dot(array(dataSet[i]), w) + b)  # dot：求向量内积
            if res <= 0:                            # 实际与预测结果异号即错判：梯度下降更新w、b（公式见《统计学习方法》P29）
                flag = False
                errorCount += 1.0
                w += learningRate * classList[i] * array(dataSet[i])
                b += learningRate * classList[i]
        errorRate = errorCount / trainSetSize
        if flag:
            break
    f = lambda x: -1 if dot(w, array(x)) + b < 0 else 1
    return f, errorRate     # 返回感知机模型函数及错误率：f(x) = sign(w * x + b)


if __name__ == "__main__":
    dataSet = [[3, 3], [4, 3], [3, 1], [0, 0], [2, 0], [0, 2], [3, 0], [2, 5]]
    classList = [1, 1, -1, 1, -1, 1, -1, 1]
    f, errorRate = classfy(dataSet, classList)
    print(f([0, -1]), errorRate)