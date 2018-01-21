from numpy import *


def classfy(dataSet, classList, learningRate=1):
    trainSetSize, dim = len(dataSet), len(dataSet[0])
    w, b = zeros((dim)), 0
    while 1:
        flag = True                                 # 修正标记：当遍历完所有样本都没有修改则完成训练
        for i in range(trainSetSize):               # 逐行数据带入判别式，判断预测分类与实际分类是否相符，直到每行都分类正确
            res = classList[i] * (dot(array(dataSet[i]), w) + b)  # dot：求向量内积
            if res <= 0:                            # 实际与预测结果异号即错判：梯度下降更新w、b（公式见《统计学习方法》P29）
                flag = False
                w += learningRate * classList[i] * array(dataSet[i])
                b += learningRate * classList[i]
                # print("w = {0}\tb = {1}".format(w, b))
                break
        if flag:
            break
    return lambda x: -1 if dot(w, array(x)) + b < 0 else 1  # 返回感知机模型函数：f(x) = sign(w * x + b)


if __name__ == "__main__":
    dataSet = [[3, 3], [4, 3], [1, 1]]
    classList = [1, 1, -1]
    f = classfy(dataSet, classList)