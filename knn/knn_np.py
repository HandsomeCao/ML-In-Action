#coding: utf-8
# K近邻算法numpy实现


import numpy as np 
import operator

def createDataset():
    """Create fake dataset"""
    group = np.array([[1.0, 1.1],
                      [1.0, 1.0],
                      [0, 0],
                      [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # np.tile([0, 1], (3, 1)) = [[0, 1], [0, 1]...]
    # diffMat means the difference of this vector with the dataset matrix
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)  # 距离差的和
    distances = sqDistances ** 0.5
    sortedDisIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel= labels[sortedDisIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 # 如果出现这个标签则加1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


if __name__ == "__main__":
    group, labels = createDataset()
    print(classify0([0, 0], group, labels, 3))
