#coding: utf-8

import numpy as np 
import operator
import matplotlib.pyplot as plt


def file2matrix(filename):
    with open(filename, 'r') as f:
        arrayOLines = f.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    for index, line in enumerate(arrayOLines):
        listFromLine = line.strip().split('\t')
        returnMat[index, :] = listFromLine[:3]
        classLabelVector.append(int(listFromLine[-1]))
    return returnMat, np.array(classLabelVector)


def show_scatter(datingDataMat, datingLabels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
        15.0 * datingLabels, 15.0 * datingLabels)
    plt.show()


def autoNorm(dataSet):
    """数据中有些数很大，有些数很小，需要归一化到相同范围"""
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(dataSet.shape)
    m = dataSet.shape[0]
    normDataSet = dataSet- np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


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


def datingClassTest():
    """测试分类器准确率"""
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('./datingTestSet2.txt')
    # print(datingDataMat)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :],
            normMat[numTestVecs:, :], datingLabels[numTestVecs:], 3)
        print(f'The classifier came back with {classifierResult},\
                the real answer is {datingLabels[i]}')
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print(f'The total error rate is: {errorCount / float(numTestVecs)}')


if __name__ == "__main__":
    datingClassTest()    
