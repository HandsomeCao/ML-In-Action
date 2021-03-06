# coding: utf-8
# 计算熵, 熵越高信息越多(类别多)
# ID3算法，采用熵来度量
# 在根据每个特征值划分数据集时，通过计算信息增益。最高的就是最好选择


import math
import operator
import pysnooper
import pickle

def calcShannoEnt(dataSet):
    """计算香农熵
    $H = -\sum{i=1}{n}p(x_{i})\log_{2}p(x_{i})
    """
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key, value in labelCounts.items():
        prob = float(value) / numEntries
        shannonEnt -= prob * math.log(prob, 2)
    return shannonEnt


def createDataset():
    """Create fake dataset"""
    dataSet = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    """
    split dataset through the value in axis==args(value)
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1 # 编号从0开始
    baseEntropy = calcShannoEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannoEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), 
        key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


@pysnooper.snoop()
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0] #类别完全相同则停止划分
    if len(dataSet[0]) == 1:
        return majorityCnt(classList) # 遍历完所有特征返回出现次数最多的
    bestFeat = chooseBestFeatureToSplit(dataSet) # 0
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(
            splitDataSet(dataSet, bestFeat, value), subLabels
        )
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    # print(featLabels, firstStr)
    featIndex= featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]) == dict:
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    with open(filename, 'w') as f:
        pickle.dump(inputTree, f)
    
def grabTree(filename):
    with open(filename) as f:
        Tree = pickle.load(f)
    return Tree


if __name__ == "__main__":
    myDat, labels = createDataset()
    copy_labels = labels[:]
    myTree = createTree(myDat, copy_labels)
    testVec1, testVec2 = [1, 0], [1, 1]
    print(classify(myTree, labels, testVec1))
    print(classify(myTree, labels, testVec2))
