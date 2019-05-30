#coding: utf-8
# 使用朴素贝叶斯进行文本分类

import numpy as np
import pysnooper

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea',
                    'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him',
                    'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute',
                    'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless',
                    'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak',
                    'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 
                    'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1代表侮辱，0代表正常
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 取并集
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            # 出现则置为1，此处使用伯努利分布，只有0和1
            returnVec[vocabList.index(word)] = 1
        else:
            print(f'the word: {word} is not in the vocabulary')
    return returnVec

@pysnooper.snoop()
def trainNB0(trainMatrix, trainCategory):
    # $$p(c_{i}|w) = \frac{p(w|c_{i})p(c_{i})}{p(w)}$$
    numTrainDocs = len(trainMatrix)  # 多少条训练文档
    numWords = len(trainMatrix[0])  # 词库大小
    # 初始化概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # label为1占总数概率
    p0Num = np.zeros(numWords)
    p1Num = np.zeros(numWords)
    p0Denom = 0.0; p1Denom = 0.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num / p1Denom  # 每个单词在该类别下出现的概率
    p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusive


if __name__ == "__main__":
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    # print(myVocabList)
    # print(setOfWords2Vec(myVocabList, listOPosts[0]))
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    print(p0V, p1V, pAb)
