# -*- coding:UTF-8 -*-
import numpy as np
import random



def sigmoid(inx):
    return .5 * (1 + np.tanh(.5 * inx))
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights.getA()



def Test():
    frTrain = open('训练集.csv')
    frTest = open('测试集.csv')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    trainWeights = gradAscent(np.array(trainingSet), trainingLabels)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights[:,0]))!= int(currLine[-1]):
            errorCount += 1
            print('出现错误')
            print([lineArr,"实际类型为:{}".format(int(currLine[-1])),"预测类型为:{}".format(int(classifyVector(np.array(lineArr), trainWeights[:,0])))])

    errorRate = (float(errorCount)/numTestVec) * 100

    return trainWeights,errorRate
def classifyPerson():
    trainWeights,errorRate=Test()
    print("测试集错误率为: %.2f%%" % errorRate)
    lineArrself =[float(input('请输入第一个数据：')),float(input('请输入第二个数据：')),float(input('请输入第三个数据：')),float(input('请输入第四个数据：')),float(input('请输入第五个数据:'))]

    print(int(classifyVector(np.array(lineArrself), trainWeights[:,0])))

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

if __name__ == '__main__':
    Test()
    classifyPerson()
