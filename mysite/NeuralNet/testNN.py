#!/usr/bin/env python

import numpy as np
import matplotlib as plt
import itertools as it
import NeuralNet.basicNN as nn
import pickle
import NeuralNet.utils as utils

import os
print("testNN:", os.getcwd())



class ConfigNet1:
    learning_rate = 0.1
    lr_gamma = 0.0001
    lr_power = 0.75
    w_decay = 0.0005
    batch_size = 128
    batch_norm = False
    sampleMean = 0.0
    sampleVar = 1.0
    momentum = 0.9
    outDir = "./tmp"
    loadPath = None
    isSave = True
    epoches = 180
    save_model = 10
    newInit = 5
    decay = 0.00001 # decay for regulerization
    regularization = "None" # available optios:  L2, L1, None
    optimizer = "default" # available options: default, default-momentum, Anneal, Anneal-momnentum, Adagrad, RMSProp and Adam

config = ConfigNet1()

def architectureSigmoid(batch_size):
    layers = {}
    layers[1] = {}
    layers[1]['name'] = "inner-product"
    layers[1]['input-shape'] = [784, batch_size]
    layers[1]['output-shape'] = [100, batch_size]

    layers[2] = {}
    layers[2]['name'] = "activation-sigmoid"
    layers[2]['input-shape'] = [100, batch_size]
    layers[2]['output-shape'] = [100, batch_size]

    layers[3] = {}
    layers[3]['name'] = "inner-product"
    layers[3]['input-shape'] = [100, batch_size]
    layers[3]['output-shape'] = [10, batch_size]

    layers[4] = {}
    layers[4]['name'] = "softmax"
    layers[4]['input-shape'] = [10, batch_size]
    layers[4]['output-shape'] = [10, batch_size]

    layers[5] = {}
    layers[5]['name'] = "loss"
    layers[5]['input-shape'] = [10, batch_size]
    layers[5]['output-shape'] = [10, batch_size]

    return layers

def architectureTanh(batch_size):
    layers = {}
    layers[1] = {}
    layers[1]['name'] = "inner-product"
    layers[1]['input-shape'] = [784, batch_size]
    layers[1]['output-shape'] = [100, batch_size]

    layers[2] = {}
    layers[2]['name'] = "activation-tanh"
    layers[2]['input-shape'] = [100, batch_size]
    layers[2]['output-shape'] = [100, batch_size]

    layers[3] = {}
    layers[3]['name'] = "inner-product"
    layers[3]['input-shape'] = [100, batch_size]
    layers[3]['output-shape'] = [10, batch_size]

    layers[4] = {}
    layers[4]['name'] = "softmax"
    layers[4]['input-shape'] = [10, batch_size]
    layers[4]['output-shape'] = [10, batch_size]

    layers[5] = {}
    layers[5]['name'] = "loss"
    layers[5]['input-shape'] = [10, batch_size]
    layers[5]['output-shape'] = [10, batch_size]

    return layers

def architectureReLU(batch_size):
    layers = {}
    layers[1] = {}
    layers[1]['name'] = "inner-product"
    layers[1]['input-shape'] = [784, batch_size]
    layers[1]['output-shape'] = [100, batch_size]

    layers[2] = {}
    layers[2]['name'] = "activation-ReLU"
    layers[2]['input-shape'] = [100, batch_size]
    layers[2]['output-shape'] = [100, batch_size]

    layers[3] = {}
    layers[3]['name'] = "inner-product"
    layers[3]['input-shape'] = [100, batch_size]
    layers[3]['output-shape'] = [10, batch_size]

    layers[4] = {}
    layers[4]['name'] = "softmax"
    layers[4]['input-shape'] = [10, batch_size]
    layers[4]['output-shape'] = [10, batch_size]

    layers[5] = {}
    layers[5]['name'] = "loss"
    layers[5]['input-shape'] = [10, batch_size]
    layers[5]['output-shape'] = [10, batch_size]

    return layers


def architectureMultiLayer(batch_size):
    layers = {}
    layers[1] = {}
    layers[1]['name'] = "inner-product"
    layers[1]['input-shape'] = [784, batch_size]
    layers[1]['output-shape'] = [100, batch_size]

    layers[2] = {}
    layers[2]['name'] = "activation-sigmoid"
    layers[2]['input-shape'] = [100, batch_size]
    layers[2]['output-shape'] = [100, batch_size]

    layers[3] = {}
    layers[3]['name'] = "inner-product"
    layers[3]['input-shape'] = [100, batch_size]
    layers[3]['output-shape'] = [100, batch_size]

    layers[4] = {}
    layers[4]['name'] = "activation-sigmoid"
    layers[4]['input-shape'] = [100, batch_size]
    layers[4]['output-shape'] = [100, batch_size]

    layers[5] = {}
    layers[5]['name'] = "inner-product"
    layers[5]['input-shape'] = [100, batch_size]
    layers[5]['output-shape'] = [10, batch_size]

    layers[6] = {}
    layers[6]['name'] = "softmax"
    layers[6]['input-shape'] = [10, batch_size]
    layers[6]['output-shape'] = [10, batch_size]

    layers[7] = {}
    layers[7]['name'] = "loss"
    layers[7]['input-shape'] = [10, batch_size]
    layers[7]['output-shape'] = [10, batch_size]

    return layers

def architectureBatacNorm(batch_size):
    layers = {}
    layers[1] = {}
    layers[1]['name'] = "batch-normalization"
    layers[1]['input-shape'] = [784, batch_size]
    layers[1]['output-shape'] = [784, batch_size]

    layers[2] = {}
    layers[2]['name'] = "inner-product"
    layers[2]['input-shape'] = [784, batch_size]
    layers[2]['output-shape'] = [100, batch_size]

    layers[3] = {}
    layers[3]['name'] = "activation-sigmoid"
    layers[3]['input-shape'] = [100, batch_size]
    layers[3]['output-shape'] = [100, batch_size]

    layers[4] = {}
    layers[4]['name'] = "inner-product"
    layers[4]['input-shape'] = [100, batch_size]
    layers[4]['output-shape'] = [100, batch_size]

    layers[5] = {}
    layers[5]['name'] = "activation-sigmoid"
    layers[5]['input-shape'] = [100, batch_size]
    layers[5]['output-shape'] = [100, batch_size]

    layers[6] = {}
    layers[6]['name'] = "inner-product"
    layers[6]['input-shape'] = [100, batch_size]
    layers[6]['output-shape'] = [10, batch_size]

    layers[7] = {}
    layers[7]['name'] = "softmax"
    layers[7]['input-shape'] = [10, batch_size]
    layers[7]['output-shape'] = [10, batch_size]

    layers[8] = {}
    layers[8]['name'] = "loss"
    layers[8]['input-shape'] = [10, batch_size]
    layers[8]['output-shape'] = [10, batch_size]
    return layers


def loadData(path, seed):
    with open(path, 'r') as file:
        lines = (line[:-1].split(",") for line in file)
        tmp_data = (i for i in lines)
        gen1, gen2 = it.tee(tmp_data)
        dataX = [np.array(i[:-1], dtype=np.float32) for i in gen1]
        dataY = [toOneHot(i[-1], 10) for i in gen2]
    dataX = np.array(dataX)
    dataY = np.array(dataY) 
    indices = np.arange(dataX.shape[0])
    np.random.seed(seed)
    np.random.shuffle(indices)
    dataX = dataX[indices, :]
    dataY = dataY[indices, :]
    return dataX, dataY

def toOneHot(digit, length):
    digit = int(digit)
    tmp = np.zeros(length, dtype=np.int8)
    tmp[digit] = 1
    return tmp

def computeSampleMeanVariance(data, config):
    config.sampleMean = np.sum(data.T, axis=1) / data.shape[0]
    config.sampleMean = config.sampleMean.reshape(config.sampleMean.shape[0], 1)
    config.sampleVar = np.sum(np.power(data.T - config.sampleMean, 2), axis=1) / data.shape[0]
    config.sampleVar = config.sampleVar.reshape(config.sampleVar.shape[0], 1)
    return config

def runModel(model, dataX, dataY, validX, validY, testX, testY, model_layers, config, iterNum):
    fmt = "epochNum={},trainCrossEntropy={},trainError={},validCrossEntropy={},validError={}"
    batches = list(zip(range(0, dataX.shape[0]+1, config.batch_size), range(config.batch_size, dataX.shape[0]+1, config.batch_size)))
    tlosses = []
    for i in range(config.epoches):
        tmpLoss = 0
        print(i)
        for start, end in batches:
            iterNum += 1
            #print iterNum
            tloss, tpredict, taccu, tparams, tparams_gradient, tlr = model.runNet(dataX[start:end, :], dataY[start:end, :], config, iterNum)
            tmpLoss += tloss
            if config.batch_norm:
                config = computeSampleMeanVariance(validX, config)
            vloss, vpredict, vaccu = model.runNet(validX, validY, config, iterNum, isTest=True)
            if config.batch_norm:
                config = computeSampleMeanVariance(testX, config)
            teloss, tepredict, teaccu = model.runNet(testX, testY, config, iterNum, isTest=True)
            # print (i+1, tloss, 1-taccu, vloss, 1-vaccu, teloss, 1-teaccu)
        utils.visualizeParam(model, "./static/param.png")
        tmpLoss /= len(batches)
        tlosses.append(tmpLoss)
        utils.plotLoss(range(1, len(tlosses)+1), tlosses, config.epoches+1, 2.5, "./static/plot.png")
    with open(config.outDir + "/" + model.name, 'wb') as file:
        pickle.dump(model, file, protocol=2)
    #print "model saved"

def main():
    model_layers = architectureSigmoid(config.batch_size)
    pathTrain = "/home/lui/CMU/Semester3/10707/hw1/data/digitstrain.txt"
    pathTest = "/home/lui/CMU/Semester3/10707/hw1/data/digitstest.txt"
    pathValid = "/home/lui/CMU/Semester3/10707/hw1/data/digitsvalid.txt"
    dataX, dataY = loadData(pathTrain, 1)
    testX, testY = loadData(pathTest, 1)
    validX, validY = loadData(pathValid, 1)

    model = nn.NeuralNet("demo_test_bestPerform_5_9_100-100_180_none")    
    # training process
    model.initiate(model_layers, config)
    iterNum = 0
    runModel(model, dataX, dataY, validX, validY, testX, testY, model_layers, config, iterNum)


def initImage():
    # draw initial image
    pass


if __name__ == "__main__":
# start a training process
    main()
