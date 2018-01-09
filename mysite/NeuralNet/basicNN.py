#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import math
import NeuralNet.utils as utils
# import utils

class NeuralNet:
    def __init__(self, name):
        self.name = name
        self.params = {}
        self.cumulate_gradients = {}

    def initiate(self, layout, config):
        self.layers = layout
        numLayer = len(self.layers)
        for i in range(1, numLayer+1):
            layer = self.layers[i]
            self.params[i] = {}
            self.cumulate_gradients[i] = {}
            #print layer['name']
            if layer['name'] == "inner-product":
                self.params[i]['w'] = self._initiate_uniform(layer['input-shape'][0],
                                                             layer['output-shape'][0],
                                                             [layer['output-shape'][0], layer['input-shape'][0]])
                self.params[i]['b'] = self._initiate_uniform(layer['input-shape'][0],
                                                             layer['output-shape'][0],
                                                             [layer['output-shape'][0], 1])
                # init gradient
                self.cumulate_gradients[i]['w'] = 0
                self.cumulate_gradients[i]['b'] = 0
            elif layer['name'] == "batch-normalization":
                self.params[i]['gamma'] = np.zeros([layer['input-shape'][0], 1]) + 0.5
                self.params[i]['beta'] = np.zeros([layer['input-shape'][0], 1]) + 0.5
                self.cumulate_gradients[i]['gamma'] = 0
                self.cumulate_gradients[i]['beta'] = 0

        if config.optimizer == "default-momemtum" or "Anneal-momemtum" or "Adam":
            self.momentums = self._init_momentum(0, self.params)# stores m_t = f(m_t-1, grad, momemtum)
        return self.params

    def _initiate_gaussian(self, shape, mean=0, stdvar=0.1):
        '''
        Compute the initial value of the parameters
        parameter:
            mean: the mean of the gaussian
            stdvar: the scale
            shape: list of tuple
        '''
        return np.random.normal(mean, stdvar, shape)

    def _initiate_uniform(self, h1, h2, shape):
        '''
        Compute the initial value of the parameters
        parrameter:
            h1: the scale (number of hidden units) of the current hiddent layer
            h2: the scale (number of hidden units) of the previous hiddent layer
            shape: list or tuple
        '''
        b = math.sqrt(6.0) / math.sqrt(h1 + h2)
        return np.random.uniform(-b, b, shape)

    def _init_momentum(self, initval, weights):
        tmp = {}
        for i in weights:
            tmp[i] = {}
            if len(weights[i]) == 0:
                continue
            for key in weights[i]:
                tmp[i][key] = np.zeros(weights[i][key].shape) + initval
        return tmp

    def runNet(self, dataX, dataY, config, iterNum, isTest=False):
        '''
        Define the nework structure.
        parameter:
            layout: a dictionary defines the layout of network
                    layout:
                           key: index of layer, start from 1 and the original data is 0 layer
                           value: dict {}
                                  key: "name", value: $NameOfLayers
                                  key: "input-shape", value: a list or tuple
                                  key: "output-shape", value: a list or tuple
            dataX: the X, shape [batch, dataporints] # for MNST, datapoints = 784 
            dataY: the Y, shape [batch, label] one hot matrix

        for the shape of data:
            currently, all of the data are 2D array
        '''
        dataY = dataY.T
        # forward
        numLayers = len(self.layers)
        outputs = self._initOutputs(dataX) # key = number, value a dict
        #print "forward"
        for i in range(1, numLayers+1):
            # layer is the layer name
            _input = outputs[i-1]
            param = self.params[i]
            #print "size of param:", len(param)
            layer = self.layers[i]
            #print "layer", i, layer['name']
            if layer['name'] == "inner-product":
                # do lineaer combination
                outputs[i] = self.forward_linear(_input, param)
            elif layer['name'] == "batch-normalization":
                sampleMean = config.sampleMean
                sampleVar = config.sampleVar
                outputs[i] = self.forward_batchNorm(_input, param, sampleMean, sampleVar, not isTest)
            elif layer['name'] == "activation-sigmoid":
                # do "activation-sigmoid":
                outputs[i] = self.forward_sigmoid(_input)
            elif layer['name'] == "activation-tanh":
                # do "activation-tanh":
                outputs[i] = self.forward_tanh(_input)
            elif layer['name'] == "activation-ReLU":
                # do "acitvation-ReLU":
                outputs[i] = self.forward_ReLU(_input)
            elif layer['name'] == "softmax":
                # do "softmax":
                idx_softmax = i
                outputs[i] = self.forward_softmax(_input)
            elif layer['name'] == "loss":
                # do "loss":
                outputs[i] = self.forward_cross_entropy(_input, dataY)
            #print "forward output shape", layer['name'],  outputs[i]['data'].shape

        # compute the gradient of the loss function
        loss = outputs[numLayers]['data'] / dataY.shape[1]
        predict = np.argmax(outputs[idx_softmax]['data'], axis=0)
        softList = outputs[idx_softmax]['data']
        accuracy = self._precision(predict, dataY)
        # backward propergation, compute the gradient of each parameters
        if isTest:
            return loss, predict, accuracy
        params_gradient = {}
        #print "backward"
        for i in range(numLayers, 0, -1):
            # for each iteration, do:
            # compute outputs[i]['diff']
            # compute gradient of each parameters of current layer
            layer = self.layers[i]
            output = outputs[i]
            _input = outputs[i-1]
            #print "layer", i, layer['name']
            #print "input output shape", output['data'].shape, _input['data'].shape
            param_grad = {}
            param = self.params[i]
            if layer['name'] == "inner-product":
                _input, param_grad = self.backward_linear(output, _input, param, param_grad)
                params_gradient[i] = param_grad
            elif layer['name'] == "batch-normalization":
                _input, param_grad = self.backward_batchNorm(output, _input, param, param_grad)
                params_gradient[i] = param_grad
            elif layer['name'] == "activation-sigmoid":
                _input = self.backward_sigmoid(output, _input)
            elif layer['name'] == "activation-ReLU":
                _input = self.backward_ReLU(output, _input)
            elif layer['name'] == "activation-tanh":
                _input = self.backward_tanh(output, _input)
            elif layer['name'] == "softmax":
                _input = self.backward_softmax(output, _input, dataY)
            elif layer['name'] == "loss":
                output['diff'] = 1.0
                _input = self.backward_cross_entropy(output, _input, dataY)
            outputs[i-1] = _input

        # SGD for updating each parameters.
        #lr = self._get_lr_annealing(iterNum, config)
        #self.learning_rate = lr
        self.params = self.Optimizer(config.learning_rate, self.params, params_gradient, config, iterNum)
        self.iterNum = iterNum
        return loss, predict, accuracy, self.params, params_gradient, config.learning_rate

    def pred(self, dataX, config):
        # forward
        numLayers = len(self.layers)
        outputs = self._initOutputs(dataX) # key = number, value a dict
        #print "forward"
        for i in range(1, numLayers+1):
            # layer is the layer name
            _input = outputs[i-1]
            param = self.params[i]
            #print "size of param:", len(param)
            layer = self.layers[i]
            #print "layer", i, layer['name']
            if layer['name'] == "inner-product":
                # do lineaer combination
                outputs[i] = self.forward_linear(_input, param)
            elif layer['name'] == "batch-normalization":
                sampleMean = config.sampleMean
                sampleVar = config.sampleVar
                outputs[i] = self.forward_batchNorm(_input, param, sampleMean, sampleVar, not isTest)
            elif layer['name'] == "activation-sigmoid":
                # do "activation-sigmoid":
                outputs[i] = self.forward_sigmoid(_input)
            elif layer['name'] == "activation-tanh":
                # do "activation-tanh":
                outputs[i] = self.forward_tanh(_input)
            elif layer['name'] == "activation-ReLU":
                # do "acitvation-ReLU":
                outputs[i] = self.forward_ReLU(_input)
            elif layer['name'] == "softmax":
                # do "softmax":
                idx_softmax = i
                outputs[i] = self.forward_softmax(_input)
            #print "forward output shape", layer['name'],  outputs[i]['data'].shape

        # compute the gradient of the loss function
        predict = np.argmax(outputs[idx_softmax]['data'], axis=0)
        return predict

    def _precision(self, logic, label):
        tmp = np.argmax(label, axis=0)
        return np.mean(logic == tmp)

    def _initOutputs(self, dataX):
        tmp = {}
        tmp[0] = {}
        tmp[0]['data'] = dataX.T
        tmp[0]['diff'] = None
        return tmp

    def forward_batchNorm(self, _input, param, sampleMean, sampleVar, isTrain=True, epsilon=0.00001):
        X = _input['data']
        output = {}
        mean = np.sum(X, axis=1) / X.shape[1] if isTrain else sampleMean
        mean = mean.reshape(mean.shape[0], 1)
        var = np.sum(np.power(X - mean, 2), axis=1) / X.shape[1] if isTrain else sampleVar
        var = var.reshape(var.shape[0], 1)
        xhat = (X - mean) / np.sqrt(var + epsilon)
        x_norm = xhat * param['gamma'] + param['beta']
        output['data'] = x_norm
        output['diff'] = None
        _input['tmp'] = (xhat, 1.0 / np.sqrt(var + epsilon))
        return output

    def backward_batchNorm(self, _output, _input, param, param_gradient):
        diff = _output['diff'] # diff with shape [data, batch]
        Xout = _output['data'] # Xout with shape [data, batch]
        X = _input['data']
        xhat, ivarEps = _input['tmp']
        param_gradient['beta'] = np.sum(Xout, axis=1).reshape(Xout.shape[0], 1)
        param_gradient['gamma'] = np.sum(diff * Xout, axis=1).reshape(Xout.shape[0], 1)
        grad_xhat = diff * param['gamma']
        sum_grad_xhat = np.sum(grad_xhat, axis=1).reshape(grad_xhat.shape[0], 1)
        sum_gxhat_xhat = np.sum(grad_xhat * xhat, axis=1).reshape(grad_xhat.shape[0], 1)
        _input['diff'] = (1.0 / X.shape[1]) * ivarEps * (X.shape[1] * grad_xhat -
                                                         sum_grad_xhat -
                                                         xhat * sum_gxhat_xhat)
        return _input, param_gradient


    def forward_linear(self, _input, _param):
        '''
        Compute the linear combination of X based on W and b
        parameters:
            input[data] : X  input data with shape, [data, batch]
            param['w'] : W  weight matrix with shape, [output_size, input_size]
            param['b'] : b  bias, with shape [data, batch]
        '''
        output = {}
        X = _input['data']
        W = _param['w']
        b = _param['b']
        #print "w", W.shape, "b", b.shape, "X", X.shape
        output['data'] = np.dot(W, X) + b
        output['diff'] = None
        #print "linear", output['data']
        return output

    def backward_linear(self, _output, _input, param, param_gradient):
        '''
        Compute the backpropergation of gradient at the linear
        combintaion operation.
        parameters:
            X:  input data with shape, [batch, data]
            W:  weight matrix with shape, [intput_size, output_size]
            b:  bias, with shape [batch, output_size]
        return a list of gradient [W, X, b]
        '''
        diff = _output['diff'] # shape [batch, datapoints]
        X = _input['data']
        W = param['w']
        b = param['b']
        _input['diff'] = np.dot(W.T, diff)
        #print "linear", X.shape, diff.shape
        ones = np.ones((X.shape[1], 1))
        param_gradient['w'] = np.dot(diff, X.T)
        param_gradient['b'] = np.dot(diff, ones)
        return _input, param_gradient

    def forward_sigmoid(self, _input):
        '''
        Compute the forward sigmoid function of the input X
        parameter:
            X: the input variable, with shape [batch, data]
        '''
        # use basic np build-in vectorized opertion
        X = _input['data']
        if np.sum(np.abs(X) > 20) > 0:
            X[X > 20] = 20
            X[X < -20] = -20 
        rst = 1.0 / (1 + np.exp(-1 * X))
        # or use np.vectorize() to build a vectorized function
        # but the building process require some time
        output = {}
        output['data'] = rst
        output['diff'] = None
        #print "sigmoid", output['data']
        return output

    def backward_sigmoid(self, _output, _input):
        '''
        Compute the gradient of a sigmoid funciton
        parameter:
            X: is the input data
        '''
        diff = _output['diff']
        X = _input['data']
        rst = _output['data']
        _input['diff'] = np.multiply(diff, rst * (1 - rst))
        return _input

    def forward_ReLU(self, _input):
        # data >= 0 -> output, otherwise, set zero
        X = _input['data']
        output = {}
        X[X < 0] = 0
        output['data'] = X
        output['diff'] = None
        return output

    def backward_ReLU(self, _output, _input):
        diff = _output['diff']
        tmp = np.zeros(_input['data'].shape)
        tmp[_input['data'] >= 0] = 1
        _input['diff'] = np.multiply(diff, tmp)
        return _input

    def forward_tanh(self, _input):
        X = _input['data']
        output = {}
        exp2 = np.exp(2 * X)
        output['data'] = (exp2 - 1) / (exp2 + 1)
        output['diff'] = None
        return output

    def backward_tanh(self, _output, _input):
        diff = _output['diff']
        _input['diff'] = np.multiply(diff, 1 - np.power(_output['data'], 2))
        return _input

    def forward_softmax(self, _input):
        '''
        Compute the softmax of X
        parameter:
            X: is the input, with shape [batch, data]
        '''
        X = _input['data']
        if np.sum(np.abs(X) > 20) > 0:
            X[X > 20] = 20
            X[X < -20] = -20 
        tmp = np.exp(X)
        sumval = np.sum(tmp, axis=0)
        sumval = sumval.reshape(sumval.shape[0], 1)
        output = {}
        output['data'] = (tmp.T / sumval).T
        #print output['data'].shape, np.sum(output['data'][:, 0])
        #print "softmax", output['data']
        output['diff'] = None
        return output

    def backward_softmax(self, _output, _input, y):
        '''
        Compute the gradient of softmax function
        parameter:
            X: is the data, with shape [batch, data]
        '''
        diff = _output['diff'] # shape [batch, datapoints]
        X = _input['data']
        #print X
        tmp = np.exp(X)
        sumval = np.sum(tmp, axis=0)
        sumval = sumval.reshape(sumval.shape[0], 1)
        #print sumval
        tmp = (tmp.T / sumval).T
        #print tmp.shape, "tmp3", diff == None
        _input['diff'] = tmp - y
        #print "input diff", _input['diff']
        return _input

    def forward_cross_entropy(sefl, _input, y):
        '''
        Compute the cross_entropy of X and y (between two distribution)
        parameter:
            X: the output softmax (a distribution), shape [batch, datapoints]
            y: one-hot vector
        '''
        X = _input['data']
        output = {}
        output['data'] = np.sum(np.multiply(-1.0 * y, np.log(X + 1e-5)))
        #print "cross-entropy", np.multiply(-1.0 * y, np.log(X)), y
        output['diff'] = None
        return output

    def backward_cross_entropy(self, _output, _input, y):
        '''
        Compute the cross_entropy of X and y (between two distribution)
        parameter:
            X: the output softmax (a distribution), shape [batch, datapoints]
            y: one-hot vector
        '''
        diff = _output['diff']
        X = _input['data']
        _input['diff'] = np.multiply(-1.0 * y, 1 / (X+ 1e-5)) * diff
        #print _input['diff']
        return _input

    def _get_lr_annealing(self, iterNum, config):
        '''
        get_lr() computes the new learning rate given the number of iteration and initial
        learning rate
        '''
        return config.learning_rate / math.pow((1 + config.lr_gamma * iterNum), config.lr_power)

    def _regularization(self, config, i, varId):
        if varId == 'b':
            return np.zeros(self.params[i][varId].shape) 
        if config.regularization == "L2":
            return self.params[i][varId] * config.decay
        elif config.regularization == "L1":
            # Wij = 0 -> gradient = 1
            return utils.L1Gradient(self.params[i][varId]) * config.decay
        else:
            return np.zeros(self.params[i][varId].shape)

    def _get_delta_advanced(self, config, iterNum, curGradient, i, varId):
        # compute the delta of gradient with differnt algo.
        # return updated learning rate, gradient and delta value
        curGradient /= config.batch_size # normalized
        regularTerm = self._regularization(config, i, varId)
        if config.optimizer == "default":
            tmp = regularTerm + curGradient
            return config.learning_rate, tmp, (-1 * config.learning_rate * tmp)
        elif config.optimizer == "default-momentum":
            self.momentums[i][varId] = self.momentums[i][varId] * config.momentum + curGradient
            tmp = self.momentums[i][varId] + regularTerm
            return config.learning_rate, self.momentums[i][varId], (-1 * config.learning_rate * self.momentums[i][varId])
        elif config.optimizer == "Anneal":
            lr = self._get_lr_annealing(iterNum, config)
            return lr, curGradient, (-1 * lr * curGradient) 
        elif config.optimizer == "Anneal-momentum":
            lr = self._get_lr_annealing(iterNum, config)
            self.momentums[i][varId] = self.momentums[i][varId] * config.momentum + curGradient
            delta = -1 * self.momentums[i][varId] * lr
            return lr, self.momentums[i][varId], delta
        elif config.optimizer == "Adagrad":
            lr, self.cumulate_gradients[i][varId] = utils.Adagrad(config.learning_rate,
                                                                  curGradient, 
                                                                  self.cumulate_gradients[i][varId])
            return lr, curGradient, (-1 * lr * curGradient)
        elif config.optimizer == "RMSProp":
            lr, self.cumulate_gradients[i][varId] = utils.RMSProp(0.5,
                                                                  config.learning_rate,
                                                                  curGradient,
                                                                  self.cumulate_gradients[i][varId])
            return lr, curGradient, (-1 * lr * curGradient)
        elif config.optimizer == "Adam":
            self.momentums[i][varId] = self.momentums[i][varId] * config.momentum + curGradient
            regu_momemtum = self.momentums[i][varId]
            lr, self.cumulate_gradients[i][varId] = utils.Adam(0.5,
                                                               config.learning_rate,
                                                               curGradient,
                                                               self.cumulate_gradients[i][varId])
            regu_cumuGrad = self.cumulate_gradients[i][varId]
            return lr, regu_momemtum, (-1 * lr * regu_momemtum)


    def Optimizer(self, lr, weights, gradients, config, iterNum):
        '''
        Compute the Stochastic gradient descent
        parameter:
            lr: learning rate, a constant
            weights: dictionary of weight matrix
            config: configures of this model
        return:
            updated weigth dictionary
        this version of SGD is in mini-batch style
        '''
        #print "learning rate", lr
        for i in weights:
            if len(weights[i]) == 0:
                continue
            for key in weights[i]:
                _, _, delta_key = self._get_delta_advanced(config,
                                                         iterNum, 
                                                         gradients[i][key], 
                                                         i,
                                                         key)
            weights[i][key] = weights[i][key] + delta_key 
            
        return weights

        