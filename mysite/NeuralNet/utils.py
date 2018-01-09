#!/usr/bin/env python

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt


#
# utils.py impelements some utility functions 
#

#==============================================
# adapting learning rates
#==============================================

def Adagrad(lr, curGradient, preValue, epsilon=0.0001):
	# Adagrad rescales the learning rate w.r.t the cumulative sum of 
	# gradients from step 1 to current step t
	# preValue is the cumulative sum of gradients/Adagrad-value of step t-1
	# curGradient is the gradient of current step
	# lr is the basic learning rate
	# epsilon is the value to ensure non-zero division
	preValue +=  np.sum(np.power(curGradient, 2))
	return lr / np.sqrt(preValue + epsilon), preValue

def RMSProp(beta, lr, curGradient, preValue, epsilon=0.0001):
	# RMSProp is the special case of AdagradDelta, with beta equal to 0.5
	preValue = beta * preValue + (1 - beta) * np.sum(np.power(curGradient, 2))
	return lr / np.sqrt(preValue + epsilon), preValue

def Adam(beta, lr, curGradient, preValue, epsilon=0.0001):
	# Adam trys to compute the scaling of new learning rate w.r.t
	# the momentum instead of the cumulative sum of gradients.
	preValue = beta * preValue + (1 - beta) * np.sum(np.power(curGradient, 2))
	return lr / np.sqrt(preValue + epsilon), preValue	



#
# L1 regularization gradient, Wij = 0 -> gradient 1
#

def L1Gradient(param):
	indices = param >= 0
	rst = np.zeros(param.shape)
	rst[indices] += 1
	return rst


def visualizeParam(model, outDir):
	weight = model.params[1]['w'] # a dictionary
	totalIMAGE = np.zeros((302, 302), dtype=np.int32)
	for i in range(weight.shape[0]):
		img_array = weight[i].reshape(28, 28)
		minval = np.min(img_array)
		img_array = img_array - minval
		maxval = np.max(img_array)
		img_array = img_array / maxval * 255
		rowNum = i // 10
		colNum = i % 10
		totalIMAGE[rowNum*30 + 2:(rowNum+1) * 30, colNum*30+2:(colNum+1) * 30] = img_array
		#im.save(outDir + "/PARAM" + str(i+1), "png")
	# im = plt.imshow(totalIMAGE)
	# im.savefig(outDir + "test.png")
	scipy.misc.imsave(outDir, totalIMAGE)

def plotLoss(x, ys, colors, tags, xmax, ymax, outDir):
	plt.axis([0, xmax, 0, ymax])
	for i, y in enumerate(ys):
		plt.plot(x, y, color=colors[i], linestyle="--", label=tags[i])
	plt.legend()
	# plt.xticks(range(xmax, 10), range(xmax, 10))
	plt.savefig(outDir)


if __name__ == "__main__":
	test = np.array([1,0,-1, 0]).reshape(2,2)
	print (test)
	print (L1Gradient(test))
	x = range(3)
	ys = [[1, 2, 3], [2,3, 4], [3, 4, 5]]
	colors = ["green", "orange", "red"]
	tags = ["a", "b", "c"]
	xmax = 6
	ymax = 6
	outDir = "./test.png"
	plotLoss(x, ys, colors, tags, xmax, ymax, outDir)