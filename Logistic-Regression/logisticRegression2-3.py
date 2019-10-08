#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 20:44:51 2019

@author: mrinalrawool
"""
# imports

import imp
lr = imp.load_source('logisticRegressionFun.py', './logisticRegressionFun.py')


#####################################################################################################
# dataset ordered keeping 0s first
#####################################################################################################
x_train = [1,3,5,4,2,6,7,8]                             # number of weeks of inaction
x0 = [1,1,1,1,1,1,1,1]                                  # x0 is always 1
y_train = [0,0,0,1,1,1,1,1]                             # 0 = fail, 1 = pass

#####################################################################################################

#####################################################################################################
# model parameters:
#####################################################################################################
epochs = 200                                            # number of epochs
n = 0.01                                                # learning rate between 1 and 10^-6
w_init = [0.1,0.1]                                      # w0, w1 initial weights
batchSize = 0                                           # for SGD, use non zero
use_sgd = 0                                             # sgd flag 0 = no
early_stopping = True                                   # set to false when you want # epochs to be exhausted

#####################################################################################################
# train
#####################################################################################################
model_weights = lr.logistic_reg_train(epochs, 
                                      n , 
                                      w_init, 
                                      x_train, 
                                      x0, 
                                      y_train, 
                                      early_stopping, 
                                      use_sgd, 
                                      batchSize)

#####################################################################################################
# dataset ordered keeping 1s first
#####################################################################################################
x_train = [4,2,6,7,8,1,3,5]                             # number of weeks of inaction
x0 = [1,1,1,1,1,1,1,1]                                  # x0 is always 1
y_train = [1,1,1,1,1,0,0,0]                             # 0 = fail, 1 = pass

#####################################################################################################

#####################################################################################################
# model parameters:
#####################################################################################################
epochs = 200                                            # number of epochs
n = 0.01                                                # learning rate between 1 and 10^-6
w_init = [0.1,0.1]                                      # w0, w1 initial weights
batchSize = 0                                           # for SGD, use non zero
use_sgd = 0                                             # sgd flag 0 = no
early_stopping = True                                   # set to false when you want # epochs to be exhausted

#####################################################################################################
# train
#####################################################################################################
model_weights = lr.logistic_reg_train(epochs, 
                                      n , 
                                      w_init, 
                                      x_train, 
                                      x0, 
                                      y_train, 
                                      early_stopping, 
                                      use_sgd, 
                                      batchSize)






























