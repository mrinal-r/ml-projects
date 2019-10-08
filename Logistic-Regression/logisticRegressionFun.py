#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:35:30 2019

@author: mrinalrawool
"""

import math
from random import randrange

#####################################################################################################
# general sigmoid function for calculating predicted outcome:
#####################################################################################################
def sigmoid(x):
    if (x<0):
        return 1 - 1/(1 + math.exp(x))
    else:
        return 1/(1 + math.exp(-x))

#####################################################################################################
# functions for generating partial derivatives:
#####################################################################################################
def derror_by_dw0(W, Xj, X0j, Yj):
    result = X0j*(Yj - (math.exp(W[0]*X0j+W[1]*Xj)/(1 + math.exp(W[0]*X0j+W[1]*Xj))))
    return result

def derror_by_dw1(W, Xj, X0j, Yj):
    result = Xj*(Yj - (math.exp(W[0]*X0j+W[1]*Xj)/(1 + math.exp(W[0]*X0j+W[1]*Xj))))
    return result

#####################################################################################################
# functions for generating confusion Matrix:
#####################################################################################################
def confusion_mtrx_scores(y_pred_train, y_train):
    obs = len(y_train)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(0, obs, 1):
        if ((y_train[i] == y_pred_train[i]) and (y_train[i]==1)):
            tp +=1
        if ((y_train[i] == y_pred_train[i]) and (y_train[i]==0)):
            tn +=1
        if ((y_train[i] != y_pred_train[i]) and (y_train[i]==1)):
            fn +=1    
        if ((y_train[i] != y_pred_train[i]) and (y_train[i]==0)):
            fp +=1 
    print("Training analysis:")
    print("True Positives:{0}".format(tp))
    print("True Negatives:{0}".format(tn))
    print("False Positives:{0}".format(fp))
    print("False Negatives:{0}".format(fn))
    print("Accuracy:{0}".format(round((tp+tn)/obs, 2)))
    if ((tp+fp)!=0):
        print("Precision:{0}".format(round(tp/(tp+fp), 2)))
    else:
        print("Unable to calculate precision since true positives and false positives is zero")
    if ((tp+fn)!=0):
        print("Recall:{0}".format(round(tp/(tp+fn), 2)))
    else:
        print("Unable to calculate recall since true positives and false negative is zero")
            
#####################################################################################################
# main function
#####################################################################################################
def logistic_reg_train(epochs, n , w_init, x_train, x0, y_train, early_stopping, sgd, batchSize):
    y_preds = list(list())                              # variable to keep track of predicted outcomes
    error_tracking = list()                             # variable to keep track of error percentage changes

    len_train_dataset = len(x_train)
    prev_err_pct = 100                                  # for early stopping
    max_epoch = epochs
    num_wt_upd = 0
    w_optimum = w_init.copy()
    
    while(epochs >0):
        print("Epoch: {0}   Weights: {1}".format(max_epoch-epochs+1, w_init))
        partialError_w0 = 0
        partialError_w1 = 0
        
        # generate y_pred for training set
        y_pred_train = list()
        y_pred_raw = list()
        for j in range(0,len_train_dataset,1):
            d = w_init[0]*x0[j] + w_init[1]*x_train[j]  # determinant        
            val = sigmoid(d)
            y_pred_train.append(round(val))             # populate y_pred
            y_pred_raw.append(val)
        y_preds.append(y_pred_train)                    # tracking predicted outcomes 
        #print(y_pred_train)
        #print(y_pred_raw)
        
        # accuracy
        tally = 0
        for j in range(0,len_train_dataset,1):          
            if (y_train[j] != y_pred_train[j]):
                tally += 1
        error_pct = tally*100/len_train_dataset
        print("Error%: {0}".format(error_pct))
        error_tracking.append(error_pct)
        
        print("Previous epoch error: {0}, Current error: {1}".format(prev_err_pct, error_pct))
        
        if (early_stopping):
            if (prev_err_pct>=error_pct):
                go = True                               # flag that dictates weight updating condition
            else:
                go = False
        else:
            go = True
        
        if (go):
            
            if (sgd == 0):
                # generate partial error derivative
                for j in range(0,len_train_dataset,1):      # using gradient descent
                    val =   derror_by_dw0(w_init, x_train[j],x0[j], y_train[j])
                    partialError_w0 += val
                    
                for j in range(0,len_train_dataset,1):      # using gradient descent
                    val =   derror_by_dw1(w_init, x_train[j],x0[j], y_train[j])
                    partialError_w1 += val
                print("partialError_w0: {0}     partialError_w1: {0}".format(round(partialError_w0, 4), round(partialError_w1, 4)))
                
                # weight updates   
                print("Updating weights using batch GD...") 
            
            # SGD
            if (sgd == 1):
                batch = list()
                for i in range(0, batchSize):
                    batch.append(randrange(len_train_dataset))   # pick a random index from training set
                print("Sample indices {0}".format(batch))
                
                # generate partial error derivative
                partialError_w0 = 0
                for j in batch:                         # using stochastic gradient descent
                    val =   derror_by_dw0(w_init, x_train[j],x0[j], y_train[j])
                    partialError_w0 += val
                    
                partialError_w1 = 0
                for j in batch:                         # using stochastic gradient descent
                    val =   derror_by_dw1(w_init, x_train[j],x0[j], y_train[j])
                    partialError_w1 += val
                print("partialError_w0: {0}     partialError_w1: {0}".format(round(partialError_w0, 4), round(partialError_w1, 4)))
                
                # weight updates   
                print("Updating weights using SGD...") 
            
            num_wt_upd +=1   
            w_optimum = w_init.copy()                   # store prev weights in case the update makes the error rate go high                                        
            w_init[0] = w_init[0] - n*partialError_w0   # w0
            w_init[1] = w_init[1] - n*partialError_w1   # w1
    
            prev_err_pct=error_pct                      # update error for comparison
            epochs -= 1
            if (epochs == 0):
                print("Finalizing training weights after {0} weight updates".format(num_wt_upd))
                confusion_mtrx_scores(y_pred_train, y_train)
        else:
            # only when early stopping happens
            print("Found the optimal weights at error rate {0} after {1} weight updates. Early stopping...".format(prev_err_pct, num_wt_upd))
            # confusion matrix is to be generated on optimal weights, hence length -2
            confusion_mtrx_scores(y_preds[len(y_preds)-2], y_train)
            break;
    
    print("Plotting model training progress...")
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    time = np.arange(1,(len(error_tracking)+1),1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(1,(len(error_tracking)+1))
    ax.set_ylim(0, 100)
    ax.plot(time, error_tracking)
    ax.set_xlabel('X (Epochs)')
    ax.set_ylabel('Y (Error)')
    plt.show()
    
    return w_optimum

#####################################################################################################
# testing 
#####################################################################################################
def logistic_reg_test(w_optimum, x_test):
    d = w_optimum[0] + w_optimum[1]*x_test  # determinant        
    val = sigmoid(d)
    if (round(val)==1):
        print("{0}% chances of failing or {1}% chances of passing".format(round(val*100, 3), round((1-val)*100, 3)))
        print("Class: {0}".format(round(val)))
    else:
        print("{0}% chances of passing or {1}% chances of failing".format(round((1-val)*100, 3), round(val*100, 3)))
        print("Class: {0}".format(round(val)))