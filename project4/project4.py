#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 22:59:13 2019

@author: mrinalrawool
"""
import numpy as np
from numpy import random

#random.seed(0)  # 0, 100, 100

def squared(n):
    return round(n*n, 4)


def gen_input_sample():
    return round(random.uniform(-1.0, 1.0), 2)

####################### define the # datasets here ###########################################
N = 1000                                         

####################### code to generate the datasets ########################################
datasets = np.empty([N, 2, 2], dtype=float)
for i in range(N):
    for j in range(2):
        x = gen_input_sample()
        datasets[i][j][0] = x
        datasets[i][j][1] = squared(x)
            
#print(datasets[:2])            

####################### to store the final hypothesis per dataset. Total N entries ###########

gDx = np.empty([N, 2], dtype = float)
for i in range(N):
    m = 0
    c = -1
    for j in range(2):
        m += datasets[i][j][0]
        c *= datasets[i][j][0]
    gDx[i][0] =  m                # m or a = x1+x2
    gDx[i][1] =  c                # b or c = -x1x2

#print(gDx[:2])

# calculate the average final hypothesis that is independent of the datasets

gbarX = np.ones([2], dtype = float)
for i in range(N):
    gbarX[0] += gDx[i][0]
    gbarX[1] += gDx[i][1]
    
gbarX[0] /= N
gbarX[1] /= N

#print(gbarX)

# collect all x's used so far - this step is just for convenience of plotting

x_list = list() # x
for i in range(N):
    for j in range(2):
        x_list.append(datasets[i][j][0])
x_list.sort()

######################### dataset for f(x) #########################
fx_list = list() # f(x)
for i in x_list:
    fx_list.append(squared(i))

######################### dataset for gbar(x) #########################
gbarx_list = list() # gbar(x)
for i in x_list:
    y = round(gbarX[0]*i + gbarX[1], 5)
    gbarx_list.append(y)

#print('X')
#print(x_list[:10])
#print('f(x)')
#print(fx_list[:10])
#print('gbar(x)')
#print(gbarx_list[:10])

######################### calculate bias and variance #########################

num_data = len(x_list)

######################### BIAS #########################
# bias is easier since there is no dependence on the dataset
bias = 0
for i in range(num_data):
    bias += squared(gbarx_list[i] - fx_list[i])
    
bias /= num_data

######################### VARIANCE #########################
# gDx = ax + b , gbarx = Ax + B, gDx - gbarX = (a-A)x + (b-B), 
# I'm just going to do this for all x's and all gdx and take care of Ex and Ed together

varx = 0
var = 0
for i in range(num_data):
    vard = 0
    for j in range(N):
        # calculates for each x
        val = x_list[i]*(gDx[j][0] - gbarX[0]) + (gDx[j][1] - gbarX[1])
        vard += squared(val)
    # divide by n
    varx += vard/N
var = varx/num_data

print('Number of datasets considered : {0}'.format(N))
print('Bias : {0}'.format(bias))
print('Variance : {0}'.format(var))
print('Empirical Eout : {0}'.format(bias + var))
print('Expected Eout : {0}'.format(8/15))

######################### Plot for f(x) vs gbarX #########################

import matplotlib.pyplot as plt
#%matplotlib inline

fig = plt.figure(figsize = (4,3))
ax = fig.add_subplot(111)
ax.set_xlim(-1.5,1.5)
ax.set_ylim(-0.1, 1.1)

p1 = ax.plot(x_list, fx_list)    #f(x)

p2 = ax.plot(x_list, gbarx_list)    # gbar(x)

ax.set_xlabel('X (Dim)')
ax.set_ylabel('Y (Target)')
ax.set_title('Plot for f(x) vs gbar')
plt.show()



