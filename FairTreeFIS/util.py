import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.special import expit
import copy
from scipy.special import expit


def eqop(data,label, prediction, protectedIndex, protectedValue):
    protected = [(x,l,p) for (x,l,p) in zip(data,label, prediction) 
        if (x[protectedIndex] == protectedValue and l==1)]   
    el = [(x,l,p) for (x,l,p) in zip(data,label, prediction) 
        if (x[protectedIndex] != protectedValue and l==1)]
    protected_negative = [(x,l,p) for (x,l,p) in zip(data,label, prediction) 
        if (x[protectedIndex] == protectedValue and l==0)] 
    el_negative = [(x,l,p) for (x,l,p) in zip(data,label, prediction) 
        if (x[protectedIndex] != protectedValue and l==0)]
    tp_protected = sum(1 for (x,l,p) in protected if l == p)
    
    tp_el = sum(1 for (x,l,p) in el if l == p)
    tn_protected = sum(1 for (x,l,p) in protected_negative if l == p)
    tn_el = sum(1 for (x,l,p) in el_negative if l == p)
   
   
    tpr_protected = tp_protected / len(protected) if len(protected) != 0 else 0
    tpr_el = tp_el / len(el) if len(el) != 0 else 0
    
    tnr_protected = tn_protected / len(protected_negative) if len(protected_negative)!= 0 else 0
    tnr_el = tn_el / len(el_negative) if len(el_negative)!= 0 else 0
    negative_rate = tnr_protected - tnr_el
    eqop = (tpr_el - tpr_protected)
    
    return (eqop)
# %%

def DP(data, labels, prediction,protectedIndex, protectedValue):
    #print("changed")
    protectedClass = [(x,l) for (x,l) in zip(data, prediction) 
        if x[protectedIndex] == protectedValue]   
    elseClass = [(x,l) for (x,l) in zip(data, prediction) 
        if x[protectedIndex] != protectedValue]
    p = sum(1 for (x,l) in protectedClass if l == 1)
    q = sum(1 for (x,l) in elseClass  if l == 1)
    protectedProb = sum(1 for (x,l) in protectedClass if l == 1) / len(protectedClass) if len(protectedClass) != 0 else 0
    elseProb = sum(1 for (x,l) in elseClass  if l == 1) / len(elseClass) if len(elseClass) != 0 else 0
    #print("protected class, non-protected class, protected positive, non-protected positive",len(protectedClass),len(elseClass),p,q)
    return (elseProb - protectedProb)
#%%
def gini(y):
    total_classes, count = np.unique(y, return_counts=True)
    probability = np.zeros(len(total_classes), dtype=float)
    n = len(y)
    for i in range(len(total_classes)):
        probability[i] = (count[i]/n)**2
    if n == 0:
        return 0.0
    gini = 1 - np.sum(probability)
    return gini

# %%
def ma(x, window):
        return np.convolve(x, np.ones(window), 'valid') / window

#%%
def draw_plot(x,y,dest,name):
    sns.set_context("talk")
    fig = plt.figure(figsize = (10, 5))
    plt.bar(x, y)
    plt.xlabel("Feature")
    plt.ylabel(name)
    plt.savefig(dest)
    plt.show()

#%%
def fairness(leftX,lefty,rightX,righty,protected_attribute,protected_val,fairness_metric, triangle):
    #print("probabilistic")
    valueLeft, countLeft = np.unique(lefty, return_counts=True)
    valueRight, countRight = np.unique(righty, return_counts=True)
    if len(countLeft) == 2:
        left0, left1 = countLeft[0]/len(lefty), countLeft[1]/len(lefty)
    if len(countRight) == 2:
        right0, right1 = countRight[0]/len(righty), countRight[1]/len(righty)
    if len(countLeft) == 1:
        left0 = countLeft[0]/len(lefty) if valueLeft[0] == 0 else 0
        left1 = countLeft[0]/len(lefty) if valueLeft[0] == 1 else 0
    if len(countRight) == 1:
        right0 = countRight[0]/len(righty) if valueRight[0] == 0 else 0
        right1 = countRight[0]/len(righty) if valueRight[0] == 1 else 0
    if len(countLeft) == 0 or len(countRight) == 0:
        return 0

    x = np.concatenate((leftX,rightX),axis=0)
    y = np.concatenate((lefty,righty),axis = 0)
    pred00 = np.concatenate((np.zeros(len(lefty)),np.zeros(len(righty))), axis = 0)
    pred01 = np.concatenate((np.zeros(len(lefty)),np.ones(len(righty))), axis = 0)
    pred10 = np.concatenate((np.ones(len(lefty)),np.zeros(len(righty))), axis = 0)
    pred11 = np.concatenate((np.ones(len(lefty)),np.ones(len(righty))), axis = 0)
    if fairness_metric == 1:
        #fairness_score00 = eqop(x,y,pred00,protected_attribute,protected_val)
        fairness_score01 = eqop(x,y,pred01,protected_attribute,protected_val)
        #print("left 0 prob, right 1 prob", left0, right1)
        fairness_score10 = eqop(x,y,pred10,protected_attribute,protected_val)
        #print("left 1 prob, right 0 prob", left1, right0)
        #fairness_score11 = eqop(x,y,pred11,protected_attribute,protected_val)
    else:
        #fairness_score00 = DP(x,y,pred00,protected_attribute,protected_val)
        fairness_score01 = DP(x,y,pred01,protected_attribute,protected_val)
        #print("left 0 prob, right 1 prob", left0, right1)
        fairness_score10 = DP(x,y,pred10,protected_attribute,protected_val)
        #print("left 1 prob, right 0 prob", left1, right0)
        #fairness_score11 = DP(x,y,pred11,protected_attribute,protected_val)
    
    #print(fairness_score00, fairness_score01, fairness_score10, fairness_score11)
    if triangle == True:
        fairness_score =  abs(fairness_score01)*right1 + abs(fairness_score10)*left1
    else:
        fairness_score =  (fairness_score01)*right1 + (fairness_score10)*left1
    return abs(fairness_score)


def fairness_regression(leftX,lefty,rightX,righty,protected_attribute,protected_val,previous,alpha = 1):
    #print("probabilistic")
    left_pred = np.mean(lefty)
    right_pred = np.mean(righty)
    
    left_protected = len([l for (x,l) in zip(leftX,lefty) if
        x[protected_attribute] == protected_val])
    left_el = len(lefty) - left_protected
    right_protected = len([l for (x,l) in zip(rightX,righty) if
        x[protected_attribute] == protected_val])
    right_el = len(lefty) - right_protected

    pred_protected = (left_protected*left_pred + right_protected * right_pred)/(left_protected + right_protected) if (left_protected + right_protected) != 0 else 0
    pred_el = (left_el*left_pred + right_el * right_pred)/(left_el + right_el)  if (left_el + right_el) != 0 else 0  
    bias = (abs(pred_protected - pred_el))    
    return (bias)

def fairness_multiclass(leftX,lefty,rightX,righty,protected_attribute,protected_val,alpha = 1):
    max = 0
    x = np.concatenate((leftX,rightX),axis=0)
    y = np.concatenate((lefty,righty),axis = 0)
    num_classes = np.unique(y)
    total_protected = len([l for (x,l) in zip(x, y) 
        if x[protected_attribute] == protected_val])
    total_el = len([l for (x,l) in zip(x, y) 
        if x[protected_attribute] != protected_val])
    for i in range(len(num_classes)):
        left_pk = len([l for (x,l) in zip(leftX, lefty) 
        if l == i and x[protected_attribute] == protected_val])
        right_pk = len([l for (x,l) in zip(leftX, lefty) 
        if l == i and x[protected_attribute] == protected_val])
        left_nk = len([l for (x,l) in zip(leftX, lefty) 
        if l == i and x[protected_attribute] != protected_val])
        right_nk = len([l for (x,l) in zip(leftX, lefty) 
        if l == i and x[protected_attribute] != protected_val])
        prob_leftk = ((left_pk) + (left_nk))/ len(leftX)
        prob_rightk = ((right_pk) + (right_nk))/ len(rightX)
        if total_protected == 0:
            bias = 1
        elif total_el == 0:
            bias = 1
        else:
            bias_left = (left_pk / total_protected - left_nk/ total_el)*prob_leftk
            bias_right = (right_pk / total_protected - right_nk/ total_el)*prob_rightk
            bias = abs(bias_left + bias_right)
        if bias > max:
            max = bias
    return 1 - max

# %%






