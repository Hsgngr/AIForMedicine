# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:59:55 2020

@author: Ege
"""
import numpy as np

p = np.array([[0.3,0.7,0.3],[0.7,0.9,0.7],[0.3,0.7,0.3]])
g = np.array([[0,1,0],[1,1,1],[0,1,0]])
p2 = np.array([[0.5,0.7,0.5],[0.7,0.9,0.7],[0.5,0.7,0.5]])

def softmax(p,g):
    numerator = sum(sum(2 * np.multiply(p,g)))
    denominator = sum(sum((p **2 + g **2)))
    
    loss = 1 - (numerator / denominator)
    return loss

softmax(p,g)
softmax(p2,g)
softmax(g,g)
