import numpy as np
import matplotlib.pylab as plt
import random

class Logistic_Regression:
    '''
    Contains a logistic regression algorithm which can be trained

    Parameters:
    weight (float): Initial value for the weight
    bias (float): Initial value for the bias
    '''
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def sigmoid(self, x):
        # sigmoid implementation
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x):
        # input feature x is passed through sigmoid(wx+b) and returned
        x = (self.weight * x) + self.bias
        print(x)
        out = self.sigmoid(x)
        return out
    
    def binary_cross_entropy(y, y_pred):
        # binary cross entropy loss
        return -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))