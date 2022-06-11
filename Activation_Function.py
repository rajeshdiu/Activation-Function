from cmath import exp
import numpy as np


class Activation_Function:
    def identity(x):
        return x
    def binary_step(x):
        return np.heaviside(x,1)
    def tanh(x):
        return np.tanh(x)
    
    def sigmoid(x):
        return 1/ (1+np.exp(-x))

    def softmax(x):
        return np.exp(x)/np.sum(np.exp(x),axis=0)

    def ReLu(x):
        newListConvertNegativeToZero=[]

        for i in x:
            if i<0:
                newListConvertNegativeToZero.append(0)
            else:
                newListConvertNegativeToZero.append(i)
        return newListConvertNegativeToZero

    def LeakyReLu(x):

        newListConvertNegativeToZero=[]

        for i in x:
            if i<0:
                newListConvertNegativeToZero.append(0.01*i)
            else:
                newListConvertNegativeToZero.append(i)
        return newListConvertNegativeToZero

    def arcTan(x):
        return np.arctan(x)

        