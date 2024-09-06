import numpy as np
import matplotlib.pylab as plt
import random
from utils import *
from model_utils import Logistic_Regression

def main():
    # generate class0 and class1 values using a sin wave
    data_class0, data_class1 = data_generation(samples=60, 
                                               show_plots=False)

    # splitting the data into train and test
    train, test = train_test_split(data_class0=data_class0, 
                                   data_class1=data_class1, 
                                   train_percentage=80)

    #  initializing the model
    weight = np.round(random.uniform(0, 2), 2)   #Assigning random weight between 0 and 2 with round off to 2 decimal places
    bias = np.round(random.uniform(0, 2), 2)    #Assigning random bias between 0 and 2 with round off to 2 decimal places
    model = Logistic_Regression(weight=weight, bias=bias)   #Calling the model with the initialized weight and bias






if __name__ == '__main__':
    main()