import numpy as np
import matplotlib.pylab as plt
import random

def data_generation(samples, show_plots=False):
    '''
    Used to generate datapoints using a sin wave

    Parameters:
    samples (int): No of samples to generate for each class
    show_plots (bool): True if plots are to be displayed else False

    Returns:
    sin_values_class0 (np.array): Class 0 samples
    sin_values_class1 (np.array): Class 1 samples
    '''
    #generating class 0
    pi_range1_class0 = np.linspace(-np.pi, -1, (samples//2) + 1) # samples of pi values from -pi to -1
    pi_range2_class0 = np.linspace(1, np.pi, (samples//2) + 1) #samples of pi values from 1 to pi
    sin_values1_class0 = np.sin(pi_range1_class0)   # sin(x) values across -pi to -1
    sin_values2_class0 = np.sin(pi_range2_class0)    # sin(x) values across 1 to pi
    sin_values_class0 = np.concatenate((sin_values1_class0, sin_values2_class0), axis=0)

    if show_plots:
        plt.plot(pi_range1_class0, sin_values1_class0, 'o')
        plt.plot(pi_range2_class0, sin_values2_class0, 'o')
        plt.title('Class 0')
        plt.xlabel('Radians')
        plt.ylabel('Sin(x)')
        plt.show()

    #generating class 1
    pi_range_class1 = np.linspace(-1, 1, samples + 1) # samples of pi values from -1 to 1
    sin_values_class1 = np.sin(pi_range_class1)   # sin(x) values across -1 to 1

    if show_plots:
        plt.plot(pi_range_class1, sin_values_class1, 'o')
        plt.title('Class 1')
        plt.xlabel('Radians')
        plt.ylabel('Sin(x)')
        plt.show()


    return sin_values_class0, sin_values_class1

def train_test_split(data_class0, data_class1, train_percentage=80):
    '''
    Used to split the data into train and test respectively

    Parameters:
    data_class0 (np.array): Whole class 0 feature data in a (1,) shape
    data_class1 (np.array): Whole class 1 feature data in a (1,) shape
    train_percentage (int): Percentage of data to be assigned to the train set

    Returns:
    train_data (np.array): Train split with shape (train%, 1) (2nd dimension is labels) 
    test_data (np.array): Test split with shape (test%, 1) (2nd dimension is labels) 
    '''

    # combining the class label with data
    array_of_zeros_class0 = np.zeros(data_class0.shape[0]) # has a length of the number of samples in class 0
    array_of_ones_class1 = np.ones(data_class1.shape[0]) # has a length of the number of samples in class 0

    data_class0 = data_class0.reshape(-1, 1)   # reshape and add a dimension(column) to concatenate with the zeros array
    array_of_zeros_class0 = array_of_zeros_class0.reshape(-1, 1)    # reshape and add a dimension(column) to concatenate with the feature array

    data_class1 = data_class1.reshape(-1, 1)   # reshape and add a dimension(column) to concatenate with the zeros array
    array_of_ones_class1 = array_of_ones_class1.reshape(-1, 1)    # reshape and add a dimension(column) to concatenate with the feature array

    data_class0_with_labels = np.concatenate((data_class0, array_of_zeros_class0), axis=1)  #create an array to store feature along with its label
    data_class1_with_labels = np.concatenate((data_class1, array_of_ones_class1), axis=1)  #create an array to store feature along with its label

    combined_data = np.concatenate((data_class0_with_labels, data_class1_with_labels), axis=0)   #Combine the data together
    random.shuffle(combined_data)    #shuffling the data

    # splitting into train and test sets respectively
    total_samples = len(combined_data)  #total samples
    train_data = combined_data[: int(total_samples * (train_percentage/100))]  #% of training data is split
    test_data = combined_data[int(total_samples * (train_percentage/100)):]   #% of testing data is split

    return train_data, test_data