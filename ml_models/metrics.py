import numpy as np 

def mean_square_error(y_true, y_pred):
    return np.mean((y_pred - y_true)**2)


def accuracy(y_true, y_pred):
    return np.sum(y_true==y_pred) / len(y_true)
