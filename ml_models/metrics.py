import numpy as np 

def mean_square_error(y_true, y_pred):
    return np.mean((y_pred - y_true)**2)


def accuracy(y_true, y_pred):
    return np.sum(y_true==y_pred) / len(y_true)


def r2_score(y_true, y_pred):
    TSS = np.sum( (y_true - y_true.mean())**2 )
    ESS = np.sum( (y_true - y_pred)**2 )
    return 1 - (ESS/TSS)