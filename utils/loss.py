import numpy as np
from sklearn.preprocessing import OneHotEncoder
# loss
def cross_entropy(pred, Y):
    # Y = Y.reshape(-1, 1)
    Y = np.eye(10)[Y]
    assert Y.shape == pred.shape

    epsilon = 1e-10
    loss = (-Y * np.log(pred + epsilon)).sum() / pred.shape[0]
    return loss