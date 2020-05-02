import numpy as np

# loss
def cross_entropy(pred, labels):
    loss = -sum([labels[i]*np.log(pred[i]) for i in range(len(pred))])
    return loss