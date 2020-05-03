import numpy as np
def cross_entropy(pred, Y):
    temp =  np.argmax(pred, axis=1)
    ctr = 0
    for i in range(len(temp)):
        if temp[i] == Y[i]:
            ctr +=1
    Y = np.eye(10)[Y]
    assert Y.shape == pred.shape

    epsilon = 1e-10
    loss = (-Y * np.log(pred + epsilon)).sum() / pred.shape[0]
    print (loss, ctr)
    return loss