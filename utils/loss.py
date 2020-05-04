import numpy as np
def cross_entropy(probs,y_true):
    true_vals = np.squeeze(np.eye(10)[y_true])
    # label = np.argmax(probs, axis=1)
    entropy = -np.sum(true_vals * np.log(probs))
    res = np.argmax(probs, axis=1)
    accuracy = 1 - (np.count_nonzero((res - y_true)) / len(y_true))
    # print(accuracy)
    return (entropy)

    # # Y = np.eye(10)[y]
    # label = np.argmax(pred, axis=1)
    # epsilon = 1e-10
    # # loss = (-y * np.log(label + epsilon)).sum() / label.shape[0]
    #
    # label = np.argmax(pred, axis=1)
    # loss = -np.sum(y * np.log(label) + (1 - y) * np.log(1 - label)) / len(y)
    # accuracy = np.sum(y==label) / float(len(label))
    # print(loss,accuracy)
    # return loss