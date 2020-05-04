import numpy as np
def cross_entropy(probs,y_true):
    y_true = np.eye(10)[y_true]
    true_vals = np.squeeze(y_true)
    # print(y_true.shape, true_vals.shape)

    label = np.argmax(probs, axis=1)
    print(label)
    entropy = -np.sum(true_vals * np.log(probs))
    accuracy = np.sum(y_true == label) *100/ float(len(label))
    print(entropy, accuracy*100)
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