import numpy as np

class FullyConnected:

    def __init__(self, num_inputs, num_outputs, name):
        self.weights = 0.01*np.random.rand(num_inputs, num_outputs)
        self.bias = np.zeros((num_outputs, 1))
        self.name = name

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(np.squeeze(self.inputs), self.weights) + self.bias.T

    def backward(self, dy, lr):

        db = np.sum(dy.T, axis=1,keepdims=True)
        dx = np.dot(dy, self.weights.T)
        dw = np.dot(self.inputs.T, dy )

        print(self.bias.shape,db.shape)
        n_shape = self.weights.shape
        dw = dw.reshape(n_shape)
        print(self.weights.shape, dw.shape)
        self.weights -= lr * dw
        # print(dw)
        # print(db)
        self.bias -= lr * db
        # print(self.bias)

        return dx

    def extract(self):
        return {self.name+'.weights':self.weights, self.name+'.bias':self.bias}

    def feed(self, weights, bias):
        self.weights = weights
        self.bias = bias