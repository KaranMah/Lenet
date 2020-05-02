import numpy as np

class Softmax:
    def __init__(self, name='Softmax'):
        self.name=name
    def forward(self, inputs):
        exp = np.exp(inputs, dtype=np.float)
        self.out = exp/np.sum(exp)
        return self.out
    def backward(self, dy, lr):
        return (self.out - dy) / self.out.shape[0]
    def extract(self):
        return