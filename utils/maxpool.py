import numpy as np

class Maxpooling:

    def __init__(self, pool_size, stride, name):
        self.pool = pool_size
        self.s = stride
        self.name = name

    def forward(self, inputs):
        self.inputs = inputs
        N, C, W, H = inputs.shape
        self.out = inputs.reshape(N, C, W//self.s, self.s, H//self.s, self.s).max(axis=(3,5))

        return self.out

    def backward(self, dy, lr):
        self.dx = np.zeros(self.inputs.shape)

        fmap = np.repeat(np.repeat(self.out, self.s, axis=2), self.s, axis=3)
        dmap = np.repeat(np.repeat(dy, self.s, axis=2), self.s, axis=3)

        self.dx = (fmap == self.inputs) * dmap

        return self.dx
    def extract(self):
        return
