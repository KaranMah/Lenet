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

        # self.inputs = inputs
        # N, C, W, H = inputs.shape
        # new_width = (W - self.pool) / self.s + 1
        # new_height = (H - self.pool) / self.s + 1
        # out = np.zeros((N, C, new_width, new_height))
        # for n in range(N):
        #     for c in range(C):
        #         for w in range(W / self.s):
        #             for h in range(H / self.s):
        #                 out[n, c, w, h] = np.max(
        #                     self.inputs[n, c, w * self.s:w * self.s + self.pool, h * self.s:h * self.s + self.pool])

        return self.out



    def backward(self, dy, lr):
        self.dx = np.zeros(self.inputs.shape)
        fmap = np.repeat(np.repeat(self.out, self.s, axis=2), self.s, axis=3)
        dmap = np.repeat(np.repeat(dy, self.s, axis=2), self.s, axis=3)

        self.dx = (fmap == self.inputs) * dmap

        return self.dx
    def extract(self):
        return
