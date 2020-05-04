import numpy as np
from utils.util import *
class conv:
    def __init__(self, cin, cout, filtersize, name, strid=1, pading=0):
        self.strid = strid
        self.pading = pading
        self.par = dict()
        w = np.random.uniform(-0.1, 0.1, (cout, cin, filtersize, filtersize))
        self.par['w'] = w
        b = np.random.uniform(-0.1, 0.1, cout)
        self.par['b'] = b
        self.gra = dict()
        self.gra['w'] = None
        self.gra['b'] = None
        self.x = None
        self.col = None
        self.col_w = None
        self.name=name

    def forward(self, x):
        FN, C, FH, FW = self.par['w'].shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pading - FH) / self.strid)
        out_w = 1 + int((W + 2*self.pading - FW) / self.strid)

        col = im2col(x, FH, FW, self.strid, self.pading)
        col_W = self.par['w'].reshape(FN, -1).T

        out = np.dot(col, col_W) + self.par['b']
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout, lr):
        FN, C, FH, FW = self.par['w'].shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.gra['b'] = np.sum(dout, axis=0)
        self.gra['w'] = np.dot(self.col.T, dout)
        self.gra['w'] = self.gra['w'].transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        # print(dcol.shape)
        dx = col2im(dcol, self.x.shape, FH, FW, self.strid, self.pading)
        # print(self.name, "bias", self.gra['b'])
        self.par['b'] -= lr * self.gra['b']
        self.par['w'] -= lr * self.gra['w']

        return dx