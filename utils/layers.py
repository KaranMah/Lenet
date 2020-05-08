from utils.util import *

class conv:
    def __init__(self, cin, cout, filter_size, name):
        self.w = np.random.uniform(-0.1, 0.1, (cout, cin, filter_size, filter_size))
        self.b = np.random.uniform(-0.1, 0.1, cout)
        self.name = name

    def forward(self, x):
        FN, C, FH, FW = self.w.shape
        N, C, H, W = x.shape
        out_h = 1 + (H - FH)
        out_w = 1 + (W - FW)

        col = im2col(x, FH, FW)
        col_W = self.w.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout, lr):
        FN, C, FH, FW = self.w.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dw = np.dot(self.col.T, dout)
        self.dw= self.dw.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW)
        self.b -= lr * self.db
        self.w -= lr * self.dw

        return dx

class FullyConnected:
    def __init__(self, kernel_size, name):
        self.w = np.random.uniform(-0.1, 0.1, kernel_size)
        self.b = np.random.uniform(-0.1, 0.1, kernel_size[1])
        self.name = name

    def forward(self, X):
        self.x = X
        self.out = np.dot(np.squeeze(X), self.w) + self.b
        return self.out

    def backward(self, dy, lr):
        X = self.x
        self.dx = np.dot(dy, self.w.T)
        self.dx=self.dx.reshape(X.shape)
        self.dw = np.dot(X.T, dy)
        self.db = np.sum(dy, axis=0)
        self.w -= np.squeeze(lr * self.dw)
        self.b -= lr * self.db
        return self.dx


class Maxpooling:
    def __init__(self, size, stride=1, name="pool"):
        self.pool_h = size
        self.pool_w = size
        self.stride = stride
        self.name = name
    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout, lr):
        dout = dout.transpose(0, 2, 3, 1)
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size),
             self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h,
                    self.pool_w, self.stride)
        return dx

class ReLu:
    def __init__(self, name):
        self.name = name

    def forward(self, X):
        self.x = X
        self.out = (X > 0) * X
        return self.out

    def backward(self, dy, lr):
        self.dx = dy * (self.x > 0)
        return self.dx

class Softmax:
    def __init__(self, name):
        self.name = name

    def forward(self, X):
        dummy = np.exp(X)
        self.Y = dummy/np.sum(dummy, axis=1, keepdims=True)
        return self.Y

    def backward(self, output, lr):
        self.dx =  (self.Y - output) / self.Y.shape[0]
        return self.dx
