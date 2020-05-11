from utils.util import *

class conv:
    def __init__(self, cin, cout, filter_size, name):
        self.w = np.random.uniform(-0.1, 0.1, (cout, cin, filter_size, filter_size))
        self.b = np.random.uniform(-0.1, 0.1, cout)
        self.name = name

    def forward(self, x):
        wN, C, wH, wW = self.w.shape
        N, C, H, W = x.shape
        new_h = 1 + (H - wH)
        new_w = 1 + (W - wW)

        self.col = im2col(x, wH, wW)
        self.col_W = self.w.reshape(wN, -1).T

        y = np.dot(self.col, self.col_W) + self.b
        y = y.reshape(N, new_h, new_w, -1).transpose(0, 3, 1, 2)

        self.x = x

        return y

    def backward(self, dy, lr):
        wN, C, wH, wW = self.w.shape
        dy = dy.transpose(0, 2, 3, 1).reshape(-1, wN)

        self.db = np.sum(dy, axis=0)
        self.dw = np.dot(self.col.T, dy)
        self.dw= self.dw.transpose(1, 0).reshape(wN, C, wH, wW)

        dcol = np.dot(dy, self.col_W.T)
        dx = col2im(dcol, self.x.shape, wH, wW)
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
        new_h = int(1 + (H - self.pool_h) / self.stride)
        new_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        self.arg_max = np.argmax(col, axis=1)
        y = np.max(col, axis=1)
        y = y.reshape(N, new_h, new_w, C).transpose(0, 3, 1, 2)

        self.x = x
        return y

    def backward(self, dy, lr):
        dy = dy.transpose(0, 2, 3, 1)
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dy.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dy.flatten()
        dmax = dmax.reshape(dy.shape + (pool_size,))
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride)
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
        temp = np.exp(X)
        self.Y = temp/np.sum(temp, axis=1, keepdims=True)
        return self.Y

    def backward(self, output, lr):
        self.dx =  (self.Y - output) / self.Y.shape[0]
        return self.dx
