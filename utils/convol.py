import numpy as np
from utils.util import *

class Convol2d:

    def __init__(self, layer_size, kernel_size, name):
        self.depth, self.height, self.width = layer_size
        self.weights = np.random.uniform(-0.1, 0.1, kernel_size)
        self.bias = np.random.uniform(-0.1, 0.1, kernel_size[0])
        self.bias = np.mat(self.bias).T
        self.name = name

    def forward(self, X):
        # self.cache = self.Weights, self.bias
        n_filters, d_filter, h_filter, w_filter = self.weights.shape
        n_x, d_x, h_x, w_x = X.shape
        h_out = (h_x - h_filter) + 1
        w_out = (w_x - w_filter) + 1

        X_col = im2col_indices(X, h_filter, w_filter)
        W_col = self.weights.reshape(n_filters, -1)
        temp = W_col @ X_col
        self.out = temp + self.bias
        self.out = np.array(self.out).reshape(n_x, n_filters, h_out, w_out)
        self.cache = (X, self.weights, self.bias, X_col)

        return self.out



    def backward(self, dy, lr):
        X, W, b, X_col = self.cache
        n_filter, d_filter, h_filter, w_filter = W.shape

        db = np.sum(dy, axis=(0, 2, 3))
        db = db.reshape(n_filter, -1)

        dy_reshaped = dy.transpose(1, 2, 3, 0).reshape(n_filter, -1)
        dW = dy_reshaped @ X_col.T
        dW = dW.reshape(W.shape)

        W_reshape = W.reshape(n_filter, -1)
        dX_col = W_reshape.T @ dy_reshaped
        dX = col2im_indices(dX_col, X.shape, h_filter, w_filter)
        self.weights -= lr * dW
        self.bias -= lr * db
        return dX

