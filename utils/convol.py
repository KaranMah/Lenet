import numpy as np
from utils.util import *

class Convol2d:

    def __init__(self, layer_size, kernel_size, name):
        self.depth, self.height, self.width = layer_size
        self.weights = np.random.uniform(-0.01, 0.01, kernel_size)
        self.bias = np.random.uniform(-0.01, 0.01, kernel_size[0])
        self.bias = np.mat(self.bias)
        self.name = name
        self.weight_diff = 0
        self.bias_diff = 0
        self.momentum = 0.95

    # def forward(self, x):
    #
    #     k = self.weights.shape[2]
    #     n, d, h, w = x.shape
    #     x = x.reshape(n,h,w,d)
    #     self.x = x
    #     h_out = h - (k - 1)
    #     w_out = w - (k - 1)
    #
    #     w_n = self.weights.shape[0]
    #     weight = self.weights.reshape(-1, w_n)
    #     # print(n, h_out, w_out, w_n)
    #     output = np.zeros((n, h_out, w_out, w_n))
    #     for i in range(h_out):
    #         for j in range(w_out):
    #             inp = x[:, i:i+k, j:j+k, :].reshape(n, -1)
    #             out = inp.dot(weight) + self.bias
    #             output[:, i, j, :] = out.reshape(n, -1)
    #     # print(output.shape)
    #     output = output.reshape(n, w_n, h_out, w_out)
    #     return output
    #
    # def backward(self, diff,lr):
    #     n, c, h, w = diff.shape
    #     diff = diff.reshape(n, h, w, c)
    #     w_n = self.weights.shape[0]
    #     w_o = self.weights.shape[1]
    #     k = self.weights.shape[2]
    #     h_in = h + (k - 1)
    #     w_in = w + (k - 1)
    #
    #     weight_diff = np.zeros((k, k, w_o, w_n))
    #     for i in range(k):
    #         for j in range(k):
    #             #inp = (n, 28, 28, c) => (n*28*28, c) => (c, n*28*28)
    #             inp = self.x[:, i:i+h, j:j+w, :].reshape(-1, w_o).T
    #             #diff = n, 28, 28, 6 => (n*28*28, 6)
    #             diff_out = diff.reshape(-1, w_n)
    #             # print(inp.shape,diff_out.shape)
    #             weight_diff[i, j, :, :] = inp.dot(diff_out)
    #     bias_diff = np.sum(diff, axis=(0, 1, 2))
    #
    #     pad = k - 1
    #     diff_pad = np.pad(diff, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
    #     rotated_weight = self.weights[::-1, ::-1, :, :].transpose(0, 1, 3, 2).reshape(-1, w_o)
    #     back_diff = np.zeros((n, h_in, w_in, w_o))
    #     for i in range(h_in):
    #         for j in range(w_in):
    #             diff_out = diff_pad[:, i:i+k, j:j+k, :].reshape(n, -1)
    #             out = diff_out.dot(rotated_weight)
    #             back_diff[:, i, j, :] = out.reshape(n, -1)
    #
    #     # weight_diff, bias_diff = self.sgd_momentum(weight_diff, bias_diff)
    #     self.weights -= lr * weight_diff.T.reshape(self.weights.shape)
    #     self.bias -= lr * bias_diff
    #     n, h, w, d = self.x.shape
    #     back_diff = back_diff.reshape(n, d, h, w)
    #
    #     return back_diff
    #
    # def sgd_momentum(self, weight_diff, bias_diff):
    #     self.weight_diff = self.momentum * self.weight_diff + (1 - self.momentum) * weight_diff
    #     self.bias_diff = self.momentum * self.bias_diff + (1 - self.momentum) * bias_diff
    #     return self.weight_diff, self.bias_diff
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
        # X_col =col2im_indices(X_col, self.out.shape, h_out, w_out)
        # print(self.out.shape, X_col.shape)
        self.cache = (X, self.weights, self.bias, X_col)

        return self.out



    def backward(self, dy, lr):
        X, W, b, X_col = self.cache
        n_filter, d_filter, h_filter, w_filter = W.shape

        db = np.sum(dy, axis=(0, 2, 3))
        db = db.reshape(n_filter, -1)
        # print(dy.shape, X_col.shape)
        dy_reshaped = dy.transpose(1, 2, 3, 0).reshape(n_filter, -1)
        # print(dy_reshaped.shape)
        dW = dy_reshaped @ X_col.T
        dW = dW.reshape(W.shape)

        W_reshape = W.reshape(n_filter, -1)
        dX_col = W_reshape.T @ dy_reshaped
        dX = col2im_indices(dX_col, X.shape, h_filter, w_filter)
        # print(dX.shape)
        self.weights -= lr * dW
        self.bias -= lr * db
        return dX

