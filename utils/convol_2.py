import numpy as np
from utils.util import *

class Convol2d:

    def __init__(self, layer_size, kernel_size, name):
        self.depth, self.height, self.width = layer_size
        f = np.sqrt(6) / np.sqrt(kernel_size[1] + layer_size[0])
        print(f)
        epsilon = 1e-6
        self.weights = np.random.uniform(-f, f+epsilon, kernel_size)
        self.bias = np.random.uniform(-f, f+epsilon, kernel_size[0])
        self.bias = np.mat(self.bias)
        self.name = name
        self.weight_diff = 0
        self.bias_diff = 0
        self.momentum = 0.95

    def forward(self, x):

        k = self.weights.shape[2]
        n, d, h, w = x.shape
        self.cache = x
        x = x.reshape(n,h,w,d)
        self.x = x
        h_out = h - (k - 1)
        w_out = w - (k - 1)

        w_n = self.weights.shape[0]
        weight = self.weights.reshape(-1, w_n)
        # print(n, h_out, w_out, w_n)
        output = np.zeros((n, h_out, w_out, w_n))
        for i in range(h_out):
            for j in range(w_out):
                inp = x[:, i:i+k, j:j+k, :].reshape(n, -1)
                out = inp.dot(weight) + self.bias
                output[:, i, j, :] = out.reshape(n, -1)
        # print(output.shape)
        output = output.reshape(n, w_n, h_out, w_out)

        return output

    def backward(self, dZ, lr):


        ### START CODE HERE ###
        # Retrieve information from "cache"
        A_prev = self.cache

        # Retrieve dimensions from A_prev's shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        # Retrieve dimensions from W's shape
        (n_C, w_m, f, f) = self.weights.shape


        # Retrieve dimensions from dZ's shape
        (m, n_C, n_H, n_W) = dZ.shape

        # Initialize dA_prev, dW, db with the correct shapes
        dA_prev = np.zeros(A_prev.shape)
        dW = np.zeros(self.weights.shape)
        db = np.zeros(self.bias.shape)

        for i in range(m):  # loop over the training examples

            # select ith training example from A_prev_pad and dA_prev_pad
            a_prev = A_prev[i, :, :, :]
            da_prev = dA_prev[i, :, :, :]
            # print(a_prev.shape, A_prev.shape, da_prev.shape, dA_prev.shape)

            for h in range(n_H):  # loop over vertical axis of the output volume
                for w in range(n_W):  # loop over horizontal axis of the output volume
                    for c in range(n_C):  # loop over the channels of the output volume

                        # Find the corners of the current "slice"
                        vert_start = h
                        vert_end = vert_start + f
                        horiz_start = w
                        horiz_end = horiz_start + f

                        # Use the corners to define the slice from a_prev_pad
                        a_slice = a_prev[:, vert_start: vert_end, horiz_start: horiz_end]
                        # print(db.shape,dW.shape)
                        print(da_prev[:, vert_start:vert_end, horiz_start:horiz_end].shape, self.weights[i, :, :, :].shape, dZ.shape)
                        # Update gradients for the window and the filter's parameters using the code formulas given above
                        da_prev[:, vert_start:vert_end, horiz_start:horiz_end] += self.weights[i, :, :, :] * dZ[i, c, h, w]
                        dW[i, :, :, :] += a_slice * dZ[i, c, h, w]
                        db += dZ[i, c, h, w]

            # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
            dA_prev[i, :, :, :] = da_prev[:,:,:]
        ### END CODE HERE ###

        # Making sure your output shape is correct
        assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

        self.weights -= lr*dW
        self.bias -= lr*db

        return dA_prev
