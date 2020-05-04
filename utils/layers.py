import numpy as np
from utils.util import *
import scipy.signal

class Convolution:
    def __init__(self, layer_size, kernel_size, name):
        """
        layer_size: (6,32,32)
        kernal_size: (6,1,32,32)
        fan: (1,6)

        Input:
            layer_size: tuple consisting (depth, height, width)
            kernel_size: tuple consisting (number_of_kernels, inp_depth, inp_height, inp_width)
            fan: tuple of number of nodes in previous layer and this layer
            params: directory consists of pad_len and stride,
                    filename (to load weights from file)
        """
        self.depth, self.height, self.width = layer_size
        self.weights = np.random.uniform(-0.1, 0.1 , kernel_size)
        self.bias = np.random.uniform(-0.1, 0.1 , kernel_size[0])
        self.name = name


    def forward(self, input):
        """
        Computes the forward pass of Conv Layer.
        Input:
            X: Input data of shape (N, D, H, W)
        Variables:
            kernel: Weights of shape (K, K_D, K_H, K_W)
            bias: Bias of each filter. (K)
        where, N = batch_size or number of images
               H, W = Height and Width of input layer
               D = Depth of input layer
               K = Number of filters/kernels or depth of this conv layer
               K_H, K_W = kernel height and Width
        Output:
        """
        N, D, H, W = input.shape
        K, K_D, K_H, K_W = self.weights.shape

        assert self.depth == K

        assert D == K_D
        assert K == self.bias.shape[0]

        conv_h = (H - K_H ) + 1
        conv_w = (W - K_W ) + 1

        # print("\n\n%d, %d "%(self.height,conv_h))

        assert self.height == conv_h

        assert self.width == conv_w

        # feature map of a batch
        self.out = np.zeros([N, K, conv_h, conv_w])


        kernel_180 = np.rot90(self.weights, 2, (2,3))
        for img in range(N):
            for conv_depth in range(K):
                for inp_depth in range(D):
                    self.out[img, conv_depth] += scipy.signal.convolve2d(input[img, inp_depth], kernel_180[conv_depth, inp_depth], mode='valid')
                self.out[img, conv_depth] += self.bias[conv_depth]


        self.cache = input

        return self.out


    def backward(self, dy, lr):
        """
        Computes the backward pass of Conv layer.
        Input:
            delta: derivatives from next layer of shape (N, K, conv_h, conv_w)
        """

        input = self.cache


        N, D, H, W = input.shape
        K, K_D, K_H, K_W = self.weights.shape

        assert self.depth == K
        assert D == K_D
        assert K == self.bias.shape[0]

        conv_h = (H - K_H ) + 1
        conv_w = (W - K_W ) + 1

        assert self.height == conv_h
        assert self.width == conv_w



        dx = np.zeros(input.shape)
        self.dw = np.zeros(self.weights.shape)
        self.db = np.zeros(self.bias.shape)

        # Delta X
        for img in range(N):
            for conv_depth in range(K):
                for h in range(0, H  - K_H + 1):
                    for w in range(0, W - K_W + 1):
                        dx[img, :, h:h+K_H, w:w+K_W] += dy[img, conv_depth, h, w] * self.weights[conv_depth]

        assert dx.shape == input.shape

        # Delta kernel
        for img in range(N):
            for kernel_num in range(K):
                for h in range(conv_h):
                    for w in range(conv_w):
                        self.dw[kernel_num,:,:,:] += dy[img, kernel_num, h, w] * dx[img, :, h:h+K_H, w:w+K_W]

        # Delta Bias
        self.delta_b = np.sum(dy, (0,2,3))
        self.weights -= lr * self.dw
        self.bias -= lr * self.db
        return dx

class FullyConnected:
    """docstring forRELU_LAYER."""
    def __init__(self, layer_size, kernel_size, name):
        """
        layer_size =
        kernal = 120,84
        Input:
            layer_size: number of neurons/nodes in fc layer
            kernel: kernel of shape (nodes_l1 , nodes_l2)
        """
        self.nodes = layer_size
        self.weights = np.random.uniform(-0.1, 0.1, kernel_size)
        self.bias = np.random.uniform(-0.1, 0.1, kernel_size[1])
        self.name = name

    def forward(self, X):
        """
        Computes the forward pass of Sigmoid Layer.
        Input:
            X: Input data of shape (N, nodes_l1)
        Variables:
            kernel: Weight array of shape (nodes_l1, nodes_l2)
            bias: Biases of shape (nodes_l2)
        where,
            nodes_l1: number of nodes in previous layer
            nodes_l2: number of nodes in this fc layer
        """
        weights, bias = self.weights, self.bias
        self.cache = (X, weights, bias)
        self.out = np.dot(np.squeeze(X), weights) + bias
        assert self.out.shape == (X.shape[0], bias.shape[0])
        return self.out

    def backward(self, dy, lr):
        """
        Computes the backward pass of Sigmoid Layer.
        Input:
            delta: Shape of delta values (N, nodes_l2)
        """
        X, weights, bias = self.cache

        self.dx = np.dot(dy, weights.T)
        self.dx=self.dx.reshape(X.shape)
        self.dw = np.dot(X.T, dy)
        self.db = np.sum(dy, axis=0)
        self.weights -= np.squeeze(lr * self.dw)
        self.bias -= lr * self.db
        return self.dx


class Maxpooling:
    def __init__(self, size, stride=1, pad=0, name="pool"):
        self.par = None
        self.pool_h = size
        self.pool_w = size
        self.stride = stride
        self.pad = pad
        self.name = name

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
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
                    self.pool_w, self.stride, self.pad)

        return dx
# class Maxpooling:
#         """MAX_POOL_LAYER only reduce dimensions of height and width by a factor.
#            It does not put max filter on same input twice i.e. stride = factor = kernel_dimension
#         """
#
#         def __init__(self, size, stride, name):
#             self.size = size
#             self.factor = stride
#             self.name = name
#
#         def forward(self, X):
#             """
#             Computes the forward pass of MaxPool Layer.
#             Input:
#                 X: Input data of shape (N, D, H, W)
#             where, N = batch_size or number of images
#                    H, W = Height and Width of input layer
#                    D = Depth of input layer
#             """
#             factor = self.factor
#             # N, D, H, W = X.shape
#             # self.cache = X
#             # self.out = X.reshape(N, D, H // factor, factor, W // factor, factor).max(axis=(3, 5))
#             # assert self.out.shape == (N, D, H//factor, W//factor)
#             n, d, h, w = X.shape
#             x_grid = X.reshape(n, d, h // factor, factor, w // factor, factor)
#             out = np.max(x_grid, axis=(3, 5))
#             self.mask = (out.reshape(n, d, h // factor, 1, w // factor, 1) == x_grid)
#             return out
#             return self.out
#
#
#         def backward(self, dy, lr):
#             """
#             Computes the backward pass of MaxPool Layer.
#             Input:
#                 dy: delta values of shape (N, D, H/factor, W/factor)
#             """
#
#             n, c, h, w = dy.shape
#             diff_grid = dy.reshape(n, c, h, 1, w, 1)
#             return (diff_grid * self.mask).reshape(n, c, h * self.factor, w * self.factor)
#             # X = self.cache
#             # if len(dy.shape) != 4:  # then it must be 2
#             #     assert dy.shape[0] == X.shape[0]
#             #
#             #     delta = dy.reshape(self.out.shape)
#             #
#             # fmap = np.repeat(np.repeat(self.out, self.factor, axis=2), self.factor, axis=3)
#             # dmap = np.repeat(np.repeat(dy, self.factor, axis=2), self.factor, axis=3)
#             # assert fmap.shape == X.shape and dmap.shape == X.shape
#             #
#             # self.dx = np.zeros(X.shape)
#             # self.dx = (fmap == X) * dmap
#             #
#             # assert self.dx.shape == X.shape
#             # return self.dx


class ReLu:
    """docstring forRELU_LAYER."""
    def __init__(self, name):
        self.name = name
        pass

    def forward(self, X):
        """
        Computes the forward pass of Relu Layer.
        Input:
            X: Input data of any shape
        """
        self.cache = X
        self.out = (X > 0) * X
        return self.out

    def backward(self, dy, lr):
        """
        Computes the backward pass of Relu Layer.
        Input:
            delta: Shape of delta values should be same as of X in cache
        """
        self.dx = dy * (self.cache > 0)
        return self.dx

class Softmax:
    """docstring forRELU_LAYER."""
    def __init__(self, name):
        self.name = name
        pass

    def forward(self, X):
        """
        Computes the forward pass of Softmax Layer.
        Input:
            X: Input data of shape (N, C)
        where,
            N: Batch size
            C: Number of nodes in SOFTMAX_LAYER or classes
        Output:
            Y: Final output of shape (N, C)
        """
        self.cache = X
        dummy = np.exp(X)
        self.Y = dummy/np.sum(dummy, axis=1, keepdims=True)
        # print(self.Y)
        return self.Y

    def backward(self, output, lr):
        """
        Computes the backward pass of Softmax Layer.
        Input:
            output: Training set ouput of shape (N, C)
        """
        assert self.Y.shape == output.shape
        self.dx =  (self.Y - output) / self.Y.shape[0]
        return self.dx

    def softmax_loss(self, Y, output):
        """
        Computes loss using cross-entropy method.
        Input:
            Y: Predicted output of network of shape (N, C)
            output: real output of shape (N, C)
        where,
            N: batch size
            C: Number of classes in the final layer
        """
        assert Y.shape == output.shape
        epsilon = 1e-10
        self.loss = (-output * np.log(Y + epsilon)).sum() / Y.shape[0]
        pass