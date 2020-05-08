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
        dx = col2im(dcol, self.x.shape, FH, FW, self.strid, self.pading)
        self.par['b'] -= lr * self.gra['b']
        self.par['w'] -= lr * self.gra['w']

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

class ReLu:
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
