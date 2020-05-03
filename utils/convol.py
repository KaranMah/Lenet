import numpy as np
import scipy.signal

class Convolution:

    def __init__(self, num_filters, kernel_size, padding, stride, name):
        self.F = num_filters[1]
        self.C = num_filters[0]
        self.K = kernel_size

        self.weights = np.zeros((self.F, self.C, self.K, self.K))
        self.bias = np.zeros((self.F, 1))
        for i in range(0, self.F):
            self.weights[i, :, :, :] = np.random.normal(loc=0, scale=np.sqrt(1. / (self.C * self.K * self.K)),
                                                        size=(self.C, self.K, self.K))
        self.p = padding
        self.s = stride
        self.name = name



    def forward(self, inputs):

        C = inputs.shape[0]
        D = inputs.shape[1]
        W = inputs.shape[2] + 2 * self.p
        H = inputs.shape[3] + 2 * self.p

        self.inputs = inputs
        WW = int((W - self.K)/self.s) + 1
        HH = int((H - self.K)/self.s) + 1
        feature_maps = np.zeros((C, self.F, WW, HH))

        weights_rot = np.rot90(self.weights, 2, (2, 3))
        for img in range(C):
            for conv_depth in range(self.F):
                for inp_depth in range(D):
                    feature_maps[img, conv_depth] += scipy.signal.convolve2d(inputs[img, inp_depth],
                                                                                 weights_rot[conv_depth, inp_depth],
                                                                                 mode='valid')
                feature_maps[img, conv_depth] += self.bias[conv_depth]

        return feature_maps

    def backward(self, dy, lr):

        N, C, W, H = self.inputs.shape
        dx = np.zeros(self.inputs.shape)
        dw = np.zeros(self.weights.shape)
        db = np.zeros(self.bias.shape)
        print(dx.shape)

        conv_h = (H - self.K ) // self.s + 1
        conv_w = (W - self.K) // self.s + 1

        N, F, W, H = dy.shape
        for img in range(N):
            for f in range(F):
                for w in range(W-self.K+1):
                    for h in range(H-self.K+1):
                        dx[img,:,w:w+self.K,h:h+self.K] += dy[img,f,w,h]*self.weights[img,:,w:w+self.K,h:h+self.K]

        for img in range(N):
            for kernel_num in range(F):
                for h in range(conv_h):
                    for w in range(conv_w):
                        dw[kernel_num, :, :, :] += dy[img, kernel_num, h, w] * self.inputs[img, :,
                                                                                            h:h + self.K,
                                                                                            w:w + self.K]

        self.db = np.sum(dy, (0, 2, 3))
        self.weights -= lr * dw
        self.bias -= lr * db
        return dx

    def extract(self):
        return {self.name+'.weights':self.weights, self.name+'.bias':self.bias}

    def feed(self, weights, bias):
        self.weights = weights
        self.bias = bias
