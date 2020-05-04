import numpy as np

class Convol2d:

    def __init__(self, layer_size, kernel_size, name):
        self.depth, self.height, self.width = layer_size
        # f = np.sqrt(6) / np.sqrt(kernel_size[1] + layer_size[0])
        # epsilon = 1e-6
        # self.weights = np.random.uniform(-f, f+epsilon, kernel_size)
        # self.bias = np.random.uniform(-f, f+epsilon, kernel_size[0])
        # self.bias = np.mat(self.bias)
        self.name = name
        self.weight_diff = 0
        self.bias_diff = 0
        num_filters, in_channels, self.k_size, k_size = kernel_size
        self.filters = (np.random.random(size=(self.k_size, self.k_size, in_channels, num_filters)) * 2 - 1) / np.sqrt(
            1.0 / k_size * num_filters)
        self.bias = np.zeros(num_filters)
        self.momentum = 0.95

    def forward(self, inputs):
        """
        inputs: B x H x W x C  ---- (B)atch size x (H)eight x (W)idth x (C)hannel size
        outputs: B x O1 x O2 x F  ---- (B)atch size x (O1)Height x (O2)Width x (F)ilter size
        O1 = (W - K1 + 2*P ) / S + 1
        O2 = (L - K2 + 2*P ) / S + 1
        F: number of filters
        """
        (batch_size, C, H_in, W_in) = inputs.shape
        inputs = inputs.reshape(batch_size, H_in, W_in, C)
        (kernel_size, kernel_size, C_in, C_out) = self.filters.shape
        print(inputs.shape, self.filters.shape)
        H_out = (H_in - kernel_size) + 1
        W_out = (W_in - kernel_size) + 1

        # inputs_padded = self.zero_pad(inputs, self.padding)

        self.outputs = np.zeros((batch_size, H_out, W_out, C_out))

        for h in range(H_out):
            for w in range(W_out):
                h1 = h
                h2 = h  + self.k_size
                w1 = w
                w2 = w  + self.k_size
                self.outputs[:, h, w, :] = np.sum(np.expand_dims(self.filters, 0) *
                                                  np.expand_dims(inputs[:, h1:h2, w1:w2, :], -1),
                                                  axis=(1, 2, 3)) + self.bias
        self.inputs = inputs
        self.outputs = self.outputs.reshape(batch_size, C_out,H_out,W_out)
        return self.outputs

    def backward(self, grad_outputs, lr):
        '''
        Backpropagation through a convolutional layer.
        '''
        (batch_size, H_in, W_in, C_input) = self.inputs.shape
        # inputs = inputs.reshape(batch_size, H_in, W_in, C_input)
        (batch_size, C_out, H_out, W_out) = grad_outputs.shape
        (kernel_size, kernel_size, C_in, C_out) = self.filters.shape

        grad_inputs = np.zeros(self.inputs.shape)
        grad_filters = np.zeros(self.filters.shape)
        grad_biases = np.zeros((C_out, 1))

        for h in range(H_out):
            for w in range(W_out):
                h1 = h
                h2 = h + self.k_size
                w1 = w
                w2 = w + self.k_size
                grad_outputs_reshaped = np.expand_dims(np.expand_dims(grad_outputs[:, h, w, :], 1), 2)
                print(grad_outputs_reshaped.shape, self.inputs[:, h1:h2, w1:w2, :].shape, grad_filters.shape)
                grad_filters += np.sum(
                    np.expand_dims(grad_outputs_reshaped, 3) * np.expand_dims(self.inputs[:, h1:h2, w1:w2, :], 4), axis=0)
                grad_inputs[:, h1:h2, w1:w2, :] += np.sum(
                    np.expand_dims(grad_outputs_reshaped, 3) * np.expand_dims(self.filters, 0), axis=4)

        grad_biases = grad_outputs.mean(axis=(0, 1, 2)) * self.inputs.shape[0]

        grad_biases = np.squeeze(grad_biases)
        self.filters -= lr * grad_filters
        self.bias -= lr * grad_biases
        grad_inputs = grad_inputs.reshape(batch_size, C_input, H_in, W_in)
        return grad_inputs
