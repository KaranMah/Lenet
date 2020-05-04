from utils.convol_3 import Convol2d
from utils.conv_4 import  conv
# from utils.maxpool import Maxpooling
# from utils.fully_connected import FullyConnected
# from utils.ReLU import ReLu
from utils.loss import cross_entropy
from utils.layers import *
from datetime import datetime
# from utils.softmax import Softmax


class LeNet5(object):
    def __init__(self):
        self.layers = []
        # self.layers.append(Convolution(num_filters=(1,6), kernel_size=5, padding=0, stride=1,
        #                   name='conv1'))
        # self.layers.append(Maxpooling(pool_size=2, stride=2, name='maxpool2'))
        # self.layers.append(ReLu(name='ReLu'))
        # self.layers.append(Convolution( num_filters=(6,16), kernel_size=5, padding=0, stride=1,
        #                   name='conv3'))
        # self.layers.append(Maxpooling(pool_size=2, stride=2, name='maxpool4'))
        # self.layers.append(ReLu(name='ReLu'))
        # self.layers.append(Convolution(num_filters=(16,120), kernel_size=5, padding=0, stride=1,
        #                   name='conv5'))
        # self.layers.append(ReLu(name='ReLu'))
        # self.layers.append(FullyConnected(num_inputs=120, num_outputs=84, name='fc6'))
        # self.layers.append(ReLu(name='ReLu'))
        # self.layers.append(FullyConnected(num_inputs=84, num_outputs=10, name='fc7'))
        # self.layers.append(Softmax())
        # self.layers.append(Convol2d(layer_size=(6, 28, 28), kernel_size=(6, 1, 5, 5), name="conv1"))
        # self.layers.append(Convol2d(layer_size=(16, 10, 10), kernel_size=(16, 6, 5, 5), name="conv3"))
        # self.layers.append(Convol2d(layer_size=(120, 1, 1), kernel_size=(120, 16, 5, 5), name="conv5"))

        self.layers.append(conv(cin=1,cout=6,filtersize=5, name="conv1"))
        self.layers.append(Maxpooling(size=2, stride=2, name="maxpool2"))
        self.layers.append(ReLu(name='ReLu'))
        self.layers.append(conv(cin=6, cout=16, filtersize=5, name="conv3"))
        self.layers.append(Maxpooling(size=2, stride=2, name="maxpool4"))
        self.layers.append(ReLu(name='ReLu'))
        self.layers.append(conv(cin=16, cout=120, filtersize=5, name="conv5"))
        self.layers.append(ReLu(name='ReLu'))
        self.layers.append(FullyConnected(layer_size=120 * 84,kernel_size=(120, 84),name="fc6"))
        self.layers.append(ReLu(name='ReLu'))
        self.layers.append(FullyConnected(layer_size=84 * 10, kernel_size=(84, 10), name="fc7"))
        self.layers.append(Softmax(name='softmax'))
        self.final = []

    def Forward_Propagation(self,batch_image, batch_label, mode):

        if mode == "train":
            N,W,H,D =(batch_image.shape)
            batch_image = batch_image.reshape(N,D,H,W)
            for layer in self.layers:
                batch_image = layer.forward(batch_image)
            self.final = np.eye(10)[batch_label]
            return cross_entropy(batch_image, batch_label)
        else:
            N, W, H, D = (batch_image.shape)
            batch_image = batch_image.reshape(N, D, H, W)
            for layer in self.layers:
                batch_image = layer.forward(batch_image)
            res = np.argmax(batch_image, axis=1)
            loss = cross_entropy(batch_image, batch_label)
            return loss, res

    def Back_Propagation(self, lr):
        rev_layers = reversed(self.layers)
        training_data = self.final
        for layer in rev_layers:
            training_data = layer.backward(training_data, lr)
        self.final = training_data
        return



