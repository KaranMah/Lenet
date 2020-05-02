from utils.convol import Convolution
from utils.maxpool import Maxpooling
from utils.fully_connected import FullyConnected
from utils.ReLU import ReLu
from utils.loss import cross_entropy
from utils.softmax import Softmax


class LeNet5(object):
    def __init__(self):
        self.layers = []
        self.layers.append(Convolution(num_filters=(1,6), kernel_size=5, padding=0, stride=1,
                          name='conv1'))
        self.layers.append(Maxpooling(pool_size=2, stride=2, name='maxpool2'))
        self.layers.append(ReLu(name='ReLu'))
        self.layers.append(Convolution( num_filters=(6,16), kernel_size=5, padding=0, stride=1,
                          name='conv3'))
        self.layers.append(Maxpooling(pool_size=2, stride=2, name='maxpool4'))
        self.layers.append(ReLu(name='ReLu'))
        self.layers.append(Convolution(num_filters=(16,120), kernel_size=5, padding=0, stride=1,
                          name='conv5'))
        self.layers.append(ReLu(name='ReLu'))
        self.layers.append(FullyConnected(num_inputs=120, num_outputs=84, name='fc6'))
        self.layers.append(ReLu(name='ReLu'))
        self.layers.append(FullyConnected(num_inputs=84, num_outputs=10, name='fc7'))
        self.layers.append(Softmax())
        self.final = []
        # print(len(self.layers))

    def Forward_Propagation(self,batch_image, batch_label, mode):

        if mode == "train":
            N,W,H,D =(batch_image.shape)
            batch_image = batch_image.reshape(N,D,W,H)
            for layer in self.layers:
                batch_image = layer.forward(batch_image)
                # print(layer.name, batch_image.shape)
            # print(batch_image)
            self.final = batch_image
            return cross_entropy(batch_image, batch_label)
        else:
            for layer in self.layers:
                batch_image = layer.forward(batch_image)
            return batch_image

    def Back_Propagation(self, lr):
        rev_layers = reversed(self.layers)
        training_data = self.final
        for layer in rev_layers:
            training_data = layer.backward(training_data, lr)
            print(layer.name, training_data.shape)
        self.final = training_data
        return



