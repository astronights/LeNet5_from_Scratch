import time
import numpy as np
from .layers import Input, Convolution, MaxPooling, FullyConnected

# LeNet5 object
class LeNet5(object):
    def __init__(self, activation):
        self.layers = {}
        self.layer_names = ['I0', 'C1', 'S2', 'C3', 'S4', 'C5', 'F6', 'F7']
        load_models = False
        self.layers['I0'] = Input(id = 'I0', num_kernels=0,
                                  input_size=(32, 32, 1))
        self.layers['C1'] = Convolution(id = 'C1', num_kernels=6, kernel_dims=(5,5), load=load_models,
                                        input_size=self.layers['I0'].output_size)
        self.layers['S2'] = MaxPooling(id = 'S2', kernel_dims=(2,2), stride=2,
                                       input_size=self.layers['C1'].output_size,
                                       activation=activation)
        self.layers['C3'] = Convolution(id = 'C3', num_kernels=16, kernel_dims=(5,5), load=load_models,
                                        input_size=self.layers['S2'].output_size)
        self.layers['S4'] = MaxPooling(id = 'S4', kernel_dims=(2,2), stride=2,
                                       input_size=self.layers['C3'].output_size,
                                       activation=activation)
        self.layers['C5'] = Convolution(id = 'C5', num_kernels=120, kernel_dims=(5,5), load=load_models,
                                        input_size=self.layers['S4'].output_size,
                                        activation=activation)
        self.layers['F6'] = FullyConnected(id= 'F6',
                                           input_size=self.layers['C5'].output_size, load=load_models,
                                           output_size=84,
                                           activation=activation)
        self.layers['F7'] = FullyConnected(id= 'F7',
                                           input_size=self.layers['F6'].output_size, load=load_models,
                                           output_size=10)
        self.y_true = None
        self.y_pred = None
        self.print_net()

    def print_net(self):
        for layer in self.layers:
            self.layers[layer].print_layer(self.layers[layer].__class__.__name__)

    def softmax(self, image):
        image = np.squeeze(image)
        z = np.exp(image)
        return(z/z.sum(axis=1, keepdims=True))

    def entropy_loss(self, image):
        probs = self.softmax(image)
        self.y_pred = probs
        true_vals = np.squeeze(self.y_true)
        entropy = -np.sum(true_vals * np.log(probs))
        return(entropy)

    def Forward_Propagation(self, input_image, input_label, mode):
        if(mode == 'train'):
            self.y_true = self.one_hot_y(input_label)
            cur_image = input_image
            for layer in self.layers:
                temp_cur = self.layers[layer].forward_prop(cur_image)
                cur_image = temp_cur
            entropy = self.entropy_loss(cur_image)
            res = np.argmax(self.softmax(cur_image), axis=1)
            accuracy = 1 - (np.count_nonzero((res-input_label))/len(input_label))
            return(entropy)
        else:
            self.y_true = self.one_hot_y(input_label)
            cur_image = input_image
            for layer in self.layers:
                temp_cur = self.layers[layer].forward_prop(cur_image)
                cur_image = temp_cur
            entropy = self.entropy_loss(cur_image)
            res = np.argmax(self.softmax(cur_image), axis=1)
            return(entropy, res)

    def one_hot_y(self, labels):
        cat = np.eye(10)[labels]
        return(cat.reshape((cat.shape[0],) + self.layers[self.layer_names[-1]].output_size))

    def softmax_delta(self, true):
        y_true = np.squeeze(true)
        delta = (self.y_pred - y_true)/self.y_pred.shape[0]
        return(delta.reshape(delta.shape + (1,1)))

    def Back_Propagation(self, lr_global):
        delta = self.softmax_delta(self.y_true)
        for layer in self.layer_names[::-1]:
            temp_delta = self.layers[layer].backward_prop(delta, lr_global)
            delta = temp_delta

        # self.save_layers()

    def save_layers(self):
        directory = 'layer/'
        for layer in self.layer_names:
            if(layer[0] == 'C' or layer[0] == 'F'):
                np.savez_compressed(directory+layer, weights=self.layers[layer].kernel["weights"], bias=self.layers[layer].kernel["bias"])
