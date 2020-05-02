import time
import numpy as np
from .layers import Input, Convolution, MaxPooling, FullyConnected

# LeNet5 object
class LeNet5(object):
    def __init__(self):
        self.layers = {}
        self.layer_names = ['I0', 'C1', 'S2', 'C3', 'S4', 'C5', 'F6', 'F7']
        self.layers['I0'] = Input(id = 'I0', num_kernels=0,
                                  input_size=(32, 32, 1))
        self.layers['C1'] = Convolution(id = 'C1', num_kernels=6, kernel_dims=(5,5),
                                        input_size=self.layers['I0'].output_size)
        self.layers['S2'] = MaxPooling(id = 'S2', kernel_dims=(2,2), stride=2,
                                       input_size=self.layers['C1'].output_size,
                                       activation='ReLU')
        self.layers['C3'] = Convolution(id = 'C3', num_kernels=16, kernel_dims=(5,5),
                                        input_size=self.layers['S2'].output_size)
        self.layers['S4'] = MaxPooling(id = 'S4', kernel_dims=(2,2), stride=2,
                                       input_size=self.layers['C3'].output_size,
                                       activation='ReLU')
        self.layers['C5'] = Convolution(id = 'C5', num_kernels=120, kernel_dims=(5,5),
                                        input_size=self.layers['S4'].output_size,
                                        activation='ReLU')
        self.layers['F6'] = FullyConnected(id= 'F6',
                                           input_size=self.layers['C5'].output_size,
                                           output_size=84,
                                           activation='ReLU')
        self.layers['F7'] = FullyConnected(id= 'F7',
                                           input_size=self.layers['F6'].output_size,
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
        print("Inside entropy loss...")
        print(true_vals.shape, probs.shape)
        entropy = -np.sum(true_vals * np.log(probs))/probs.shape[0]
        return(entropy)

    def Forward_Propagation(self, input_image, input_label, mode):
        print("Forward Propagation")
        if(mode == 'train'):
            self.y_true = self.one_hot_y(input_label)
            cur_image = input_image
            for layer in self.layers:
                print(layer+":", time.ctime())
                temp_cur = self.layers[layer].forward_prop_og(cur_image)
                cur_image = temp_cur
            entropy = self.entropy_loss(cur_image)
            print(entropy)
            res = np.argmax(self.softmax(cur_image), axis=1)
            print(res)
            self.save_to_file(res)
            print(input_label)
            return(entropy)
        else:
            self.y_true = self.one_hot_y(input_label)
            cur_image = input_image
            for layer in self.layers:
                print(layer+":", time.ctime())
                print(cur_image.shape)
                temp_cur = self.layers[layer].forward_prop_og(cur_image)
                cur_image = temp_cur
            entropy = self.entropy_loss(cur_image)
            print(entropy)
            res = np.argmax(self.softmax(cur_image), axis=1)
            return(entropy, res)

    def save_to_file(self, res):
        with open('lenet_res.txt', 'a') as outfile:
            outfile.write(''.join([str(x) for x in res]))
            outfile.write('\n')

    def one_hot_y(self, labels):
        cat = np.eye(10)[labels]
        return(cat.reshape((cat.shape[0],) + self.layers[self.layer_names[-1]].output_size))

    def softmax_delta(self, true):
        y_true = np.squeeze(true)
        delta = (self.y_pred - y_true)/self.y_pred.shape[0]
        return(delta.reshape(delta.shape + (1,1)))

    def Back_Propagation(self, lr_global):
        print("Back Propagation")
        delta = self.softmax_delta(self.y_true)
        print(delta.shape)
        for layer in self.layer_names[::-1]:
            print(layer+":", time.ctime())
            # print(delta.shape)
            temp_delta = self.layers[layer].backward_prop_og(delta, lr_global)
            delta = temp_delta

        self.save_model()

    def save_model(self):
        directory = 'model/'
        for layer in self.layer_names:
            if(layer[0] == 'C' or layer[0] == 'F'):
                np.savez_compressed(directory+layer, weights=self.layers[layer].kernel["weights"], bias=self.layers[layer].kernel["bias"])
