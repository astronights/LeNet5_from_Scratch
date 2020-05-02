import time
import numpy as np

from .layers_stoch import Convolution, MaxPooling, FullyConnected

# LeNet5 object
class LeNet5(object):
    def __init__(self):
        self.layers = {}
        self.layer_names = ['C1', 'S2', 'C3', 'S4', 'C5', 'F6', 'F7']
        self.layers['C1'] = Convolution(id = 'C1', num_kernels=6, kernel_dims=(5,5),
                                        input_size=(32,32,1))
        self.layers['S2'] = MaxPooling(id = 'S2', kernel_dims=(2,2), stride=2,
                                       input_size=self.layers['C1'].output_size,
                                       activation='ReLU')
        self.layers['C3'] = Convolution(id = 'C3', num_kernels=16, kernel_dims=(5,5),
                                        input_size=self.layers['S2'].output_size)
        self.layers['S4'] = MaxPooling(id = 'S4', kernel_dims=(2,2), stride=2,
                                       input_size=self.layers['C3'].output_size,
                                       activation='ReLU')
        self.layers['C5'] = Convolution(id = 'C5', num_kernels=120, kernel_dims=(5,5),
                                        input_size=self.layers['S4'].output_size)
        self.layers['F6'] = FullyConnected(id= 'F6',
                                           input_size=self.layers['C5'].output_size,
                                           output_size=84)
        self.layers['F7'] = FullyConnected(id= 'F7',
                                           input_size=self.layers['F6'].output_size,
                                           output_size=10)
        self.y_true = None
        self.print_net()

    def print_net(self):
        for layer in self.layers:
            self.layers[layer].print_layer(self.layers[layer].__class__.__name__)

    def loss(self, image, labels):
        image = np.squeeze(image)
        probs = image[np.arange(len(image)), labels]
        entropy = np.nansum(np.log(probs)) #/len(labels) Need to investigate loss function.
        return(entropy)

    def Forward_Propagation(self, input_image, input_label, mode):
        print("Forward Propagation")
        self.y_true = self.one_hot_y(input_label)
        cur_image = input_image
        for layer in self.layer_names:
            print(layer+":", time.ctime())
            layer_size = (input_image.shape[0],) + self.layers[layer].output_size
            temp_cur = np.ndarray(layer_size)
            for i in range(len(cur_image)):
                out = self.layers[layer].forward_prop_og(cur_image[i], i)
                temp_cur[i] = out
            cur_image = temp_cur
            # print(cur_image.shape)
            # print(cur_image[0,0,0,:])
        # print(cur_image.shape, input_label.shape)
        entropy = 0#self.loss(cur_image, input_label)
        # print(entropy)
        res = np.argmax(np.squeeze(cur_image), axis=1)
        return(entropy)

    def check_layer_results():
        with open(str(i)+'test.txt', 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(out.shape))
            for data_slice in out:
                np.savetxt(outfile, data_slice, fmt='%-7.2f')
                outfile.write('# New slice\n')

    def one_hot_y(self, labels):
        cat = np.eye(10)[labels]
        return(cat.reshape((cat.shape[0],) + self.layers[self.layer_names[-1]].output_size))


    def Back_Propagation(self, lr_global):
        print("Back Propagation")
        delta = self.y_true
        for layer in self.layer_names[::-1]:
            print(layer+":", time.ctime())
            # print(delta.shape)
            layer_size = (delta.shape[0],)+self.layers[layer].input_size
            temp_delta = np.ndarray(layer_size)
            for i in range(len(delta)):
                temp_delta[i,:] = self.layers[layer].backward_prop_og(delta[i,:], i, lr_global)
            delta = temp_delta
