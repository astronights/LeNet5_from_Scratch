from abc import ABC
import math
from scipy import signal
from .helper import Helper
import numpy as np

from .activations import ReLU, Sigmoid, tanh


class LeNetLayer(ABC):
    def __init__(self, id, num_kernels, kernel_dims, input_size, padding, stride, activation, load):
        self.id = id
        self.input_size = input_size
        self.num_kernels = num_kernels
        self.kernel_dims = kernel_dims
        self.padding = padding
        self.stride = stride
        self.activation = eval(activation)() if activation else None
        self.kernel = None
        self.load = load

    def _calc_output_size(self, N, F, p, stride, depth):
        try:
            dim = int((N[1]-F[1]+(2*p))/stride + 1)
        except:
            print("Missing values in sizes...")
            return(None)
        return((depth, dim, dim))

    def _gen_kernels(self,w_size, b_size):
        return({"weights": np.random.uniform(-0.1, 0.1, w_size), "bias": np.random.uniform(-0.1, 0.1, b_size)})

    def _load_path(self, id):
        path = "model/" + id + ".npz"
        npz = np.load(path)
        return({"weights": npz['weights'], "bias": npz['bias']})

    def print_layer(self, name):
        print(self.id, ": ", name)
        print("Input Size: ", self.input_size)
        print("Number of kernels: ", self.num_kernels)
        print("Kernal Dimensions: ", self.kernel_dims)
        print("Stride: ", self.stride)
        print("Padding: ", self.padding)
        if(self.activation):
            print("Activation Function: ", self.activation.__class__.__name__)
        else:
            print("Activation Function:  None")
        print("Output Size: ", self.output_size)
        print()


class Input(LeNetLayer):
    def __init__(self, id, input_size, kernel_dims=(0,0), num_kernels=1, padding=0, stride=1, activation=None, load=False):
        LeNetLayer.__init__(self, id, num_kernels, kernel_dims, input_size, padding, stride, activation, load)
        self.output_size = (1, 32, 32)

    def forward_prop_og(self, image):
        new_shape = (image.shape[0],) + self.output_size
        return(image.reshape(new_shape))

    def backward_prop_og(self, delta, lr):
        pass


class Convolution(LeNetLayer):
    def __init__(self, id, num_kernels, kernel_dims, input_size=None, padding=0, stride=1, activation=None, load=False):
        LeNetLayer.__init__(self, id, num_kernels, kernel_dims, input_size, padding, stride, activation, load)
        self.output_size = self._calc_output_size(self.input_size, self.kernel_dims,
                                                  self.padding, self.stride, self.num_kernels)
        if(load):
            self.kernel = self._load_path(self.id)
        else:
            self.kernel = self._gen_kernels((self.num_kernels, self.input_size[0], self.kernel_dims[0], self.kernel_dims[1]), self.num_kernels)


    def forward_prop_og(self, image):
        self.inputs = image
        image_vec = Helper.im2col(image, self.kernel_dims, self.stride, self.padding)
        weights_vec = self.kernel['weights'].reshape(self.kernel['weights'].shape[0], -1).T

        output = np.dot(image_vec, weights_vec) + self.kernel['bias']
        # print(output.shape, self.output_size)
        output = output.reshape((len(self.inputs), self.output_size[1], self.output_size[2], -1)).transpose(0,3,1,2)
        if(self.activation):
            output = self.activation.forward_prop(output)
        return(output)

    def backward_prop_og(self, delta, lr):
        if(self.activation):
            delta = self.activation.backward_prop(delta)

        delta = delta.transpose(0, 2, 3, 1).reshape(-1, self.num_kernels)

        delta_b = np.sum(delta, axis=0)

        delta_w = np.dot(Helper.im2col(self.inputs, self.kernel_dims, self.stride, self.padding).T, delta)
        delta_w = delta_w.transpose(1, 0).reshape(self.kernel['weights'].shape)

        delta_x = np.dot(delta, self.kernel['weights'].reshape(self.kernel['weights'].shape[0], -1))
        delta_x = Helper.col2im(delta_x, self.inputs.shape, self.kernel_dims, self.stride, self.padding)

        self.kernel['weights'] -= (delta_w)*lr
        self.kernel['bias'] -= (delta_b)*lr

        return delta_x

class MaxPooling(LeNetLayer):
    def __init__(self, id, kernel_dims, num_kernels=1, input_size=None, padding=0, stride=1, activation=None, load=False):
        LeNetLayer.__init__(self, id, num_kernels, kernel_dims, input_size, padding, stride, activation, load)
        self.output_size = self._calc_output_size(self.input_size, self.kernel_dims,
                                                  self.padding, self.stride, self.input_size[0])


    def forward_prop(self, image):
        res = measure.block_reduce(image, (2,2,1), np.max)
        if(self.activation == 'RelU'):
            return(np.maximum(res, 0))
        return(res)

    def forward_prop_og(self, image):
        self.inputs = image
        output = image.reshape(image.shape[0], image.shape[1], image.shape[2]//self.kernel_dims[0], self.kernel_dims[0],
                               image.shape[3]//self.kernel_dims[1], self.kernel_dims[1]).max(axis=(3,5))
        self.outputs = output
        if(self.activation):
            output = self.activation.forward_prop(output)
        return(output)

    def backward_prop_og(self, delta, lr):
        if(self.activation):
            delta = self.activation.backward_prop(delta)
        max_vals_map = np.repeat(np.repeat(self.outputs, self.stride, axis=2), self.stride, axis=3)
        delta_map = np.repeat(np.repeat(delta, self.stride, axis=2), self.stride, axis=3)
        delta_x = (max_vals_map == self.inputs) * delta_map
        return(delta_x)

class FullyConnected(LeNetLayer):
    def __init__(self, id, input_size, output_size, num_kernels=1, padding=0, stride=1, activation=None, load=False):
        kernel_dims = (input_size[0], output_size)
        LeNetLayer.__init__(self, id, num_kernels, kernel_dims, input_size, padding, stride, activation, load)
        self.output_size = (output_size, 1, 1)
        if(load):
            self.kernel = self._load_path(self.id)
        else:
            self.kernel = self._gen_kernels(self.kernel_dims, self.output_size[0])


    def forward_prop(self, image):
        res = np.matmul(image, self.kernels[0]['weights'])
        return(res)

    def forward_prop_og(self, image):
        self.inputs = image
        output = np.dot(np.squeeze(image), self.kernel['weights']) + self.kernel['bias']
        output = output.reshape((image.shape[0], ) + self.output_size)
        if(self.activation):
            output = self.activation.forward_prop(output)
        return(output)

    def backward_prop_og(self, delta, lr):
        if(self.activation):
            delta = self.activation.backward_prop(delta)
        delta_x = np.dot(np.squeeze(delta), self.kernel['weights'].T)
        delta_w = np.dot(np.squeeze(self.inputs.T) , np.squeeze(delta))
        delta_b = np.sum(delta, axis=0)
        # print(delta_b)
        self.kernel['weights'] -= np.squeeze(delta_w)*lr
        self.kernel['bias'] -= np.squeeze(delta_b)*lr
        # chomu
        return(delta_x.reshape(delta_x.shape + (1,1)))
