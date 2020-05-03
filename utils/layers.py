from abc import ABC
import math
from scipy import signal
import numpy as np

from .activations import ReLU, Sigmoid, tanh


class LeNetLayer(ABC):
    def __init__(self, id, num_kernels, kernel_dims, input_size, padding, stride, activation, path):
        self.id = id
        self.input_size = input_size
        self.num_kernels = num_kernels
        self.kernel_dims = kernel_dims
        self.padding = padding
        self.stride = stride
        self.activation = eval(activation)() if activation else None
        self.kernel = None
        self.path = path

    def _calc_output_size(self, N, F, p, stride, depth):
        try:
            dim = int((N[1]-F[1]+(2*p))/stride + 1)
        except:
            print("Missing values in sizes...")
            return(None)
        return((depth, dim, dim))

    def _gen_kernels(self,w_size, b_size):
        return({"weights": np.random.uniform(-0.1, 0.1, w_size), "bias": np.random.uniform(-0.1, 0.1, b_size)})

    def _load_path(self, path):
        npz = np.load(path)
        return(npz['weights'], npz['bias'])

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
    def __init__(self, id, input_size, kernel_dims=(0,0), num_kernels=1, padding=0, stride=1, activation=None, path=None):
        LeNetLayer.__init__(self, id, num_kernels, kernel_dims, input_size, padding, stride, activation, path)
        self.output_size = (1, 32, 32)

    def forward_prop_og(self, image):
        new_shape = (image.shape[0],) + self.output_size
        return(image.reshape(new_shape))

    def backward_prop_og(self, delta, lr):
        pass


class Convolution(LeNetLayer):
    def __init__(self, id, num_kernels, kernel_dims, input_size=None, padding=0, stride=1, activation=None, path=None):
        LeNetLayer.__init__(self, id, num_kernels, kernel_dims, input_size, padding, stride, activation, path)
        self.output_size = self._calc_output_size(self.input_size, self.kernel_dims,
                                                  self.padding, self.stride, self.num_kernels)
        if(path):
            self.kernel = self._load_path(path)
        else:
            self.kernel = self._gen_kernels((self.num_kernels, self.input_size[0], self.kernel_dims[0], self.kernel_dims[1]), self.num_kernels)



    def _do_dot(self, image, filter):
        return(signal.convolve2d(image, filter, mode='valid').astype(np.float64))

    def forward_prop(self, image):
        output=np.zeros(self.output_size)
        for k in range(output.shape[2]):
            res = (signal.fftconvolve(image, self.kernels[k]['weights'], 'valid'))
            output[:,:,k] = np.squeeze(res) + self.kernels[k]['bias']
        im = np.array([[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]],[[-4,-5],[-1,-2],[0,1]]])
        k = np.array([[[0,1],[1,0]], [[2,3], [3,1]]])
        lol = np.zeros((2,2))
        print(ndimage.convolve(input=im, weights=k, mode='constant', cval=0.0))
        return(output)

    def _transf_kernel(self, kernel):
        return(np.rot90(kernel, 2, (2,3)))


    def forward_prop_og(self, image):
        self.inputs = image
        output=np.zeros( (image.shape[0],) + self.output_size)
        temp_kernel = self._transf_kernel(self.kernel["weights"])
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                for k in range(image.shape[1]):
                    output[i,j] += self._do_dot(image[i, k], temp_kernel[j,k]).astype(np.float64)
                output[i,j] += self.kernel["bias"][j]
        if(self.activation):
            output = self.activation.forward_prop(output)
            # output = np.clip(output, a_min=0.0, a_max=None)
        return(output)

    def backward_prop_og(self, delta, lr):
        if(self.activation):
            delts = self.activation.backward_prop(delta)
        delta_x = np.zeros((delta.shape[0],) + self.input_size)
        delta_w = np.zeros((self.num_kernels, self.input_size[0]) + self.kernel_dims)
        delta_b = np.zeros(self.num_kernels)

        for i in range(delta.shape[0]):
            for j in range(self.num_kernels):
                for k in range(self.input_size[1]-self.kernel_dims[0] + 1):
                    for l in range(self.input_size[2]-self.kernel_dims[1] + 1):
                        delta_x[i, :, k:k+self.kernel_dims[0], l:l+self.kernel_dims[1]] += delta[i, j, k, l].astype(np.float64)*self.kernel['weights'][j].astype(np.float64)
                        delta_w[j,:,:,:] += delta[i, j, k, l].astype(np.float64) * self.inputs[i,:, k:k+self.kernel_dims[0], l:l+self.kernel_dims[0]].astype(np.float64)
        delta_b = np.sum(delta, axis=(0,2,3))
        self.kernel['weights'] -= (delta_w)*lr
        self.kernel['bias'] -= (delta_b)*lr
        return(delta_x)


class MaxPooling(LeNetLayer):
    def __init__(self, id, kernel_dims, num_kernels=1, input_size=None, padding=0, stride=1, activation=None, path=None):
        LeNetLayer.__init__(self, id, num_kernels, kernel_dims, input_size, padding, stride, activation, path)
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
            delts = self.activation.backward_prop(delta)
        max_vals_map = np.repeat(np.repeat(self.outputs, self.stride, axis=2), self.stride, axis=3)
        delta_map = np.repeat(np.repeat(delta, self.stride, axis=2), self.stride, axis=3)
        delta_x = (max_vals_map == self.inputs) * delta_map
        return(delta_x)

class FullyConnected(LeNetLayer):
    def __init__(self, id, input_size, output_size, num_kernels=1, padding=0, stride=1, activation=None, path=None):
        kernel_dims = (input_size[0], output_size)
        LeNetLayer.__init__(self, id, num_kernels, kernel_dims, input_size, padding, stride, activation, path)
        self.output_size = (output_size, 1, 1)
        if(path):
            self.kernel = self._load_path(path)
        else:
            self.kernel = self._gen_kernels((self.num_kernels, self.input_size[0], self.kernel_dims[0], self.kernel_dims[1]), self.num_kernels)


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
        # print(delta_w[0])
        # print(delta_b)
        self.kernel['weights'] -= np.squeeze(delta_w)*lr
        self.kernel['bias'] -= np.squeeze(delta_b)*lr
        # chomu
        return(delta_x.reshape(delta_x.shape + (1,1)))
