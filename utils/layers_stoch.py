import math
from scipy import signal, ndimage
from skimage import measure
import numpy as np


class LeNetLayer():
    def __init__(self, id, num_kernels, kernel_dims, input_size, padding, stride, activation, batch_size):
        self.id = id
        self.input_size = input_size
        self.inputs = np.ndarray((batch_size,)+input_size)
        self.num_kernels = num_kernels
        self.kernel_dims = kernel_dims
        self.padding = padding
        self.stride = stride
        self.activation = activation

    def _calc_output_size(self, N, F, p, stride, depth):
        try:
            dim = int((N[1]-F[1]+(2*p))/stride + 1)
        except:
            print("Missing values in sizes...")
            return(None)
        return((dim, dim, depth))

    def print_layer(self, name):
        print(self.id, ": ", name)
        print("Input Size: ", self.input_size)
        print("Number of kernels: ", self.num_kernels)
        print("Kernal Dimensions: ", self.kernel_dims)
        print("Stride: ", self.stride)
        print("Padding: ", self.padding)
        if(self.activation):
            print("Activation Function: ", self.activation)
        else:
            print("Activation Function:  None")
        print("Output Size: ", self.output_size)
        print()



class Convolution(LeNetLayer):
    def __init__(self, id, num_kernels, kernel_dims, input_size=None, padding=0, stride=1, activation=None, batch_size=256):
        LeNetLayer.__init__(self, id, num_kernels, kernel_dims, input_size, padding, stride, activation, batch_size)
        self.output_size = self._calc_output_size(self.input_size, self.kernel_dims,
                                                  self.padding, self.stride, self.num_kernels)
        self.kernels = {"weights": np.ndarray(kernel_dims + (input_size[2],self.num_kernels)), "bias": np.ndarray(num_kernels)}
        self._gen_kernels(self.input_size[2])

    def _gen_kernels(self,depth):
        # filter_size = (self.kernel_dims[0], self.kernel_dims[1], depth)
        self.kernels['weights'] = np.random.standard_normal(self.kernels['weights'].shape)
        self.kernels['bias'] =  np.random.standard_normal(self.num_kernels)
        # for i in range(self.num_kernels):
        #     filter = {"weights": np.random.standard_normal(filter_size), "bias": np.random.standard_normal()}
        #     self.kernels.append(filter)

    def _do_dot(self, image, k):
        product = (np.multiply(image, self.kernels['weights'][:,:,:,k]))
        return(np.sum(product)+self.kernels['bias'][k])

    def _do_dot_v2(self, image, filter):
        return(signal.convolve2d(image, filter, mode='valid'))

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

    def transf_kernel(self, kernel):
        return(np.rot90(kernel, 2, (0,1)))

    def forward_prop_og(self, image, rec):
        self.inputs[rec] = image
        output = np.zeros(self.output_size)
        temp_kernel = self.transf_kernel(self.kernels['weights'])
        # print(output.shape, image.shape, temp_kernel.shape)
        for i in range(output.shape[2]):
            for j in range(image.shape[2]):
                # print((self._do_dot_v2(image[:,:,j], temp_kernel[:,:,j,i])).shape)
                output[:,:,i] += self._do_dot_v2(image[:,:,j], temp_kernel[:,:,j,i])
            output[:,:,i] += self.kernels["bias"][i]

    def forward_prop_og_old(self, image, rec):
        self.inputs[rec] = image
        output=np.zeros(self.output_size)
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                for k in range(output.shape[2]):
                    output[i,j,k] = self._do_dot(image[i:i+self.kernel_dims[0],j:j+self.kernel_dims[0],:], k)
        return(output)

    def backward_prop_og(self, delta, rec, lr):
        delta_x = np.zeros(self.input_size)
        delta_w = np.zeros(self.kernel_dims + (self.input_size[2], self.num_kernels))
        delta_b = np.zeros(self.num_kernels)

        for i in range(self.num_kernels):
            for j in range(self.input_size[0]-self.kernel_dims[0] + 1):
                for k in range(self.input_size[1]-self.kernel_dims[1] + 1):
                    delta_x[j:j+self.kernel_dims[0], k:k+self.kernel_dims[0], :] += delta[j, k, i]*self.kernels['weights'][:,:,:,i]
                    delta_w[:,:,:,i] += delta[j, k, i] * self.inputs[rec, j:j+self.kernel_dims[0], k:k+self.kernel_dims[0], :]
        delta_b = np.sum(delta, axis=(0,1))
        self.kernels['weights'] -= (delta_w)*lr
        self.kernels['bias'] -= (delta_b)*lr
        return(delta_x)


class MaxPooling(LeNetLayer):
    def __init__(self, id, kernel_dims, num_kernels=1, input_size=None, padding=0, stride=1, activation=None, batch_size=256):
        LeNetLayer.__init__(self, id, num_kernels, kernel_dims, input_size, padding, stride, activation, batch_size)
        self.output_size = self._calc_output_size(self.input_size, self.kernel_dims,
                                                  self.padding, self.stride, self.input_size[2])


    def forward_prop(self, image):
        res = measure.block_reduce(image, (2,2,1), np.max)
        if(self.activation == 'RelU'):
            return(np.maximum(res, 0))
        return(res)

    def forward_prop_og(self, image, rec):
        self.inputs[rec] = image
        output=np.zeros(self.output_size)
        for i in range(output.shape[0]):
            new_i = i*self.stride
            for j in range(output.shape[1]):
                new_j = j*self.stride
                for k in range(output.shape[2]):
                    output[i,j,k] = np.max(image[new_i:new_i+self.kernel_dims[0],new_j:new_j+self.kernel_dims[0], k])
        output = np.clip(output, a_min=0.0, a_max=None)
        self.outputs = output
        return(output)

    def backward_prop_og(self, delta, rec, lr):
        delta_x = np.zeros(self.input_size)
        max_vals_map = np.repeat(np.repeat(self.outputs, self.stride, axis=0), self.stride, axis=1)
        delta_map = np.repeat(np.repeat(delta, self.stride, axis=0), self.stride, axis=1)
        delta_x = (max_vals_map == self.inputs[rec]) * delta_map
        return(delta_x)

class FullyConnected(LeNetLayer):
    def __init__(self, id, input_size, output_size, num_kernels=1, padding=0, stride=1, activation=None, batch_size=256):
        kernel_dims = (input_size[2], output_size)
        LeNetLayer.__init__(self, id, num_kernels, kernel_dims, input_size, padding, stride, activation, batch_size)
        self.output_size = (1, 1, output_size)
        self._gen_kernels()

    def _gen_kernels(self):
            self.kernels = {"weights": np.random.standard_normal(self.kernel_dims), "bias": np.random.standard_normal(self.output_size[2])}

    def forward_prop(self, image):
        res = np.dot(np.squeeze(image), self.kernels['weights']) + self.kernels[0]['bias']
        return(res)

    def forward_prop_og(self, image, rec):
        # print(image.shape)
        self.inputs[rec] = image
        res = np.dot(np.squeeze(image), self.kernels['weights']) + self.kernels['bias']
        return(res)

    def backward_prop(self, delta, rec, lr):
        delta_x = np.dot(delta, self.kernels['weights'].T)
        X, d = self.transf_data(self.inputs[rec], delta)
        delta_w = np.dot(X, d)
        delta_b = np.sum(delta, axis=0)

        self.kernels['weights'] -= delta_w*lr
        self.kernels['bias'] -= delta_b*lr

        return(delta_x)

    def backward_prop_og(self, delta, rec, lr):
        delta_x = np.dot(delta, self.kernels['weights'].T)
        delta_w = np.dot(self.inputs[rec].T ,delta)
        delta_b = np.sum(delta, axis=0)

        self.kernels['weights'] -= np.squeeze(delta_w)*lr
        self.kernels['bias'] -= np.squeeze(delta_b)*lr
        return(delta_x)
