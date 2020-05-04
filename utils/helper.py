import numpy as np

class Helper():
    def __init__():
        pass

    def im2col(image, kernel_dims, stride, padding):
        out_dim = (image.shape[2]+ (2*padding) -kernel_dims[0])//stride + 1
        image_vec = np.zeros(image.shape[:2] + kernel_dims + (out_dim, out_dim))
        for i in range(kernel_dims[0]):
            x = i + out_dim
            for j in range(kernel_dims[1]):
                y = j + out_dim
                image_vec[:,:,i,j, :, :] = image[:,:,i:x, j:y]
        return(image_vec.transpose(0,4,5,1,2,3).reshape(image.shape[0]*out_dim*out_dim, -1))

    def col2im(delta, input_shape, kernel_dims, stride, padding):
        out_dim = (input_shape[2]+ (2*padding) - kernel_dims[0])//stride + 1
        delta = delta.reshape((input_shape[0], out_dim, out_dim, input_shape[1]) + kernel_dims).transpose(0,3,4,5,1,2)
        delta_vec = np.zeros((input_shape[:2]) + (input_shape[2] + 2*padding + stride - 1, input_shape[3] + 2*padding + stride - 1))
        for i in range(kernel_dims[0]):
            x = i + out_dim
            for j in range(kernel_dims[1]):
                y = j + out_dim
                delta_vec[:, :, i:x, j:y] += delta[:,:,i,j,:,:]

        return(delta_vec[:,:,padding:padding+input_shape[2], padding:padding+input_shape[3]])
