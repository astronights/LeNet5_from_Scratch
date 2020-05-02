import numpy as np

class ReLU():
    def __init__(self):
        self.X = None

    def forward_prop(self, data):
        self.X = data
        return(np.maximum(data, 0))

    def backward_prop(self, delta):
        return(delta * (self.X >= 0))

class Sigmoid():
    def __init__(self):
        self.out = None

    def forward_prop(self, data):
        self.out = 1.0/(1.0 + np.exp(-data))
        return(self.out)

    def backward_prop(self, delta):
        return(delta * (self.out) * (1 - self.out))


class tanh():
    def __init__(self):
        self.X = None

    def forward_prop(self, data):
        self.X = data
        return((np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z)))

    def backward_prop(self, delta):
        return(delta * (1 - np.power(tanh(z), 2)))
