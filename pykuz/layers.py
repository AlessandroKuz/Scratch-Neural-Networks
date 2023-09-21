import numpy as np


class Dense:
    def __init__(self, n_inputs: int, n_neurons: int):
       self.weigths = np.random.randn(n_inputs, n_neurons)
       self.biases = np.zeros((1, n_neurons))
       self.output = None

    def forward(self, inputs: np.array) -> np.array:
        self.output = np.dot(inputs, self.weigths) + self.biases
        
        return self.output

    def backward(self):
        ...
