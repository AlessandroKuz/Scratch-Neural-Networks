import numpy as np


def sigmoid(z: np.array) -> np.array:
    return 1 / (1 + np.exp(-z))

def tanh(z: np.array) -> np.array:
    return np.tanh(z)

def relu(z: np.array) -> np.array:
    return np.maximum(0, z)

def softmax(z: np.array) -> np.array:
    e_i = np.exp(z - np.max(z, axis=1, keepdims=True))  # "normalizing" for numerical stability i.e. no numerical overflow
    return e_i / np.sum(e_i, axis=1, keepdims=True)
