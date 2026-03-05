"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""
import numpy as np

def softmax(z):
    z_max=np.max(z,axis=1,keepdims=True)
    exp_z=np.exp(z-z_max)
    return exp_z/np.sum(exp_z,axis=1,keepdims=True)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)
def relu(z):
    return np.maximum(0, z)
def relu_derivative(z):
    return (z > 0).astype(float)
def tanh(z):
    return np.tanh(z)
def tanh_derivative(z):
    return 1 - np.tanh(z)**2
def get_activation_function(name):
    if name == 'sigmoid':
        return sigmoid, sigmoid_derivative
    elif name == 'relu':
        return relu, relu_derivative
    elif name == 'tanh':
        return tanh, tanh_derivative
    else:
        raise ValueError(f"Unsupported activation function: {name}")