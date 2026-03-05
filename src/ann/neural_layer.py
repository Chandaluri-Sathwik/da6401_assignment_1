"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np

from .activations import get_activation_function

class NeuralLayer:
    def __init__(self,input_dim,output_dim,activation='relu',weight_init='xavier'):
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.activation=activation
        
        if weight_init=='random':
            self.W=np.random.randn(input_dim,output_dim)*0.01
        elif weight_init=='xavier':
            limit=np.sqrt(6/(input_dim+output_dim))
            self.W=np.random.uniform(-limit,limit,(input_dim,output_dim))
        else:
            raise ValueError("Unsupported weight initialization method")
        self.b=np.zeros((1,output_dim))

        if activation:
            self.activation,self.activation_derivative=get_activation_function(activation)
        else:   
            self.activation=lambda x:x
            self.activation_derivative=lambda x:np.ones_like(x)
        
        self.X=None
        self.Z=None
        self.A=None

        self.grad_W=None
        self.grad_b=None
    def forward(self,X):
        self.X=X
        self.Z=np.dot(X,self.W)+self.b
        if self.activation:
            self.A=self.activation(self.Z)
        else:
            self.A=self.Z
        return self.A
    def backward(self,dA):
        if self.activation:
            dZ=dA*self.activation_derivative(self.Z)
        else:
            dZ=dA
        self.grad_W=np.dot(self.X.T,dZ)
        self.grad_b=np.sum(dZ,axis=0,keepdims=True)
        dX=np.dot(dZ,self.W.T)
        return dX
    