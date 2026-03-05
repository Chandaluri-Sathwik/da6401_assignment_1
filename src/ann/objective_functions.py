"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np
from .activations import softmax

def one_hot(y,num_classes=10):
    y=np.asarray(y,dtype=np.int64).reshape(-1)
    y_oh=np.zeros((y.shape[0],num_classes))
    y_oh[np.arange(y.shape[0]),y]=1
    return y_oh
def cross_entropy_loss(logits,y_true):
    y_true=np.asarray(y_true,dtype=np.int64).reshape(-1)
    probabilities=softmax(logits)
    probabilities=np.clip(probabilities,1e-15,1)
    return -np.mean(np.log(probabilities[np.arange(len(y_true)),y_true]))

def cross_entropy_derivative(logits, y_true):
    y_true=np.asarray(y_true,dtype=np.int64).reshape(-1)
    batch_size=logits.shape[0]

    probabilities=softmax(logits)
    probabilities[np.arange(batch_size),y_true]-=1.0
    return probabilities/batch_size

def mse_loss(logits, y_true):
    probabilities=softmax(logits)
    y_oh=one_hot(y_true, num_classes=logits.shape[1])
    return np.mean(np.sum((probabilities-y_oh)**2, axis=1))

def mse_derivative(logits, y_true):
    batch_size=logits.shape[0]
    probabilities=softmax(logits)
    y_oh=one_hot(y_true, num_classes=logits.shape[1])
    return 2*(probabilities-y_oh)/batch_size

def compute_loss(logits,y_true,loss_type='cross_entropy'):
    if loss_type=='cross_entropy':
        return cross_entropy_loss(logits,y_true)
    elif loss_type=='mse':
        return mse_loss(logits,y_true)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
def compute_loss_derivative(logits,y_true,loss_type='cross_entropy'):
    if loss_type=='cross_entropy':
        return cross_entropy_derivative(logits,y_true)
    elif loss_type=='mse':
        return mse_derivative(logits,y_true)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")