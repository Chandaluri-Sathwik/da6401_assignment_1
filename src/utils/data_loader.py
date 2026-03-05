"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import mnist, fashion_mnist

def load_data(dataset='mnist',val_size=0.1, random_state=42):
    if dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif dataset == 'fashion_mnist':    
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError("Invalid dataset. Choose 'mnist' or 'fashion_mnist'.")
    # Normalize pixel values to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    # Reshape to (num_samples, 784)
    X_train = X_train.reshape(-1, 28*28)
    X_test = X_test.reshape(-1, 28*28)  

    X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=val_size,random_state=random_state,stratify=y_train,shuffle=True)
    return X_train, y_train, X_val, y_val, X_test, y_test
  
