"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from .activations import get_activation_function
from .objective_functions import compute_loss, compute_loss_derivative
from .neural_layer import NeuralLayer
from .optimizers import Optimizer

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):
        self.cli_args=cli_args
        self.loss_function=cli_args.loss
        input_dim=784
        output_dim=10
        if len(cli_args.hidden_size)!=cli_args.num_layers:
            raise ValueError("Number of hidden layers and hidden layer sizes must match")
        hidden_sizes=cli_args.hidden_size
        dims=[input_dim]+hidden_sizes+[output_dim]
        self.layers = []
        for i in range(len(dims)-1):
            act=cli_args.activation if i<len(dims)-2 else None
            layer=NeuralLayer(input_dim=dims[i],output_dim=dims[i+1],activation=act,weight_init=cli_args.weight_init)#didnt put weight decay
            self.layers.append(layer)
        self.optimizer=Optimizer(name=cli_args.optimizer,learning_rate=cli_args.learning_rate)
        self.grad_W = None
        self.grad_b = None
        self.activation, self.activation_derivative = get_activation_function(cli_args.activation)
        
    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns logits (no softmax applied)
        X is shape (b, D_in) and output is shape (b, D_out).
        b is batch size, D_in is input dimension, D_out is output dimension.
        """
        for layer in self.layers:
            X=layer.forward(X)
        return X

    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        Returns two numpy arrays: grad_Ws, grad_bs.
        - `grad_Ws[0]` is gradient for the last (output) layer weights,
          `grad_bs[0]` is gradient for the last layer biases, and so on.
        """
        dA=compute_loss_derivative(y_pred,y_true,self.loss_function)
        grad_W_list = []
        grad_b_list = []

        # Backprop through layers in reverse; collect grads so that index 0 = last layer
        for layer in reversed(self.layers):
            dA=layer.backward(dA)
            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

        # create explicit object arrays to avoid numpy trying to broadcast shapes
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb
        
        return self.grad_W, self.grad_b

    def update_weights(self):
        self.optimizer.step(self.layers)

    def train(self, X_train, y_train, epochs=1, batch_size=32,X_val=None,y_val=None):
        n=X_train.shape[0]
        for epoch in range(epochs):
            perm = np.random.permutation(n)
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]

            total_loss = 0.0
            total_correct = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

                logits = self.forward(X_batch)
                loss = compute_loss(logits, y_batch, self.loss_function)

                self.backward(y_batch, logits)
                self.update_weights()

                total_loss += loss * (end - start)
                preds = np.argmax(logits, axis=1)
                total_correct += np.sum(preds == y_batch)

            epoch_loss = total_loss / n
            epoch_acc = total_correct / n
            print(f"Epoch {epoch + 1}/{epochs} | loss: {epoch_loss:.4f} | acc: {epoch_acc:.4f}")
            if X_val is not None and y_val is not None:
                val_metrics = self.evaluate(X_val, y_val)
                print(f"Validation | loss: {val_metrics['loss']:.4f} | acc: {val_metrics['accuracy']:.4f}")
            

    def evaluate(self, X, y):
        logits = self.forward(X)
        loss = compute_loss(logits, y, self.loss_function)
        preds = np.argmax(logits, axis=1)
        acc = np.mean(preds == y)
        precision = precision_score(y, preds, average="macro", zero_division=0)
        recall = recall_score(y, preds, average="macro", zero_division=0)
        f1 = f1_score(y, preds, average="macro", zero_division=0)
        return {"logits": logits, "loss": float(loss), "accuracy": float(acc), "precision": float(precision), "recall": float(recall), "f1": float(f1)}

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()

