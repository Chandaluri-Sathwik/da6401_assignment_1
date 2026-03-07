"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference on test set")
    parser.add_argument("-m", "--model_path", type=str, default="./best_model.npy")
    parser.add_argument("-c", "--config_path", type=str, default="./best_config.json")
    parser.add_argument('-d','--dataset', choices=['mnist', 'fashion_mnist'], help='Dataset to use for training',default='mnist')
    parser.add_argument('-e','--epochs', type=int, help='Number of training epochs',default=10)
    parser.add_argument('-b','--batch_size', type=int, help='Mini-batch size',default=64)
    parser.add_argument('-l','--loss', choices=['cross_entropy', 'mse'], help='Loss function to use',default='cross_entropy')
    parser.add_argument('-wd','--weight_decay', type=float, help='Weight decay (L2 regularization) factor', default=0.0)    
    parser.add_argument('-lr','--learning_rate', type=float, help='Learning rate for optimizer',default=0.001 )
    parser.add_argument('-o','--optimizer', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], help='Optimizer to use for training',default='adam')
    parser.add_argument('-nhl','--num_layers', type=int,  help='List of hidden layer sizes',default=2)
    parser.add_argument('-sz','--hidden_size', type=int,nargs='+', help='Number of neurons in hidden layers',default=[128, 64])
    parser.add_argument('-a','--activation', choices=['relu', 'sigmoid', 'tanh'], help='Activation function to use',default='relu')
    parser.add_argument('-w_i','--weight_init', choices=['random', 'xavier'], help='Weight initialization method',default='xavier')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(model_path):
    return np.load(model_path, allow_pickle=True).item()


def evaluate_model(model, X_test, y_test):
    out = model.evaluate(X_test, y_test)
    logits = out["logits"]
    preds = np.argmax(logits, axis=1)

    precision = precision_score(y_test, preds, average="macro", zero_division=0)
    recall = recall_score(y_test, preds, average="macro", zero_division=0)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)

    return {
        "logits": logits,
        "loss": float(out["loss"]),
        "accuracy": float(out["accuracy"]),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main():
    args = parse_arguments()

    cfg = load_config(args.config_path)

    # Optional CLI override
    if args.dataset is not None:
        cfg["dataset"] = args.dataset

    # Build namespace expected by NeuralNetwork
    nn_args = argparse.Namespace(**cfg)

    # Ensure required keys exist (fallback defaults)
    if not hasattr(nn_args, "optimizer"):
        nn_args.optimizer = "adam"
    if not hasattr(nn_args, "learning_rate"):
        nn_args.learning_rate = 0.001
    if not hasattr(nn_args, "weight_decay"):
        nn_args.weight_decay = 0.0
    if not hasattr(nn_args, "loss"):
        nn_args.loss = "cross_entropy"
    if not hasattr(nn_args, "activation"):
        nn_args.activation = "relu"
    if not hasattr(nn_args, "weight_init"):
        nn_args.weight_init = "xavier"

    _, _, _, _, X_test, y_test = load_data(nn_args.dataset)

    model = NeuralNetwork(nn_args)
    weights = load_model(args.model_path)
    model.set_weights(weights)

    results = evaluate_model(model, X_test, y_test)

    print(
        f"loss: {results['loss']:.4f} | acc: {results['accuracy']:.4f} | "
        f"precision: {results['precision']:.4f} | recall: {results['recall']:.4f} | f1: {results['f1']:.4f}"
    )
    return results


if __name__ == "__main__":
    main()
