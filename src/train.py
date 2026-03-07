"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import json
import os
import numpy as np
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data

def parse_arguments():

    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument('-d','--dataset', choices=['mnist', 'fashion_mnist'], help='Dataset to use for training',default='mnist')
    parser.add_argument('-e','--epochs', type=int, help='Number of training epochs',default=10)
    parser.add_argument('-b','--batch_size', type=int, help='Mini-batch size',default=64)
    parser.add_argument('-l','--loss', choices=['cross_entropy', 'mse'], help='Loss function to use',default='cross_entropy')
    parser.add_argument('-wd','--weight_decay', type=float, help='Weight decay (L2 regularization) factor', default=0.0)
    parser.add_argument('-lr','--learning_rate', type=float, help='Learning rate for optimizer',default=0.0005 )
    parser.add_argument('-o','--optimizer', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], help='Optimizer to use for training',default='rmsprop')
    parser.add_argument('-nhl','--num_layers', type=int,  help='List of hidden layer sizes',default=3)
    parser.add_argument('-sz','--hidden_size', type=int,nargs='+', help='Number of neurons in hidden layers',default=[128,128, 64])
    parser.add_argument('-a','--activation', choices=['relu', 'sigmoid', 'tanh'], help='Activation function to use',default='relu')
    parser.add_argument('-w_i','--weight_init', choices=['random', 'xavier'], help='Weight initialization method',default='xavier')
    parser.add_argument('-w_p','--wandb_project', type=str, help='W&B project name',default='da6401_assignment_1')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('-m','--model_save_path', type=str, help='Path to save trained model (relative path)',default='./best_model.npy')
    
    return parser.parse_args()

def _save_config(args, config_path):
    config = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "loss": args.loss,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "activation": args.activation,
        "weight_init": args.weight_init,
        "wandb_project": args.wandb_project,
        "use_wandb": bool(args.use_wandb),
        "model_save_path": args.model_save_path,
        "weight_decay": args.weight_decay,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    
def main():
    """
    Main training function.
    """
    args = parse_arguments()

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.dataset)

    neural_network = NeuralNetwork(args)
    run=None
    if args.use_wandb:
        try:
            import wandb  # Lazy import to avoid autograder/network side effects by default
            run=wandb.init(project=args.wandb_project,config=vars(args))
        except ImportError:
            print("wandb not found, proceeding without it. Install with `pip install wandb` for logging and visualization.")
            run=None
        except Exception as e:
            print(f"Failed to initialize W&B run: {e}. Proceeding without logging.")
            run=None
    
    best_f1=-1.0
    best_weights=None
    for epoch in range(args.epochs):
        neural_network.train(X_train,y_train,epochs=1,batch_size=args.batch_size,X_val=X_val,y_val=y_val)
        val_metrics=neural_network.evaluate(X_val,y_val)
        test_metrics = None
        if run is not None:
            test_metrics = neural_network.evaluate(X_test, y_test)
        grad_norm_layer1 = None
        if len(neural_network.layers) > 0 and neural_network.layers[0].grad_W is not None:
            grad_norm_layer1 = float(np.linalg.norm(neural_network.layers[0].grad_W))   
        # print(f"Epoch {epoch+1}/{args.epochs} - Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}, Test Loss: {test_metrics['loss']:.4f}, Test Acc: {test_metrics['accuracy']:.4f}, Test F1: {test_metrics['f1']:.4f}")
        if run is not None:
            wandb.log({
                "epoch": epoch + 1,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "grad_norm_layer1": grad_norm_layer1,
            })
            if test_metrics is not None:
                wandb.log({
                    "test_loss": test_metrics["loss"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_precision": test_metrics["precision"],
                    "test_recall": test_metrics["recall"],
                    "test_f1": test_metrics["f1"],
                })
        if val_metrics["f1"]>best_f1:
            best_f1=val_metrics["f1"]
            best_weights=neural_network.get_weights()
         

    print("Training complete!")

    # Save weights + config
    model_path = args.model_save_path
    if best_weights is None:
        best_weights=neural_network.get_weights()
    np.save(model_path, best_weights, allow_pickle=True)

    config_path = os.path.join(os.path.dirname(model_path), "best_config.json")
    _save_config(args, config_path)

    print(f"Saved model to: {model_path}")
    print(f"Saved config to: {config_path}")
    print(f"Best val F1: {best_f1:.4f}")
    if run is not None:
        run.finish()

if __name__ == '__main__':
    main()
