"""
Inference Script
Evaluate trained models on test sets
"""

import argparse

def parse_arguments():
    """
    Parse command-line arguments for inference.
    
    TODO: Implement argparse with:
    - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')
    parser.add_argument('-m','--model_path', type=str, help='Path to saved model weights (relative path)',default='models/')
    parser.add_argument('-d','--dataset', choices=['mnist', 'fashion_mnist'], help='Dataset to evaluate on',default='mnist')
    parser.add_argument('-b','--batch_size', type=int, help='Batch size for inference',default=64)
    parser.add_argument('-nhl','--num_layers', type=int,  help='List of hidden layer sizes',default=2)
    parser.add_argument('-sz','--hidden_size', type=int,nargs='+', help='Number of neurons in hidden layers',default=[128, 64])  
    
    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model from disk.
    """
    pass


def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    pass


def main():
    """
    Main inference function.

    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()
    
    print("Evaluation complete!")


if __name__ == '__main__':
    main()
