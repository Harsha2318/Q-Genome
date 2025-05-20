import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from classical.preprocess import fetch_gene_sequence, introduce_mutation, generate_mutated_sequences
from encoding.binary_encoder import dna_to_binary, prepare_dataset
from quantum.dna_classifier_qiskit import QuantumDNAClassifier

def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)

def load_or_generate_data(use_cached: bool = True, num_mutations: int = 50) -> tuple:
    """
    Load or generate DNA sequence data.
    
    Args:
        use_cached: Whether to use cached data if available
        num_mutations: Number of mutated sequences to generate if not using cached data
        
    Returns:
        tuple: (sequences, labels) where sequences is a list of DNA sequences
               and labels is a list of 0s (reference) and 1s (mutated)
    """
    data_file = 'data/brca1_sequences.npy'
    
    if use_cached and os.path.exists(data_file):
        print("Loading cached data...")
        data = np.load(data_file, allow_pickle=True)
        sequences = data['sequences']
        labels = data['labels']
    else:
        print("Generating new data...")
        # Fetch BRCA1 gene sequence
        print("Fetching BRCA1 gene sequence from NCBI...")
        try:
            reference_sequence = fetch_gene_sequence()
            print(f"Fetched sequence with {len(reference_sequence)} bases")
        except Exception as e:
            print(f"Error fetching sequence: {e}")
            print("Using a sample sequence for demonstration")
            reference_sequence = "ATGC" * 100  # Fallback sequence
        
        # Generate mutated sequences
        print(f"Generating {num_mutations} mutated sequences...")
        sequences_with_labels = generate_mutated_sequences(
            reference_sequence[:100],  # Use first 100 bases for demo
            num_mutations=num_mutations
        )
        
        sequences = [seq for seq, _ in sequences_with_labels]
        labels = [label for _, label in sequences_with_labels]
        
        # Save for future use
        np.savez(data_file, sequences=sequences, labels=labels)
    
    print(f"Loaded {len(sequences)} sequences ({sum(labels)} mutated, {len(labels)-sum(labels)} reference)")
    return sequences, np.array(labels)

def train_and_evaluate(sequences: list, labels: np.ndarray, args, max_sequence_length: int = 100) -> dict:
    """
    Train and evaluate the quantum-classical hybrid model.
    
    Args:
        sequences: List of DNA sequences
        labels: Array of binary labels (0 for reference, 1 for mutated)
        max_sequence_length: Maximum length of sequences to consider
        
    Returns:
        dict: Training history and evaluation metrics
    """
    # Prepare dataset
    X, y = prepare_dataset(sequences, labels, max_length=max_sequence_length*2)  # *2 because each base becomes 2 bits
    
    # Limit the number of qubits for demonstration
    max_qubits = 8  # Adjust based on your system's capabilities
    if X.shape[1] > max_qubits:
        print(f"Reducing feature dimension from {X.shape[1]} to {max_qubits} to fit on quantum hardware")
        X = X[:, :max_qubits]
    
    # Initialize and train the classifier
    print("\nInitializing Quantum-DNA Classifier...")
    print(f"Input features: {X.shape[1]} (will be embedded into quantum circuit)")
    print(f"Training for {args.epochs} epochs...")
    
    classifier = QuantumDNAClassifier(
        n_qubits=min(X.shape[1], 8),  # Limit to 8 qubits for practical quantum simulation
        n_layers=3,  # Number of quantum circuit layers
        learning_rate=0.001,
        epochs=args.epochs
    )
    
    print("Training model...")
    # Split data into train and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")
    history = classifier.fit(X_train, y_train, X_val, y_val)
    
    # Evaluate on test data
    print("\nEvaluating model on test set...")
    val_loss, val_accuracy = classifier.evaluate(X_val, y_val)
    
    # Show detailed metrics
    print("\n" + "="*50)
    print("Model Evaluation Results")
    print("="*50)
    print(f"Final Training Accuracy: {history['train_acc'][-1]*100:.2f}%")
    if 'test_acc' in history and len(history['test_acc']) > 0:
        print(f"Final Validation Accuracy: {history['test_acc'][-1]*100:.2f}%")
    print(f"Training Time: {history.get('training_time', 0):.2f} seconds")
    print(f"Final Validation Loss: {val_loss:.4f}")
    print(f"Final Validation Accuracy: {val_accuracy*100:.2f}%")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation accuracy
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Training Accuracy')
    if 'test_acc' in history and len(history['test_acc']) > 0:
        plt.plot(history['test_acc'], label='Test Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Training Loss')
    if 'test_loss' in history and len(history['test_loss']) > 0:
        plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Save the plot
    plot_path = 'results/training_history.png'
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Training plots saved to {plot_path}")
    
    # Save metrics to file
    metrics_path = 'results/metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write('Training Metrics:\n')
        f.write('='*50 + '\n')
        if 'test_acc' in history and len(history['test_acc']) > 0:
            f.write(f"Final Training Accuracy: {history['train_acc'][-1]*100:.2f}%\n")
            f.write(f"Final Test Accuracy: {history['test_acc'][-1]*100:.2f}%\n")
            f.write(f"Final Training Loss: {history['train_loss'][-1]:.4f}\n")
            f.write(f"Final Test Loss: {history['test_loss'][-1]:.4f}\n")
        else:
            f.write(f"Final Training Accuracy: {history['train_acc'][-1]*100:.2f}%\n")
            f.write(f"Final Training Loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"Training Time: {history.get('training_time', 0):.2f} seconds\n")
    
    print(f"Metrics saved to {metrics_path}")
    
    return {
        'history': history,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'plot_path': plot_path,
        'metrics_path': metrics_path
    }
def main():
    """Main function to run the DNA mutation detection pipeline."""
    import time
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Quantum-Classical Hybrid DNA Mutation Detection')
    parser.add_argument('--num-mutations', type=int, default=20,
                        help='Number of mutated sequences to generate')
    parser.add_argument('--max-sequence-length', type=int, default=100,
                        help='Maximum length of DNA sequences to process')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--use-cached', action='store_true',
                        help='Use cached data if available')
    
    args = parser.parse_args()
    
    try:
        print("\n" + "="*50)
        print("Q-Genome: Hybrid Quantum-Classical DNA Mutation Detection")
        print("="*50 + "\n")
        
        # Setup directories
        setup_directories()
        
        # Load or generate data
        print("Step 1: Loading/Generating DNA sequence data")
        print("-"*50)
        sequences, labels = load_or_generate_data(
            use_cached=args.use_cached,
            num_mutations=args.num_mutations * 5  # Generate more mutations for better generalization
        )
        
        # Show some examples
        print("\nExample sequences:")
        for i, seq in enumerate(sequences[:3]):
            print(f"{i+1}. {seq[:20]}... (Label: {'Mutated' if labels[i] == 1 else 'Reference'})")
        
        # Train and evaluate the model
        print("\nStep 2: Training Quantum-Classical Model")
        print("-"*50)
        start_time = time.time()
        results = train_and_evaluate(
            sequences,
            labels,
            args,
            max_sequence_length=args.max_sequence_length
        )
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Final Validation Accuracy: {results['val_accuracy']*100:.2f}%")
        print(f"Final Validation Loss: {results['val_loss']:.4f}")
        print("\n" + "="*50)
        print("Analysis Complete!")
        print("="*50)
    except Exception as e:
        print(f"An error occurred: {e}")

print("\nNext steps:")
print("1. Check 'results/training_history.png' for training curves")
print("2. Check 'results/metrics.txt' for detailed metrics")
print("3. Try different hyperparameters using command line arguments")
print("4. Experiment with different sequence lengths and mutation types")
    

if __name__ == "__main__":
    main()
