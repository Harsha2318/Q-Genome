import os
import sys
import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from quantum.dna_classifier_qiskit import QuantumDNAClassifier
except ImportError as e:
    print(f"Error importing QuantumDNAClassifier: {e}")
    print("Please make sure all required dependencies are installed.")
    sys.exit(1)

def generate_test_data(n_samples=1000, n_features=8, random_state=42):
    """Generate synthetic DNA sequence data for testing."""
    # Generate random binary classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=0,
        n_classes=2,
        random_state=random_state
    )
    
    # Scale features to [0, 1] range
    X = (X - X.min()) / (X.max() - X.min())
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

def test_quantum_classifier():
    """
    Test the quantum DNA classifier implementation with BRCA1 DNA sequences.
    """
    print("=== Testing Quantum DNA Classifier ===")
    
    # Load BRCA1 DNA sequence data
    print("\nLoading BRCA1 DNA sequence data...")
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    
    # Load the BRCA1 sequences data
    data = np.load('data/brca1_sequences.npy.npz')
    sequences = data['sequences']
    y = data['labels']
    
    # Convert DNA sequences to numerical features
    def one_hot_encode(sequence):
        """
        Convert DNA sequence to one-hot encoding.
        """
        encoding = {
            'A': [1, 0, 0, 0],
            'C': [0, 1, 0, 0],
            'G': [0, 0, 1, 0],
            'T': [0, 0, 0, 1],
            'N': [0, 0, 0, 0]  # For unknown bases
        }
        
        # Convert sequence to one-hot encoding
        encoded = []
        for base in sequence:
            encoded.extend(encoding.get(base, [0, 0, 0, 0]))
        return np.array(encoded)
    
    # Convert all sequences
    X = np.array([one_hot_encode(seq) for seq in sequences])
    
    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize quantum classifier
    print("\nInitializing quantum classifier...")
    classifier = QuantumDNAClassifier(
        n_qubits=4,  # Number of qubits
        n_layers=2,  # Number of quantum layers
        learning_rate=0.01,
        epochs=10  # Reduced epochs for testing
    )
    
    # Train the classifier
    print("\nTraining quantum classifier...")
    history = classifier.fit(X_train, y_train, X_test, y_test)
    
    print("\nTraining complete!")
    print(f"Final training accuracy: {history['train_acc'][-1]*100:.2f}%")
    if 'val_acc' in history and history['val_acc']:
        print(f"Final validation accuracy: {history['val_acc'][-1]*100:.2f}%")
    
    # Make predictions on test set
    y_pred = classifier.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {test_acc*100:.2f}%")
    
    return test_acc > 0.7  # Expect at least 70% accuracy on test set

if __name__ == "__main__":
    success = test_quantum_classifier()
    if success:
        print("\n✅ Test passed! The quantum classifier is working correctly.")
    else:
        print("\n❌ Test failed! The quantum classifier did not achieve the expected accuracy.")
