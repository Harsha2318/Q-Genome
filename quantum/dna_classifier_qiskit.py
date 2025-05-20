from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional, Union, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time

# Pennylane imports
import pennylane as qml
from pennylane import numpy as qnp

# Set random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
np.random.seed(42)
qnp.random.seed(42)

# Initialize PennyLane device with default.qubit
n_qubits = 4
n_layers = 2
dev = qml.device('default.qubit', wires=n_qubits)

class QuantumNN(nn.Module):
    """Hybrid Quantum-Classical Neural Network for DNA sequence classification."""
    
    def __init__(self, 
                 n_qubits: int = 4,
                 n_layers: int = 2,
                 learning_rate: float = 0.01,
                 epochs: int = 100):
        """
        Initialize the Quantum Neural Network.
        
        Args:
            n_qubits: Number of qubits in the quantum circuit
            n_layers: Number of layers in the quantum circuit
            learning_rate: Learning rate for the optimizer
            epochs: Number of training epochs
        """
        super().__init__()
        
        # Initialize PCA for dimensionality reduction
        self.pca = None
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Initialize weights as float32 (3 parameters per qubit per layer)
        self.weights = torch.nn.Parameter(torch.randn(n_layers * n_qubits * 3, dtype=torch.float32))
        
        # Define quantum circuit as a class method
        @qml.qnode(dev)
        def quantum_circuit(inputs, weights):
            # Enhanced feature embedding
            for i in range(n_qubits):
                qml.RX(inputs[i], wires=i)
                qml.RZ(inputs[i], wires=i)
            
            # Enhanced variational layers with more entanglement
            for layer in range(n_layers):
                # Parametrized gates
                for i in range(n_qubits):
                    qml.RY(weights[layer * n_qubits + i], wires=i)
                    qml.RZ(weights[layer * n_qubits + i + n_qubits], wires=i)
                
                # Full entanglement
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        qml.CNOT(wires=[i, j])
                
                # Additional rotation layer
                for i in range(n_qubits):
                    qml.RX(weights[layer * n_qubits + i + 2 * n_qubits], wires=i)
            
            # Measurement
            return qml.expval(qml.PauliZ(0))
        
        # Store the quantum circuit as an instance method
        self.quantum_circuit = quantum_circuit
        
        # Enhanced classical post-processing layers
        self.classical_net = nn.Sequential(
            nn.Linear(1, 32),  # More neurons in first layer
            nn.ReLU(),
            nn.Linear(32, 16),  # Additional hidden layer
            nn.ReLU(),
            nn.Linear(16, 8),  # Another hidden layer
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Loss function
        self.criterion = nn.BCELoss()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Initialize weights as float32 (3 parameters per qubit per layer)
        self.weights = torch.nn.Parameter(torch.randn(n_layers * n_qubits * 3, dtype=torch.float32))
        
        # Define quantum circuit as a class method
        @qml.qnode(dev)
        def quantum_circuit(inputs, weights):
            # Enhanced feature embedding
            for i in range(n_qubits):
                qml.RX(inputs[i], wires=i)
                qml.RZ(inputs[i], wires=i)
            
            # Enhanced variational layers with more entanglement
            for layer in range(n_layers):
                # Parametrized gates
                for i in range(n_qubits):
                    qml.RY(weights[layer * n_qubits + i], wires=i)
                    qml.RZ(weights[layer * n_qubits + i + n_qubits], wires=i)
                
                # Full entanglement
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        qml.CNOT(wires=[i, j])
                
                # Additional rotation layer
                for i in range(n_qubits):
                    qml.RX(weights[layer * n_qubits + i + 2 * n_qubits], wires=i)
            
            # Measurement
            return qml.expval(qml.PauliZ(0))
        
        # Store the quantum circuit as an instance method
        self.quantum_circuit = quantum_circuit
        
        # Enhanced classical post-processing layers
        self.classical_net = nn.Sequential(
            nn.Linear(1, 32),  # More neurons in first layer
            nn.ReLU(),
            nn.Linear(32, 16),  # Additional hidden layer
            nn.ReLU(),
            nn.Linear(16, 8),  # Another hidden layer
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Loss function
        self.criterion = nn.BCELoss()
        
    def _reduce_dimension(self, x: np.ndarray) -> np.ndarray:
        """
        Reduce input dimension using PCA.
        
        Args:
            x: Input features (n_samples, n_features)
            
        Returns:
            Reduced features (n_samples, n_qubits)
        """
        if self.pca is None:
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components=self.n_qubits)
            self.pca.fit(x)
        
        return self.pca.transform(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the hybrid network."""
        # Ensure input is in the correct shape
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # Convert to numpy for dimensionality reduction
        x_np = x.detach().numpy()
        
        # Reduce dimensionality
        x_reduced = self._reduce_dimension(x_np)
        
        # Process each sample in the batch
        outputs = []
        for i in range(x.shape[0]):
            # Forward pass through quantum circuit
            quantum_output = self.quantum_circuit(x_reduced[i], self.weights.detach().numpy())
            outputs.append(quantum_output)
        
        # Convert back to tensor and ensure float32 dtype
        quantum_outputs = torch.tensor(outputs, dtype=torch.float32).unsqueeze(1)
        
        # Pass through classical layers
        final_output = self.classical_net(quantum_outputs)
        
        return final_output
        
    def train_model(self, 
                   X_train: torch.Tensor, 
                   y_train: torch.Tensor,
                   X_test: Optional[torch.Tensor] = None,
                   y_test: Optional[torch.Tensor] = None) -> Dict[str, List[float]]:
        """
        Train the hybrid model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Optional test features
            y_test: Optional test labels
            
        Returns:
            Dictionary containing training history
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            # Training
            self.train()
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self(X_train).squeeze()
            loss = self.criterion(outputs, y_train.float())
            
            # Backward and optimize
            loss.backward()
            self.optimizer.step()
            
            # Calculate training accuracy
            preds = (outputs > 0.5).float()
            train_acc = (preds == y_train).float().mean().item()
            
            # Update history
            history['train_loss'].append(loss.item())
            history['train_acc'].append(train_acc)
            
            # Validation
            if X_test is not None and y_test is not None:
                self.eval()
                with torch.no_grad():
                    test_outputs = self(X_test).squeeze()
                    test_loss = self.criterion(test_outputs, y_test.float())
                    test_preds = (test_outputs > 0.5).float()
                    test_acc = (test_preds == y_test).float().mean().item()
                    
                    history['test_loss'].append(test_loss.item())
                    history['test_acc'].append(test_acc)
                
                print(f"Epoch {epoch+1}/{self.epochs}, "
                      f"Train Loss: {loss.item():.4f}, "
                      f"Test Loss: {test_loss.item():.4f}, "
                      f"Train Acc: {train_acc*100:.2f}%, "
                      f"Test Acc: {test_acc*100:.2f}%")
            else:
                print(f"Epoch {epoch+1}/{self.epochs}, "
                      f"Train Loss: {loss.item():.4f}, "
                      f"Train Acc: {train_acc*100:.2f}%")
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        history['training_time'] = training_time
        
        return history
        
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions on new data."""
        self.eval()
        with torch.no_grad():
            outputs = self(X).squeeze()
            predictions = (outputs > 0.5).float()
        return predictions
        
    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Evaluate the model on test data."""
        self.eval()
        with torch.no_grad():
            outputs = self(X).squeeze()
            loss = self.criterion(outputs, y.float())
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == y).float().mean().item()
            
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'predictions': predictions
        }
    

    
    def train_model(self, 
                   X_train: torch.Tensor, 
                   y_train: torch.Tensor,
                   X_test: Optional[torch.Tensor] = None,
                   y_test: Optional[torch.Tensor] = None) -> Dict[str, List[float]]:
        """
        Train the hybrid model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Optional test features
            y_test: Optional test labels
            
        Returns:
            Dictionary containing training history
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            # Training
            self.train()
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self(X_train).squeeze()
            loss = self.criterion(outputs, y_train.float())
            
            # Backward and optimize
            loss.backward()
            self.optimizer.step()
            
            # Calculate training accuracy
            preds = (outputs > 0.5).float()
            train_acc = (preds == y_train).float().mean().item()
            
            # Update history
            history['train_loss'].append(loss.item())
            history['train_acc'].append(train_acc)
            
            # Validation
            if X_test is not None and y_test is not None:
                self.eval()
                with torch.no_grad():
                    test_outputs = self(X_test).squeeze()
                    test_loss = self.criterion(test_outputs, y_test.float())
                    test_preds = (test_outputs > 0.5).float()
                    test_acc = (test_preds == y_test).float().mean().item()
                    
                    history['test_loss'].append(test_loss.item())
                    history['test_acc'].append(test_acc)
                
                print(f"Epoch {epoch+1}/{self.epochs}, "
                      f"Train Loss: {loss.item():.4f}, "
                      f"Test Loss: {test_loss.item():.4f}, "
                      f"Train Acc: {train_acc*100:.2f}%, "
                      f"Test Acc: {test_acc*100:.2f}%")
            else:
                print(f"Epoch {epoch+1}/{self.epochs}, "
                      f"Train Loss: {loss.item():.4f}, "
                      f"Train Acc: {train_acc*100:.2f}%")
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        history['training_time'] = training_time
        
        return history
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions on new data."""
        self.eval()
        with torch.no_grad():
            outputs = self(X).squeeze()
            predictions = (outputs > 0.5).float()
        return predictions
    
    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Evaluate the model on test data."""
        self.eval()
        with torch.no_grad():
            outputs = self(X).squeeze()
            loss = self.criterion(outputs, y.float())
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == y).float().mean().item()
            
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'predictions': predictions
        }

class QuantumDNAClassifier:
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, learning_rate: float = 0.01, epochs: int = 50):
        """
        Initialize the quantum neural network classifier.
        
        Args:
            n_qubits: Number of qubits to use in the quantum circuit
            n_layers: Number of layers in the quantum circuit
            learning_rate: Learning rate for the optimizer
            epochs: Number of training epochs
        """
        self.n_qubits = min(n_qubits, 4)  # Limit to 4 qubits for simulation efficiency
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the quantum neural network
        self.model = QuantumNN(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            learning_rate=learning_rate,
            epochs=epochs
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> dict:
        """
        Train the quantum neural network.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training history
        """
        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            
            # Train with validation
            history = self.model.train_model(
                X_train_tensor, 
                y_train_tensor,
                X_val_tensor,
                y_val_tensor
            )
        else:
            # Train without validation
            history = self.model.train_model(X_train_tensor, y_train_tensor)
        
        # Update history
        self.history = {
            'train_loss': history['train_loss'],
            'train_acc': history['train_acc'],
            'val_loss': history.get('test_loss', []),
            'val_acc': history.get('test_acc', [])
        }
        
        return self.history
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Evaluate the model on the given data.
        
        Args:
            X: Input features (n_samples, n_features)
            y: True labels (n_samples,)
            
        Returns:
            Tuple of (loss, accuracy)
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        metrics = self.model.evaluate(X_tensor, y_tensor)
        return metrics['loss'], metrics['accuracy']
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Predicted probabilities (n_samples,)
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        predictions = self.model.predict(X_tensor)
        return predictions.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for X.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Array of shape (n_samples, 2) with class probabilities
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor).squeeze()
            probas = torch.sigmoid(outputs).cpu().numpy()
        return np.column_stack((1 - probas, probas))
    
    def save_model(self, path: str):
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'history': self.history
        }, path)
    
    @classmethod
    def load_model(cls, path: str, device: str = None):
        """
        Load a saved model from a file.
        
        Args:
            path: Path to the saved model
            device: Device to load the model on ('cuda' or 'cpu')
            
        Returns:
            Loaded QuantumDNAClassifier instance
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        checkpoint = torch.load(path, map_location=device)
        
        # Create a new instance with the saved parameters
        classifier = cls(
            n_qubits=checkpoint['n_qubits'],
            n_layers=checkpoint['n_layers'],
            learning_rate=checkpoint['learning_rate'],
            epochs=checkpoint['epochs']
        )
        
        # Load the model state
        classifier.model.load_state_dict(checkpoint['model_state_dict'])
        classifier.history = checkpoint['history']
        
        return classifier

class ClassicalNN(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class QuantumDNAClassifierOld:
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, learning_rate: float = 0.01, epochs: int = 50):
        """
        Initialize the quantum neural network classifier.
        
        Args:
            n_qubits: Ignored (kept for compatibility)
            n_layers: Number of hidden layers (not used directly, kept for compatibility)
            learning_rate: Learning rate for the optimizer
            epochs: Number of training epochs
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.scaler = StandardScaler()
        self.model = None
        self.criterion = nn.BCELoss()
        self.optimizer = None
    
    def build_model(self, n_inputs: int) -> nn.Module:
        """
        Build a classical neural network model.
        
        Args:
            n_inputs: Number of input features
            
        Returns:
            nn.Module: PyTorch model
        """
        return ClassicalNN(input_size=n_inputs)
    
    def fit(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> dict:
        """
        Train the hybrid model.
        
        Args:
            X: Input features
            y: Target labels
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            dict: Training history with loss and accuracy
        """
        import time
        start_time = time.time()
        
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=random_state
            )
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
            
            # Build model
            self.model = self.build_model(X.shape[1])
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # Training loop
            history = {
                'train_loss': [], 
                'test_loss': [], 
                'train_acc': [], 
                'test_acc': [],
                'training_time': 0.0
            }
            
            print(f"Starting training for {self.epochs} epochs...")
            
            for epoch in range(self.epochs):
                try:
                    # Training
                    self.model.train()
                    self.optimizer.zero_grad()
                    
                    outputs = self.model(X_train_tensor)
                    loss = self.criterion(outputs, y_train_tensor)
                    loss.backward()
                    self.optimizer.step()
                    
                    # Calculate training accuracy
                    train_preds = (outputs > 0.5).float()
                    train_acc = (train_preds == y_train_tensor).float().mean()
                    
                    # Evaluation
                    self.model.eval()
                    with torch.no_grad():
                        test_outputs = self.model(X_test_tensor)
                        test_loss = self.criterion(test_outputs, y_test_tensor)
                        test_preds = (test_outputs > 0.5).float()
                        test_acc = (test_preds == y_test_tensor).float().mean()
                    
                    # Update history
                    history['train_loss'].append(loss.item())
                    history['test_loss'].append(test_loss.item())
                    history['train_acc'].append(train_acc.item())
                    history['test_acc'].append(test_acc.item())
                    
                    if (epoch + 1) % 1 == 0:  # Print every epoch for better monitoring
                        print(f"Epoch {epoch+1}/{self.epochs}, "
                            f"Train Loss: {loss.item():.4f}, "
                            f"Test Loss: {test_loss.item():.4f}, "
                            f"Train Acc: {train_acc.item()*100:.2f}%, "
                            f"Test Acc: {test_acc.item()*100:.2f}%")
                            
                except Exception as e:
                    print(f"Error during epoch {epoch+1}: {str(e)}")
                    continue
            
            # Calculate total training time
            history['training_time'] = time.time() - start_time
            print(f"\nTraining completed in {history['training_time']:.2f} seconds")
            
            return history
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'train_loss': [], 
                'test_loss': [], 
                'train_acc': [], 
                'test_acc': [],
                'training_time': 0.0,
                'error': str(e)
            }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Predicted probabilities
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
            
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).numpy()
            
        return (predictions > 0.5).astype(int).flatten()
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate the model on test data.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            dict: Dictionary with accuracy and loss
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
            
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            preds = (outputs > 0.5).float()
            accuracy = (preds == y_tensor).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }
