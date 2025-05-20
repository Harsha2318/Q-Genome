from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time

class ClassicalNN(nn.Module):
    """A classical neural network for DNA sequence classification with regularization."""
    def __init__(self, input_size: int, hidden_size: int = 64, dropout: float = 0.3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.model(x)

class QuantumDNAClassifier:
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, learning_rate: float = 0.01, epochs: int = 50):
        """
        Initialize the classical neural network classifier.
        
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
