from typing import List, Tuple, Dict, Union
import numpy as np

def dna_to_binary(sequence: str) -> str:
    """
    Convert a DNA sequence to binary representation.
    
    Args:
        sequence: A string containing only A, T, G, C
        
    Returns:
        str: Binary string where A=00, T=01, G=10, C=11
    """
    mapping = {'A': '00', 'T': '01', 'G': '10', 'C': '11'}
    binary = []
    for base in sequence.upper():
        if base in mapping:
            binary.append(mapping[base])
    return ''.join(binary)

def binary_to_dna(binary: str) -> str:
    """
    Convert a binary string back to DNA sequence.
    
    Args:
        binary: Binary string (length must be even)
        
    Returns:
        str: DNA sequence
    """
    reverse_mapping = {'00': 'A', '01': 'T', '10': 'G', '11': 'C'}
    dna = []
    for i in range(0, len(binary), 2):
        if i+1 < len(binary):
            chunk = binary[i] + binary[i+1]
            if chunk in reverse_mapping:
                dna.append(reverse_mapping[chunk])
    return ''.join(dna)

def encode_sequences(sequences: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode a list of DNA sequences into binary feature vectors.
    
    Args:
        sequences: List of DNA sequences
        
    Returns:
        tuple: (features, max_length) where features is a 2D numpy array
               and max_length is the length of the longest binary encoding
    """
    binary_sequences = [dna_to_binary(seq) for seq in sequences]
    max_length = max(len(seq) for seq in binary_sequences) if binary_sequences else 0
    
    # Pad sequences to have the same length
    features = np.zeros((len(sequences), max_length), dtype=int)
    for i, seq in enumerate(binary_sequences):
        for j, bit in enumerate(seq):
            if j < max_length:
                features[i, j] = int(bit)
    
    return features, max_length

def prepare_dataset(sequences: List[str], labels: List[int], max_length: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare dataset for machine learning.
    
    Args:
        sequences: List of DNA sequences
        labels: List of corresponding labels (0 for reference, 1 for mutated)
        max_length: Maximum sequence length (if None, use the longest sequence)
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the label vector
    """
    binary_seqs = [dna_to_binary(seq) for seq in sequences]
    
    if max_length is None:
        max_length = max(len(seq) for seq in binary_seqs) if binary_seqs else 0
    
    X = np.zeros((len(sequences), max_length), dtype=int)
    y = np.array(labels, dtype=int)
    
    for i, seq in enumerate(binary_seqs):
        for j in range(min(len(seq), max_length)):
            X[i, j] = int(seq[j])
    
    return X, y
