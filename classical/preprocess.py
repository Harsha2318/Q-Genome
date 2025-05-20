from Bio import Entrez, SeqIO
from typing import Optional

def fetch_gene_sequence(genbank_id: str = "NM_007294", email: str = "your.email@example.com") -> str:
    """
    Fetch a gene sequence from NCBI's GenBank.
    
    Args:
        genbank_id: The GenBank accession number (default: BRCA1 gene)
        email: Your email address (required by NCBI)
        
    Returns:
        str: The DNA sequence as a string
    """
    try:
        Entrez.email = email
        handle = Entrez.efetch(db="nucleotide", id=genbank_id, rettype="fasta", retmode="text")
        record = SeqIO.read(handle, "fasta")
        return str(record.seq)
    except Exception as e:
        print(f"Error fetching sequence: {e}")
        # Return a sample sequence in case of failure
        return "ATGC" * 100  # Sample sequence for testing

def introduce_mutation(sequence: str, position: int, mutation: str) -> str:
    """
    Introduce a mutation at a specific position in the DNA sequence.
    
    Args:
        sequence: The original DNA sequence
        position: Position to introduce mutation (0-based)
        mutation: The new base to insert
        
    Returns:
        str: Mutated sequence
    """
    if position < 0 or position >= len(sequence):
        raise ValueError("Position out of range")
    return sequence[:position] + mutation + sequence[position+1:]

def generate_mutated_sequences(reference: str, num_mutations: int = 10) -> list:
    """
    Generate mutated versions of the reference sequence.
    
    Args:
        reference: The reference DNA sequence
        num_mutations: Number of mutated sequences to generate
        
    Returns:
        list: List of (sequence, label) tuples where label is 0 for reference, 1 for mutated
    """
    import random
    
    sequences = [(reference, 0)]  # Reference sequence with label 0
    
    for _ in range(num_mutations):
        # Choose a random position and base to mutate to
        pos = random.randint(0, len(reference) - 1)
        current_base = reference[pos]
        # Choose a different base
        possible_bases = {'A', 'T', 'G', 'C'} - {current_base}
        new_base = random.choice(list(possible_bases))
        # Create mutated sequence
        mutated = reference[:pos] + new_base + reference[pos+1:]
        sequences.append((mutated, 1))  # Mutated sequences with label 1
    
    return sequences
