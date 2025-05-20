"""
Test script to verify that all required packages are installed correctly.
"""
import sys

def check_imports():
    packages = {
        'qiskit': 'qiskit',
        'pennylane': 'pennylane',
        'Bio': 'biopython',
        'sklearn': 'scikit-learn',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'torch': 'torch',
        'qiskit_aer': 'qiskit-aer'
    }
    
    missing = []
    versions = {}
    
    for name, pkg in packages.items():
        try:
            module = __import__(name)
            versions[name] = getattr(module, '__version__', 'version not found')
        except ImportError:
            missing.append(pkg)
    
    return missing, versions

def main():
    print("Checking required packages...\n")
    missing, versions = check_imports()
    
    if missing:
        print("❌ Missing packages:")
        for pkg in missing:
            print(f"- {pkg}")
        print("\nPlease install missing packages using:")
        print("pip install -r requirements.txt")
        if 'torch' in missing:
            print("\nFor PyTorch, you might need to install it separately with:")
            print("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        if 'qiskit-aer' in missing:
            print("\nFor Qiskit Aer, install it with:")
            print("pip install qiskit-aer")
        sys.exit(1)
    else:
        print("✅ All required packages are installed successfully!\n")
        print("Package versions:")
        for name, version in versions.items():
            print(f"- {name}: {version}")
        print("\nYou're all set to run the Q-Genome project!")

if __name__ == "__main__":
    main()
