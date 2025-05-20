from setuptools import setup, find_packages

setup(
    name="qgenome",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'torch>=1.9.0',
        'scikit-learn>=0.24.0',
        'qiskit>=0.34.0',
        'qiskit-machine-learning>=0.3.0',
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A hybrid quantum-classical DNA sequence classifier",
    python_requires=">=3.7",
)
