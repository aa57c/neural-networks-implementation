# Neural Network Project

A comprehensive neural network implementation featuring both from-scratch components and practical heart disease prediction models.

## Project Overview

This project contains two main components:

1. **From-Scratch Neural Network Implementation** - A complete neural network built from the ground up using only NumPy
2. **Heart Disease Prediction Models** - Practical application using Keras to predict heart disease from patient data

## Project Structure

```
src/
├── FromScratchTutorial/
│   ├── ActivationFunction.py    # Tanh activation function implementation
│   ├── ActivationLayer.py       # Activation layer class
│   ├── FCLayer.py               # Fully connected layer implementation
│   ├── Layer.py                 # Base layer abstract class
│   ├── Loss.py                  # Loss functions (MSE)
│   ├── Network.py               # Main network class
│   └── XOR.py                   # XOR problem demonstration
├── Network_Ver1.py              # Heart disease prediction (v1)
└── Network_Ver2.py              # Heart disease prediction (v2)
```

## Features

### From-Scratch Implementation
- **Modular Design**: Separate classes for different layer types and functions
- **Activation Functions**: Tanh implementation with derivatives
- **Fully Connected Layers**: Complete forward and backward propagation
- **Training**: Custom backpropagation algorithm
- **XOR Problem**: Classic neural network test case

### Heart Disease Prediction
- **Data Processing**: Feature scaling and one-hot encoding
- **Multiple Architectures**: Two different network configurations
- **Performance Metrics**: Accuracy scoring and confusion matrices
- **Data Export**: Training/testing data saved for analysis

## Requirements

```python
numpy
pandas
keras/tensorflow
scikit-learn
```

## Installation

1. Clone the repository
2. Install required dependencies:
```bash
pip install numpy pandas tensorflow scikit-learn
```

## Usage

### From-Scratch Neural Network

Run the XOR example to see the from-scratch implementation in action:

```python
python src/FromScratchTutorial/XOR.py
```

This demonstrates:
- Creating a 2-3-1 network architecture
- Training on the XOR problem
- Forward and backward propagation

### Heart Disease Prediction

Run either version of the heart disease prediction model:

```python
# Version 1: Uses one-hot encoding for chest pain types
python src/Network_Ver1.py

# Version 2: Simplified input features
python src/Network_Ver2.py
```

**Dataset**: Uses the processed Cleveland heart disease dataset with 13 input features.

**Target Variable**: Binary classification (0 = no heart disease, 1 = heart disease present)

## Model Architectures

### Version 1
- **Input**: 16 features (after one-hot encoding)
- **Hidden Layers**: 2 layers with 6 neurons each (ReLU activation)
- **Output**: 1 neuron (Tanh activation)
- **Optimizer**: Adam
- **Loss**: Binary crossentropy

### Version 2
- **Input**: 13 features (no one-hot encoding)
- **Hidden Layers**: 2 layers with 7 neurons each (ReLU activation)
- **Output**: 1 neuron (Tanh activation)
- **Optimizer**: Adam
- **Loss**: Binary crossentropy

## Key Components

### From-Scratch Implementation

**Layer Base Class** (`Layer.py`):
- Abstract base class defining the interface for all layer types
- Requires implementation of forward and backward propagation

**Fully Connected Layer** (`FCLayer.py`):
- Implements dense/fully connected layers
- Handles weight initialization, forward pass, and gradient computation
- Updates weights and biases during backpropagation

**Activation Layer** (`ActivationLayer.py`):
- Applies activation functions element-wise
- Computes gradients for backpropagation
- Currently supports tanh activation

**Network Class** (`Network.py`):
- Manages the entire neural network
- Handles training loop with forward/backward propagation
- Supports prediction on new data

## Data Processing

The heart disease prediction models include comprehensive data preprocessing:

- **Feature Scaling**: StandardScaler normalization for all numerical features
- **Categorical Encoding**: One-hot encoding for chest pain types (Version 1)
- **Target Transformation**: Multi-class labels converted to binary classification
- **Train/Test Split**: 80/20 split with shuffling

## Results and Output

Both models output:
- Training progress with epoch-by-epoch error/accuracy
- Confusion matrix for test set predictions
- Final accuracy score
- Saved CSV files with predictions and processed data

## Educational Value

This project demonstrates:
- **Neural Network Fundamentals**: Building networks from mathematical foundations
- **Gradient Descent**: Manual implementation of backpropagation
- **Practical ML Pipeline**: Data preprocessing, model training, and evaluation
- **Comparative Analysis**: Different architectures and preprocessing approaches

## Notes

- The from-scratch implementation uses only NumPy for educational purposes
- Heart disease models use Keras for practical application
- All data is saved to CSV files for analysis and debugging
- Random seeds may need to be set for reproducible results

## Future Improvements

- Add more activation functions (ReLU, Sigmoid, etc.)
- Implement different optimizers (SGD, RMSprop)
- Add regularization techniques
- Cross-validation for better model evaluation
- Hyperparameter tuning

## References

- From-scratch tutorial based on: [Math Neural Network from Scratch in Python](https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65)
- Dataset: Cleveland Heart Disease Dataset
