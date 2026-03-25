# LPI (Layer-wise Probability Whitening DNN Prediction Process)

A PyTorch-based framework for layer-wise interpretation and explanation of deep neural networks. This project implements methods to analyze and understand the internal representations of different neural network architectures including DNN, ResNet, VGG, and Vision Transformers.

## Project Structure

LPI/ 
└── LPI_model/ 

├── layer_wise_prediction_DNN.py # Layer-wise prediction for Deep Neural Networks 

├── layer_wise_prediction_Resnet.py # Layer-wise prediction for ResNet architectures 

├── layer_wise_prediction_VGG.py # Layer-wise prediction for VGG architectures 

└── layer_wise_prediction_Transformer.py # Layer-wise prediction for Vision Transformers

This project provides a unified framework for interpreting deep learning models at different layers. The core idea is to:

1. **Extract** features from intermediate layers of pre-trained models
2. **Predict** class labels using these features
3. **Reconstruct** the original features from predictions to ensure consistency
4. **Analyze** how information is transformed across layers

## Key Components

### Architecture-Specific Implementations

### Key Features

- **Hook Mechanism**: Uses PyTorch forward hooks to capture intermediate layer features
- **Flexible Architecture**: Easy to extend to other model architectures
- **Multiple Loss Functions**: Combines classification and reconstruction objectives
- **GPU Support**: Automatically detects and uses CUDA if available

### Dependencies

- PyTorch
- NumPy
- tqdm (for progress bars)
