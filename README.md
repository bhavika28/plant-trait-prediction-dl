# Deep Learning for Agricultural Plant Trait Prediction

## 🎯 Research Problem
Developing advanced deep learning architectures to predict plant traits from image data, advancing precision agriculture and crop optimization. This research explores the potential of computer vision models to automate plant phenotyping for agricultural applications.

## 📊 Dataset & Methodology
- **Dataset Size**: 3.17GB with high-resolution plant images
- **Data Structure**: Image data paired with tabular trait measurements
- **Source**: Kaggle agricultural dataset (archived)
- **Approach**: Multi-modal learning combining visual and tabular data
- **Training**: 50 epochs with GPU acceleration (40s/epoch)

## 🔑 Key Results
- **Model Performance**: [Add your best accuracy/F1 score]
- **Architecture Comparison**: Systematic evaluation of CNN, RNN, LSTM models
- **Training Efficiency**: Optimized for HPC environments (~40 minutes total runtime)
- **Research Impact**: Demonstrates feasibility of automated plant phenotyping

## 🛠️ Technologies Used
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

## 🚀 Quick Start

### Prerequisites
- **System**: Linux/MacOS with Conda
- **Hardware**: GPU recommended for training
- **Storage**: 4GB+ free space for dataset

### Setup Instructions
```bash
# Create conda environment
conda create -n plant_traits python=3.8
conda activate plant_traits
```

```bash
# Install dependencies
pip install -r requirements.txt
```

```bash
# Download dataset (3.17GB) - Contact for access
# Place data in /data folder with images/ and tab_data.csv
```

```bash
# Run the complete pipeline
python main.py
```

## 📁 Project Structure
### Core Files:

- main.py - Main execution pipeline and training orchestration
- buildModel.py - Deep learning model architectures (CNN, RNN, LSTM)
- loadData.py - Data loading, image processing, and dataset management
- preprocessData.py - Data preprocessing, augmentation, and feature engineering
- setupGPUS.py - GPU configuration and memory optimization
- visualization.py - Results visualization and performance analysis
- requirements.txt - Python dependencies and library versions


## 🔬 Model Architectures Explored
### Convolutional Neural Networks (CNN)

- Extract spatial features from plant images
- Multi-layer convolution with pooling layers
- Performance: [Add your CNN results]

### Recurrent Neural Networks (RNN/LSTM)

- Process sequential plant growth data
- LSTM layers for temporal pattern recognition
- Performance: [Add your RNN/LSTM results]

### Hybrid Approaches

- Multi-modal fusion combining image and tabular data
- Ensemble methods with multiple architecture voting
- Transfer learning with pre-trained agricultural models

## 📈 Training Details

- Epochs: 50 (configurable)
- Training Time: ~40 minutes on HPC infrastructure
- Hardware: GPU-accelerated with memory optimization
- Validation: Robust evaluation with cross-validation
- Metrics: Accuracy, F1-score, precision, recall

## 🎯 Research Contributions

- Architecture Comparison: Systematic evaluation of deep learning approaches for plant phenotyping
- Multi-modal Learning: Effective fusion of visual and tabular agricultural data
- Scalability: Production-ready pipeline for large-scale agricultural datasets
- Performance Optimization: GPU-accelerated training for practical deployment

## 🌱 Applications in Agriculture

- Precision Farming: Automated trait assessment for crop optimization
- Breeding Programs: Accelerated phenotyping for plant selection
- Yield Prediction: Early-season crop performance forecasting
- Disease Detection: Visual pattern recognition for plant health monitoring
