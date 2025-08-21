# CLOSEgaps Tutorial: Deep Learning for Metabolic Network Gap-Filling

This tutorial provides a step-by-step guide to understanding and implementing CLOSEgaps, a deep learning-driven tool for automatic gap-filling in Genome-scale Metabolic Models (GEMs).

## Overview

CLOSEgaps addresses a critical challenge in systems biology where incomplete knowledge of metabolic processes hinders the accuracy of metabolic models. It models the gap-filling problem as a **hyperedge prediction problem** within metabolic networks using deep learning.

## Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install torch torch-geometric torch-scatter torch-sparse
pip install pandas numpy scikit-learn rdkit tqdm matplotlib seaborn
pip install cobra optlang joblib
```

### 2. Run the Tutorial

You have two options:

#### Option A: Jupyter Notebook (Recommended)
```bash
# Start Jupyter
jupyter lab

# Open CLOSEgaps_Tutorial.ipynb
```

#### Option B: Python Script
```bash
# Run the complete tutorial
python CLOSEgaps_Tutorial.py
```

## Tutorial Structure

The tutorial is organized into 10 main sections:

### 1. **Setup and Dependencies**
- Install required packages
- Set up environment and random seeds

### 2. **Data Processing Functions**
- Parse reaction strings
- Convert SMILES to molecular fingerprints
- Compute similarity matrices

### 3. **CLOSEgaps Model Architecture**
- Hypergraph Neural Network implementation
- Multi-head attention mechanisms
- Molecular feature encoding

### 4. **Data Loading and Preprocessing**
- Load metabolic networks from datasets
- Create incidence matrices
- Generate molecular features

### 5. **Create Training Data**
- Generate negative samples
- Split data into train/validation/test sets
- Prepare tensors for training

### 6. **Model Training**
- Train the hypergraph neural network
- Monitor validation metrics
- Save best model

### 7. **Model Evaluation**
- Evaluate on test data
- Calculate performance metrics (F1, AUC, AUPR)
- Analyze prediction quality

### 8. **Prediction on New Data**
- Predict missing reactions
- Identify high-probability candidates
- Rank predictions

### 9. **Visualization and Analysis**
- Plot training curves
- Analyze metabolite participation
- Visualize network structure

### 10. **Summary and Next Steps**
- Review key insights
- Suggest further experiments
- Discuss applications

## Key Concepts

### Hypergraph Representation
- **Nodes**: Metabolites (with SMILES molecular representations)
- **Hyperedges**: Reactions (connecting multiple metabolites)
- **Features**: Morgan fingerprints (2048-bit molecular descriptors)

### Model Architecture
- **Input**: Molecular fingerprints + incidence matrix
- **Processing**: Hypergraph convolution with attention
- **Output**: Reaction probability scores

### Training Strategy
- **Positive samples**: Real reactions from GEMs
- **Negative samples**: Perturbed reactions
- **Validation**: Biological relevance through FBA

## Available Datasets

The tutorial includes several datasets:

| Dataset | Species | Metabolites | Reactions |
|---------|---------|-------------|-----------|
| iAF1260b | E. coli | 765 | 1612 |
| iMM904 | S. cerevisiae | 533 | 1026 |
| iJO1366 | E. coli | 812 | 1713 |
| iAF692 | M. barkeri | 422 | 562 |
| USPTO_3k | Chemical | 6706 | 3000 |
| USPTO_8k | Chemical | 15405 | 8000 |

## Customization

### Change Dataset
```python
# In the tutorial, change this line:
dataset_name = 'iAF1260b'  # Try: 'iMM904', 'iJO1366', 'iAF692'
```

### Modify Model Parameters
```python
# Adjust hyperparameters in train_model function:
emb_dim=64      # Embedding dimension
conv_dim=128    # Convolution dimension
head=6          # Number of attention heads
L=2             # Number of layers
epochs=50       # Training epochs
```

### Experiment with Different Strategies
- Try different negative sample generation methods
- Test various molecular fingerprint types
- Experiment with different similarity kernels

## Expected Results

With the default settings, you should achieve:
- **AUC**: 94-99%
- **F1 Score**: 87-96%
- **AUPR**: 93-99%

## Biological Validation

The tutorial includes a framework for biological validation:
- Flux Balance Analysis (FBA) integration
- Metabolite production prediction
- Pathway analysis

## Troubleshooting

### Common Issues

1. **PyTorch Geometric Installation**
   ```bash
   # For CUDA support
   pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
   ```

2. **RDKit Installation**
   ```bash
   # Using conda (recommended)
   conda install -c conda-forge rdkit
   
   # Using pip
   pip install rdkit-pypi
   ```

3. **Memory Issues**
   - Reduce batch size
   - Use smaller datasets
   - Enable gradient checkpointing

### Performance Tips
- Use GPU if available
- Reduce feature dimension for faster training
- Use smaller datasets for initial testing

## Further Reading

- Original CLOSEgaps paper: [arXiv:2409.13259](https://arxiv.org/abs/2409.13259)
- PyTorch Geometric documentation: [https://pytorch-geometric.readthedocs.io/](https://pytorch-geometric.readthedocs.io/)
- COBRA toolbox: [https://opencobra.github.io/cobrapy/](https://opencobra.github.io/cobrapy/)