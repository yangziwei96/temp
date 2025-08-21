# CLOSEgaps Validation Suite

This folder contains validation tools for the CLOSEgaps project, implementing the methodology described in the original paper.

## Overview

The validation suite includes:
1. **Database Summary** - Analysis of datasets used in the CLOSEgaps paper
2. **5-Fold Cross Validation** - Comprehensive validation following paper methodology
3. **Model Training** - Pre-training models for validation and testing

## Files

### Documentation
- `database_summary.md` - Comprehensive analysis of datasets and performance metrics from the paper
- `README.md` - This file
- `VALIDATION_REPORT.md` - Validation results report

### Validation Scripts
- `cross_validation.py` - 5-fold cross-validation implementation
- `train_models.py` - Model training script for pre-training models

## Database Summary

Based on the original paper analysis, CLOSEgaps was evaluated on:

### Metabolic Network Datasets (BiGG GEMs)
| Dataset | Species | Metabolites | Reactions | Reported Performance |
|---------|---------|-------------|-----------|---------------------|
| Yeast8.5 | *S. cerevisiae* | 1,136 | 2,514 | F1: 96%, AUC: 99% |
| iMM904 | *S. cerevisiae* S288C | 533 | 1,026 | F1: 93.87%, AUC: 98.38% |
| iAF1260b | *E. coli* K-12 | 765 | 1,612 | AUC: 98.21%, AUPR: 98.27% |
| iJO1366 | *E. coli* K-12 | 812 | 1,713 | AUC: 97.55%, AUPR: 97.58% |
| iAF692 | *M. barkeri* | 422 | 562 | AUC: 97.84%, AUPR: 97.83% |

### Chemical Reaction Datasets (USPTO)
| Dataset | Type | Metabolites | Reactions | Performance |
|---------|------|-------------|-----------|-------------|
| USPTO_3k | Chemical reactions | 6,706 | 3,000 | AUC: 95.13%, AUPR: 95.72% |
| USPTO_8k | Chemical reactions | 15,405 | 8,000 | AUC: 94.56%, AUPR: 93.62% |

## Usage Guide

### Step-by-Step Validation Process

Follow these steps in order to run the complete validation suite:

#### 1. Test Setup and Dependencies
```bash
cd validation
python test_setup.py
```
**Purpose**: Verify all dependencies are installed and data files are accessible
**What it does**:
- Checks Python packages (torch, torch-geometric, rdkit, etc.)
- Validates data file structure
- Tests ChEBI database access
- Verifies model architecture compatibility

#### 2. Train Models
```bash
python train_models.py
```
**Purpose**: Pre-train models on all available datasets
**What it does**:
- Trains models on iAF1260b, iAF692, iJO1366, iMM904
- Uses ChEBI-based negative sampling (original methodology)
- Saves trained models to `trained_models/` directory
- Saves training results to `training_results.pkl`
- **Expected time**: 10-30 minutes per dataset

#### 3. Run Cross-Validation
```bash
python cross_validation.py
```
**Purpose**: Perform comprehensive 5-fold cross-validation
**What it does**:
- Performs 5-fold cross-validation on each dataset
- Uses exact CLOSEgaps methodology with ChEBI database
- Generates comprehensive performance metrics
- Saves results to CSV files for each dataset
- **Expected time**: 15-45 minutes per dataset

#### 4. Analyze Results
```bash
# View validation results
cat validation_results_iAF1260b.csv
cat validation_results_iAF692.csv
cat validation_results_iJO1366.csv
cat validation_results_iMM904.csv
```
**Purpose**: Review and analyze validation performance
**What it produces**:
- `validation_results_{dataset}.csv` - Cross-validation results
- `training_results.pkl` - Training performance data
- `trained_models/{dataset}_model.pth` - Trained model files

#### 5. Generate Validation Report
```bash
# The validation report is automatically generated
# Check VALIDATION_REPORT.md for comprehensive results
```
**Purpose**: Get comprehensive validation summary
**What it contains**:
- Performance metrics comparison across datasets
- Methodology validation
- Results interpretation and recommendations

### Quick Start (Minimal Validation)
If you want to run a quick validation test:

```bash
cd validation
python test_setup.py                    # Step 1: Verify setup
python cross_validation.py              # Step 3: Run validation (uses pre-trained models if available)
```

### Full Validation (Complete Process)
For comprehensive validation following the paper methodology:

```bash
cd validation
python test_setup.py                    # Step 1: Verify setup
python train_models.py                  # Step 2: Train models
python cross_validation.py              # Step 3: Run validation
# Review results in VALIDATION_REPORT.md # Step 4: Analyze results
```

### Expected Output Files

After running the complete validation suite, you should have:

#### Training Output
- `trained_models/` - Directory containing saved models
  - `iAF1260b_model.pth`
  - `iAF692_model.pth`
  - `iJO1366_model.pth`
  - `iMM904_model.pth`
- `training_results.pkl` - Pickle file with training results

#### Validation Output
- `validation_results_iAF1260b.csv` - Cross-validation results for E. coli model
- `validation_results_iAF692.csv` - Cross-validation results for M. barkeri model
- `validation_results_iJO1366.csv` - Cross-validation results for E. coli model
- `validation_results_iMM904.csv` - Cross-validation results for S. cerevisiae model
- `VALIDATION_REPORT.md` - Comprehensive validation summary

### Troubleshooting

#### Common Issues and Solutions

1. **Missing Dependencies**
   ```bash
   pip install -r ../requirements.txt
   ```

2. **CUDA/GPU Issues**
   - The scripts automatically detect and use CPU if CUDA is not available
   - Check `test_setup.py` output for device information

3. **Memory Issues**
   - Reduce batch size in training scripts if needed
   - Use smaller datasets for testing

4. **Data File Issues**
   - Ensure all data files are in the correct locations
   - Check `test_setup.py` for data validation

5. **Model Loading Issues**
   - If pre-trained models are missing, run `train_models.py` first
   - Check file permissions in `trained_models/` directory

## Validation Methodology

### Data Processing
1. **Reaction Parsing** - Extract metabolites and coefficients from reaction strings
2. **Molecular Fingerprints** - Convert SMILES to 2048-dimensional Morgan fingerprints
3. **Negative Sampling** - Create negative samples by perturbing positive reactions (1:1 ratio)
4. **Data Splitting** - Stratified splits for training/validation/testing

### Model Architecture
- **Hypergraph Neural Network** with attention mechanism
- **Embedding dimension**: 64
- **Convolution dimension**: 128
- **Attention heads**: 6
- **Layers**: 2
- **Similarity matrix**: Gaussian Interaction Profile (GIP) kernel

### Training Process
- **Optimizer**: Adam (lr=1e-2, weight_decay=1e-3)
- **Loss function**: Cross-entropy
- **Batch size**: 256
- **Early stopping**: Based on validation F1 score
- **Patience**: 10 epochs

### Evaluation Metrics
- **F1 Score** - Harmonic mean of precision and recall
- **Precision** - True positives / (True positives + False positives)
- **Recall** - True positives / (True positives + False negatives)
- **AUC** - Area under ROC curve
- **AUPR** - Area under Precision-Recall curve

## Data Requirements

### Available Data
- [AVAILABLE] iAF1260b (reactions + metabolites)
- [AVAILABLE] iAF692 (reactions + metabolites)
- [AVAILABLE] iJO1366 (reactions + metabolites)
- [AVAILABLE] iMM904 (reactions + metabolites)
- [AVAILABLE] uspto_3k (reactions + metabolites)
- [AVAILABLE] uspto_8k (reactions + metabolites)

### Missing Data
- [MISSING] Pre-trained models (need to train from scratch)
- [MISSING] 24 CarveMe draft GEMs (used in fermentation study)
- [MISSING] Complete training splits for main.py

## Dependencies

Required packages (see `../requirements.txt`):
- PyTorch
- PyTorch Geometric
- RDKit
- scikit-learn
- pandas
- numpy
- tqdm

## Output Files

### Training Output
- `trained_models/` - Directory containing saved models
- `training_results.pkl` - Pickle file with training results

### Validation Output
- `validation_results_{dataset}.csv` - Cross-validation results for each dataset
- Console output with detailed performance metrics