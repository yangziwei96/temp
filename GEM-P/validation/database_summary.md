# CLOSEgaps Database Summary

**Based on: "A generalizable framework for unlocking missing reactions in genome-scale metabolic networks using deep learning"**

## Overview

CLOSEgaps was evaluated on multiple datasets including metabolic networks and chemical reaction datasets. The paper reports comprehensive performance metrics across different validation scenarios.

## Primary Datasets

### 1. Metabolic Network Datasets (BiGG GEMs)

| Dataset | Species | Metabolites (vertices) | Reactions (hyperlinks) | Performance |
|---------|---------|----------------------|----------------------|-------------|
| **Yeast8.5** | *Saccharomyces cerevisiae* (Jul. 2021) | 1,136 | 2,514 | F1: 96%, AUC: 99%, AUPR: 99%, Precision: 95%, Recall: 96% |
| **iMM904** | *Saccharomyces cerevisiae* S288C (Oct. 2019) | 533 | 1,026 | AUC: 98.38%, AUPR: 98.28%, Recall: 96.6%, Precision: 91.28%, F1: 93.87% |
| **iAF1260b** | *Escherichia coli* str. K-12 substr. MG1655 | 765 | 1,612 | AUC: 98.21%, AUPR: 98.27% |
| **iJO1366** | *Escherichia coli* str. K-12 substr. MG1655 | 812 | 1,713 | AUC: 97.55%, AUPR: 97.58% |
| **iAF692** | *Methanosarcina barkeri* str. *Fusaro* | 422 | 562 | AUC: 97.84%, AUPR: 97.83% |

### 2. Chemical Reaction Datasets (USPTO)

| Dataset | Type | Metabolites | Reactions | Performance |
|---------|------|-------------|-----------|-------------|
| **USPTO_3k** | Chemical reactions | 6,706 | 3,000 | AUC: 95.13%, AUPR: 95.72% |
| **USPTO_8k** | Chemical reactions | 15,405 | 8,000 | AUC: 94.56%, AUPR: 93.62% |


## Data Sources

### Primary Data Sources
1. **BiGG Database**: High-quality GEMs for model organisms
2. **ChEBI Database**: Chemical entities of biological interest for metabolite representation
3. **USPTO Dataset**: Chemical reaction patents for organic chemistry validation
4. **CarveMe Pipeline**: Automatic GEM reconstruction for 24 bacterial organisms

### Data Processing
- **SMILES Representation**: Metabolites converted to molecular fingerprints
- **Negative Sampling**: 50% metabolite replacement strategy using ChEBI database
- **Feature Engineering**: 2048-dimensional molecular fingerprints
- **Similarity Matrix**: Gaussian Interaction Profile (GIP) kernel computation

## Data Availability in Repository

### Available Datasets
- [AVAILABLE] iAF1260b (reactions + metabolites)
- [AVAILABLE] iAF692 (reactions + metabolites)
- [AVAILABLE] iJO1366 (reactions + metabolites)
- [AVAILABLE] iMM904 (reactions + metabolites)
- [AVAILABLE] yeast (validation data only)
- [AVAILABLE] uspto_3k (reactions + metabolites)
- [AVAILABLE] uspto_8k (reactions + metabolites)
- [AVAILABLE] GEM XML files in data/gems/

### Missing Components
- [MISSING] Pre-trained models (.pth files)
- [MISSING] Complete training/validation splits for main.py
- [MISSING] 24 CarveMe draft GEMs used in fermentation study