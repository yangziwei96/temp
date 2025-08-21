# CLOSEgaps Validation Report

## Summary

This report presents the results of 5-fold cross-validation for CLOSEgaps using the **exact methodology** from the original source code.

## Methodology Applied

### 1. **ChEBI-Based Negative Sampling**
- **Implementation**: Used exact `create_neg_rxn()` function from source code
- **Process**: Replace 50% of metabolites with ChEBI metabolites of same atom count
- **Data Source**: `../data/pool/cleaned_chebi.csv` (44,359 metabolites)
- **Ratio**: 1:1 negative to positive samples

### 2. **Data Processing Pipeline**
- **Reaction Parsing**: Used `get_coefficient_and_reactant()` from source code
- **Incidence Matrix**: Proper creation from reaction strings
- **Molecular Features**: 2048-dimensional Morgan fingerprints
- **Similarity Matrix**: Gaussian Interaction Profile (GIP) kernel

### 3. **Data Splitting**
- **Training**: 60% of reactions
- **Validation**: 20% of reactions  
- **Testing**: 20% of reactions
- **Cross-validation**: 5-fold stratified split

## Results Summary

### Performance Metrics by Dataset

| Dataset | F1 Score | Precision | Recall | AUC | AUPR |
|---------|----------|-----------|--------|-----|------|
| **iAF1260b** | 75.56% ± 5.25% | 72.39% ± 9.60% | 81.63% ± 11.20% | 80.07% ± 6.70% | 74.95% ± 5.89% |
| **iAF692** | 66.75% ± 29.44% | 91.71% ± 9.05% | 61.89% ± 31.07% | 90.46% ± 4.41% | 89.83% ± 6.13% |
| **iJO1366** | 53.33% ± 28.10% | 55.77% ± 28.56% | 53.35% ± 31.05% | 75.54% ± 7.64% | 73.21% ± 8.64% |
| **iMM904** | 71.45% ± 18.43% | 79.55% ± 13.10% | 66.83% ± 22.53% | 80.63% ± 13.83% | 79.60% ± 11.28% |

### Detailed Results by Dataset

#### iAF1260b (E. coli)
- **Reactions**: 1,612 positive + 1,612 negative = 3,224 total
- **Metabolites**: 2,840 unique metabolites
- **Best Performance**: Fold 5 (F1: 82.63%, AUC: 86.23%)

#### iAF692 (M. barkeri)
- **Reactions**: 562 positive + 562 negative = 1,124 total
- **Metabolites**: 1,176 unique metabolites
- **Best Performance**: Fold 3 (F1: 88.46%, AUC: 95.42%)

#### iJO1366 (E. coli - larger model)
- **Reactions**: 1,713 positive + 1,713 negative = 3,426 total
- **Metabolites**: 3,034 unique metabolites
- **Best Performance**: Fold 2 (F1: 82.48%, AUC: 86.62%)

#### iMM904 (S. cerevisiae)
- **Reactions**: 1,026 positive + 1,026 negative = 2,052 total
- **Metabolites**: 1,893 unique metabolites
- **Best Performance**: Fold 1 (F1: 96.47%, AUC: 95.51%)

## Files Generated
- `validation_results_iAF1260b.csv`
- `validation_results_iAF692.csv`
- `validation_results_iJO1366.csv`
- `validation_results_iMM904.csv`
