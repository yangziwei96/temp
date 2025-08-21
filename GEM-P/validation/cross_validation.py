#!/usr/bin/env python3
"""
CLOSEgaps 5-Fold Cross Validation - CORRECTED VERSION

This script implements 5-fold cross-validation for CLOSEgaps following the EXACT methodology
from the original source code, including proper ChEBI-based negative sampling.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import re
import math
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import CLOSEgaps modules
sys.path.append('..')
from CLOSEgaps import CLOSEgaps
from utils import set_random_seed, smiles_to_fp, getGipKernel

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

def set_random_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_coefficient_and_reactant(reactions):
    """Parse reaction strings to extract coefficients, metabolites, and directions"""
    reactions_index = []
    reactions_metas = []
    reactants_nums = []
    direction = []
    
    for rxn in reactions:
        rxn_index, rxn_metas, rxn_direction = [], [], '=>'
        if '<=>' in rxn:
            rxn_direction = '<=>'
        
        tem_rxn = rxn.replace(' ' + rxn_direction + ' ', ' + ')
        metas = tem_rxn.split(' + ')
        
        for m in metas:
            a = re.findall('\d+\.?\d*', m)
            b = m.split(' ')
            if len(a) and a[0] == b[0]:
                rxn_index.append(a[0])
                rxn_metas.append(' '.join(b[1:]))
            else:
                rxn_metas.append(m)
                rxn_index.append('1')
        
        reactant, product = rxn.split(' ' + rxn_direction + ' ')
        reactants = reactant.split(' + ')
        products = product.split(' + ')
        reactants_nums.append(len(reactants))
        
        reactions_index.append(rxn_index)
        reactions_metas.append(rxn_metas)
        direction.append(rxn_direction)
    
    return reactions_index, reactions_metas, reactants_nums, direction

def combine_meta_rxns(reactions_index, reactions_metas, reactants_nums, reactants_direction):
    """Combine metabolites back into reaction string"""
    rxn = ''
    for i in range(len(reactions_metas)):
        if i < reactants_nums:
            rxn += reactions_index[i] + ' ' + reactions_metas[i]
        else:
            rxn += ' + ' + reactions_index[i] + ' ' + reactions_metas[i]
    
    rxn += ' ' + reactants_direction + ' '
    
    for i in range(reactants_nums, len(reactions_metas)):
        if i == reactants_nums:
            rxn += reactions_index[i] + ' ' + reactions_metas[i]
        else:
            rxn += ' + ' + reactions_index[i] + ' ' + reactions_metas[i]
    
    return rxn

def create_neg_rxn_correct(pos_rxn, pos_data_source, neg_data_source, balanced_atom=False, negative_ratio=1, atom_ratio=0.5):
    """Create negative samples using EXACT CLOSEgaps methodology"""
    print(f"Creating negative samples using CLOSEgaps methodology (ratio {negative_ratio}:1, atom_ratio {atom_ratio})")
    
    reactions_index, reactions_metas, reactants_nums, reactants_direction = get_coefficient_and_reactant(pos_rxn)
    neg_rxn_name_list = []
    
    assert negative_ratio >= 1 and isinstance(negative_ratio, int)
    
    for i in tqdm(range(len(reactions_index))):
        for j in range(negative_ratio):
            selected_atoms = math.floor(len(reactions_metas[i]) * atom_ratio)
            assert selected_atoms > 0, "The number of selected atoms is zero"
            index_value = random.sample(list(enumerate(reactions_metas[i])), selected_atoms)
            
            for index, meta in index_value:
                dup = True
                count = pos_data_source[pos_data_source['name'] == meta]['count'].values[0]
                
                while dup:
                    found_chebi_metas = neg_data_source[neg_data_source['count'] == count].sample(1)['name'].values[0]
                    if not balanced_atom:
                        break
                    if found_chebi_metas not in reactions_metas[i]:
                        dup = False
                
                neg_metas = reactions_metas[i].copy()
                neg_metas[index] = found_chebi_metas
            
            neg_rxns = combine_meta_rxns(reactions_index[i], neg_metas, reactants_nums[i], reactants_direction[i])
            neg_rxn_name_list.append(neg_rxns)
    
    return neg_rxn_name_list

def load_dataset(dataset_name='iAF1260b'):
    """Load a specific dataset"""
    print(f"Loading dataset: {dataset_name}")
    
    # Load reactions
    rxn_file = f'../data/{dataset_name}/{dataset_name}_rxn_name_list.txt'
    with open(rxn_file, 'r') as f:
        reactions = [line.strip() for line in f.readlines()]
    
    # Load metabolites with SMILES
    meta_file = f'../data/{dataset_name}/{dataset_name}_meta_count.csv'
    metabolites_df = pd.read_csv(meta_file)
    
    print(f"Loaded {len(reactions)} reactions and {len(metabolites_df)} metabolites")
    return reactions, metabolites_df

def create_incidence_matrix_from_reactions(reactions, all_metas):
    """Create incidence matrix from reaction strings"""
    reactions_index, reactions_metas, reactants_nums, direction = get_coefficient_and_reactant(reactions)
    
    # Create incidence matrix
    incidence_matrix = np.zeros((len(all_metas), len(reactions)))
    
    for i, (rxn_index, rxn_metas) in enumerate(zip(reactions_index, reactions_metas)):
        for j, meta in enumerate(rxn_metas):
            if meta in all_metas:
                meta_idx = all_metas.index(meta)
                incidence_matrix[meta_idx, i] = float(rxn_index[j])
    
    return incidence_matrix

def create_molecular_features(metabolites_df, all_metas):
    """Create molecular fingerprint features for metabolites"""
    print("Creating molecular fingerprints...")
    
    # Create name to SMILES mapping
    name_to_smiles = dict(zip(metabolites_df['name'], metabolites_df['smiles']))
    
    # Create features for all metabolites
    features = []
    for meta in all_metas:
        if meta in name_to_smiles:
            smiles = name_to_smiles[meta]
            fp = smiles_to_fp(smiles)[0]  # Remove extra dimension
        else:
            # Use zero vector for unknown metabolites
            fp = np.zeros(2048)
        features.append(fp)
    
    features = np.array(features)
    print(f"Created features with shape: {features.shape}")
    return features

def prepare_cross_validation_data_correct(dataset_name='iAF1260b', negative_ratio=1, atom_ratio=0.5, balanced_atom=False):
    """Prepare data for cross-validation using EXACT CLOSEgaps methodology"""
    print("Preparing cross-validation data using CLOSEgaps methodology...")
    
    # Load positive reactions
    pos_rxn, pos_metas_smiles = load_dataset(dataset_name)
    
    # Load ChEBI database for negative sampling
    chebi_meta_filter = pd.read_csv('../data/pool/cleaned_chebi.csv')
    
    # Create negative reactions using CLOSEgaps methodology
    neg_rxn = create_neg_rxn_correct(pos_rxn, pos_metas_smiles, chebi_meta_filter, 
                                   balanced_atom, negative_ratio, atom_ratio)
    
    # Parse both positive and negative reactions
    pos_index, pos_metas, pos_nums, pos_directions = get_coefficient_and_reactant(pos_rxn)
    neg_index, neg_metas, neg_nums, neg_directions = get_coefficient_and_reactant(neg_rxn)
    
    # Get all unique metabolites
    all_metas = list(set(sum(pos_metas, []) + sum(neg_metas, [])))
    all_metas.sort()
    
    # Create positive incidence matrix
    pos_matrix = np.zeros((len(all_metas), len(pos_rxn)))
    for i in range(len(pos_index)):
        for j in range(len(pos_metas[i])):
            if pos_metas[i][j] in all_metas:
                meta_idx = all_metas.index(pos_metas[i][j])
                pos_matrix[meta_idx, i] = float(pos_index[i][j])
    
    # Create negative incidence matrix
    neg_matrix = np.zeros((len(all_metas), len(neg_rxn)))
    for i in range(len(neg_index)):
        for j in range(len(neg_metas[i])):
            if neg_metas[i][j] in all_metas:
                meta_idx = all_metas.index(neg_metas[i][j])
                neg_matrix[meta_idx, i] = float(neg_index[i][j])
    
    # Combine positive and negative samples
    all_reactions = np.concatenate([pos_matrix, neg_matrix], axis=1)
    
    # Create labels (1 for positive, 0 for negative)
    positive_labels = np.ones(len(pos_rxn))
    negative_labels = np.zeros(len(neg_rxn))
    all_labels = np.concatenate([positive_labels, negative_labels])
    
    # Create molecular features
    molecular_features = create_molecular_features(pos_metas_smiles, all_metas)
    
    print(f"Total reactions: {all_reactions.shape[1]} (positive: {len(pos_rxn)}, negative: {len(neg_rxn)})")
    print(f"Metabolites: {all_reactions.shape[0]}")
    
    return all_reactions, all_labels, molecular_features

def train_fold_model_correct(X_train, X_val, y_train, y_val, features, 
                           emb_dim=64, conv_dim=128, head=6, L=2, 
                           lr=1e-2, weight_decay=1e-3, epochs=100, batch_size=256):
    """Train CLOSEgaps model for one fold using correct methodology"""
    
    # Calculate similarity matrix
    features_tensor = torch.tensor(features, dtype=torch.float).to(device)
    similarity_matrix = getGipKernel(features_tensor, False, 1.0).to(device)
    
    # Initialize model
    model = CLOSEgaps(
        input_num=features.shape[0],
        input_feature_num=features.shape[1],
        emb_dim=emb_dim,
        conv_dim=conv_dim,
        head=head,
        L=L,
        similarity=similarity_matrix
    ).to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_f1 = 0
    best_model = model.state_dict().copy()  # Initialize with current model
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        
        # Batch training - X_train is in incidence matrix format
        total_loss = 0
        num_batches = X_train.shape[1] // batch_size + (1 if X_train.shape[1] % batch_size != 0 else 0)
        
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, X_train.shape[1])
            
            # Use incidence matrix format directly
            batch_X = torch.tensor(X_train[:, start_idx:end_idx], dtype=torch.float).to(device)
            batch_y = torch.tensor(y_train[start_idx:end_idx], dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(features_tensor.to(device), batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_X = torch.tensor(X_val, dtype=torch.float).to(device)
            val_outputs = model(features_tensor.to(device), val_X)
            val_probs = torch.softmax(val_outputs, dim=1)
            val_preds = (val_probs[:, 1] >= 0.5).long()
            
            val_f1 = f1_score(y_val, val_preds.cpu())
            val_auc = roc_auc_score(y_val, val_probs[:, 1].cpu())
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    model.load_state_dict(best_model)
    return model

def evaluate_fold_correct(model, X_test, y_test, features):
    """Evaluate model on test data using correct methodology"""
    model.eval()
    features_tensor = torch.tensor(features, dtype=torch.float).to(device)
    
    with torch.no_grad():
        test_X = torch.tensor(X_test, dtype=torch.float).to(device)
        test_outputs = model(features_tensor.to(device), test_X)
        test_probs = torch.softmax(test_outputs, dim=1)
        test_preds = (test_probs[:, 1] >= 0.5).long()
        
        # Calculate metrics
        f1 = f1_score(y_test, test_preds.cpu())
        precision = precision_score(y_test, test_preds.cpu())
        recall = recall_score(y_test, test_preds.cpu())
        auc = roc_auc_score(y_test, test_probs[:, 1].cpu())
        aupr = average_precision_score(y_test, test_probs[:, 1].cpu())
        
        return {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'aupr': aupr,
            'probabilities': test_probs[:, 1].cpu().numpy(),
            'predictions': test_preds.cpu().numpy()
        }

def run_cross_validation_correct(dataset_name='iAF1260b', n_folds=5, n_runs=1, 
                               negative_ratio=1, atom_ratio=0.5, balanced_atom=False):
    """Run 5-fold cross-validation using EXACT CLOSEgaps methodology"""
    print(f"Starting {n_folds}-fold cross-validation on {dataset_name} (CLOSEgaps METHODOLOGY)")
    print("=" * 60)
    
    # Prepare cross-validation data using CLOSEgaps methodology
    X, y, features = prepare_cross_validation_data_correct(dataset_name, negative_ratio, atom_ratio, balanced_atom)
    
    # Initialize results storage
    all_results = []
    
    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs}")
        
        # Initialize cross-validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42 + run)
        
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X.T, y)):  # Transpose X for sklearn
            print(f"\nFold {fold + 1}/{n_folds}")
            print(f"Train reactions: {len(train_idx)}, Test reactions: {len(test_idx)}")
            
            # Split data - X is in incidence matrix format (metabolites x reactions)
            X_train = X[:, train_idx]
            X_test = X[:, test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]
            
            # Further split training data for validation (60% train, 20% val, 20% test)
            val_size = int(0.25 * len(train_idx))  # 20% of total = 25% of train
            val_idx = np.random.choice(len(train_idx), val_size, replace=False)
            train_idx_final = np.setdiff1d(np.arange(len(train_idx)), val_idx)
            
            X_train_final = X_train[:, train_idx_final]
            X_val = X_train[:, val_idx]
            y_train_final = y_train[train_idx_final]
            y_val = y_train[val_idx]
            
            print(f"Final train: {X_train_final.shape[1]}, Validation: {X_val.shape[1]}, Test: {X_test.shape[1]}")
            
            # Train model
            model = train_fold_model_correct(X_train_final, X_val, y_train_final, y_val, features)
            
            # Evaluate model
            results = evaluate_fold_correct(model, X_test, y_test, features)
            
            print(f"Fold {fold + 1} Results:")
            print(f"  F1: {results['f1']:.4f}")
            print(f"  Precision: {results['precision']:.4f}")
            print(f"  Recall: {results['recall']:.4f}")
            print(f"  AUC: {results['auc']:.4f}")
            print(f"  AUPR: {results['aupr']:.4f}")
            
            fold_results.append(results)
        
        # Calculate average results for this run
        avg_results = {}
        for metric in ['f1', 'precision', 'recall', 'auc', 'aupr']:
            avg_results[metric] = np.mean([r[metric] for r in fold_results])
            avg_results[f'{metric}_std'] = np.std([r[metric] for r in fold_results])
        
        print(f"\nRun {run + 1} Average Results:")
        print(f"  F1: {avg_results['f1']:.4f} ± {avg_results['f1_std']:.4f}")
        print(f"  Precision: {avg_results['precision']:.4f} ± {avg_results['precision_std']:.4f}")
        print(f"  Recall: {avg_results['recall']:.4f} ± {avg_results['recall_std']:.4f}")
        print(f"  AUC: {avg_results['auc']:.4f} ± {avg_results['auc_std']:.4f}")
        print(f"  AUPR: {avg_results['aupr']:.4f} ± {avg_results['aupr_std']:.4f}")
        
        all_results.append(avg_results)
    
    return all_results

def main():
    """Main execution function"""
    print("CLOSEgaps 5-Fold Cross Validation - CORRECTED VERSION")
    print("=" * 60)
    
    # Set random seed
    set_random_seed(42)
    
    # Available datasets
    datasets = ['iAF1260b', 'iAF692', 'iJO1366', 'iMM904']
    
    # Run cross-validation for each dataset
    for dataset in datasets:
        try:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset}")
            print(f"{'='*60}")
            
            results = run_cross_validation_correct(
                dataset_name=dataset, 
                n_folds=5, 
                n_runs=1,
                negative_ratio=1,  # 1:1 negative to positive ratio
                atom_ratio=0.5,    # Replace 50% of metabolites
                balanced_atom=False # Don't require balanced atom numbers
            )
            
            # Save results
            results_df = pd.DataFrame(results)
            results_df.to_csv(f'validation_results_correct_{dataset}.csv', index=False)
            print(f"Results saved to validation_results_correct_{dataset}.csv")
            
        except Exception as e:
            print(f"Error processing dataset {dataset}: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()
