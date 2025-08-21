#!/usr/bin/env python3
"""
CLOSEgaps Model Training Script

This script trains CLOSEgaps models on the available datasets following the methodology
described in the original paper. It saves trained models for later validation use.

Based on: "A generalizable framework for unlocking missing reactions in genome-scale 
metabolic networks using deep learning"
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
from sklearn.model_selection import train_test_split
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

def create_incidence_matrix(reactions, metabolites_df):
    """Create incidence matrix from reactions"""
    # Parse reactions
    reactions_index, reactions_metas, reactants_nums, direction = get_coefficient_and_reactant(reactions)
    
    # Get all unique metabolites
    all_metas = list(set(sum(reactions_metas, [])))
    all_metas.sort()
    
    # Create incidence matrix
    incidence_matrix = np.zeros((len(all_metas), len(reactions)))
    
    for i, (rxn_index, rxn_metas) in enumerate(zip(reactions_index, reactions_metas)):
        for j, meta in enumerate(rxn_metas):
            if meta in all_metas:
                meta_idx = all_metas.index(meta)
                incidence_matrix[meta_idx, i] = float(rxn_index[j])
    
    return incidence_matrix, all_metas

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

def create_negative_samples(incidence_matrix, negative_ratio=1):
    """Create negative samples by perturbing positive reactions"""
    print(f"Creating negative samples with ratio {negative_ratio}:1")
    
    n_metabolites, n_reactions = incidence_matrix.shape
    negative_matrix = np.zeros((n_metabolites, n_reactions * negative_ratio))
    
    for i in range(n_reactions):
        for j in range(negative_ratio):
            # Create negative sample by randomly perturbing the reaction
            neg_reaction = incidence_matrix[:, i].copy()
            
            # Randomly change some metabolite coefficients
            non_zero_indices = np.where(neg_reaction != 0)[0]
            if len(non_zero_indices) > 1:
                # Randomly change 50% of non-zero coefficients
                num_to_change = max(1, len(non_zero_indices) // 2)
                indices_to_change = np.random.choice(non_zero_indices, num_to_change, replace=False)
                
                for idx in indices_to_change:
                    # Randomly change the coefficient
                    neg_reaction[idx] = np.random.choice([-2, -1, 0, 1, 2])
            
            negative_matrix[:, i * negative_ratio + j] = neg_reaction
    
    return negative_matrix

def prepare_training_data(incidence_matrix, features, test_size=0.2, val_size=0.1):
    """Prepare training, validation, and test data"""
    print("Preparing training data...")
    
    # Create negative samples
    negative_matrix = create_negative_samples(incidence_matrix, negative_ratio=1)
    
    # Combine positive and negative samples
    all_reactions = np.concatenate([incidence_matrix, negative_matrix], axis=1)
    
    # Create labels (1 for positive, 0 for negative)
    positive_labels = np.ones(incidence_matrix.shape[1])
    negative_labels = np.zeros(negative_matrix.shape[1])
    all_labels = np.concatenate([positive_labels, negative_labels])
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        all_reactions.T, all_labels, test_size=test_size, random_state=42, stratify=all_labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
    )
    
    # Convert to tensors
    X_train = torch.tensor(X_train.T, dtype=torch.float)
    X_val = torch.tensor(X_val.T, dtype=torch.float)
    X_test = torch.tensor(X_test.T, dtype=torch.float)
    
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    print(f"Training samples: {X_train.shape[1]}")
    print(f"Validation samples: {X_val.shape[1]}")
    print(f"Test samples: {X_test.shape[1]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, features

def train_model(X_train, X_val, y_train, y_val, features, 
                emb_dim=64, conv_dim=128, head=6, L=2, 
                lr=1e-2, weight_decay=1e-3, epochs=100, batch_size=256):
    """Train the CLOSEgaps model"""
    print("Training CLOSEgaps model...")
    
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
    best_model = None
    patience = 10
    patience_counter = 0
    
    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        
        # Batch training
        total_loss = 0
        num_batches = X_train.shape[1] // batch_size + (1 if X_train.shape[1] % batch_size != 0 else 0)
        
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, X_train.shape[1])
            
            batch_X = X_train[:, start_idx:end_idx].to(device)
            batch_y = y_train[start_idx:end_idx].to(device)
            
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
            val_outputs = model(features_tensor.to(device), X_val.to(device))
            val_probs = torch.softmax(val_outputs, dim=1)
            val_preds = (val_probs[:, 1] >= 0.5).long()
            
            val_f1 = f1_score(y_val.cpu(), val_preds.cpu())
            val_auc = roc_auc_score(y_val.cpu(), val_probs[:, 1].cpu())
            
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
    print(f"Best validation F1: {best_val_f1:.4f}")
    
    return model

def evaluate_model(model, X_test, y_test, features):
    """Evaluate the trained model on test data"""
    print("Evaluating model on test data...")
    
    model.eval()
    features_tensor = torch.tensor(features, dtype=torch.float).to(device)
    
    with torch.no_grad():
        test_outputs = model(features_tensor.to(device), X_test.to(device))
        test_probs = torch.softmax(test_outputs, dim=1)
        test_preds = (test_probs[:, 1] >= 0.5).long()
        
        # Calculate metrics
        f1 = f1_score(y_test.cpu(), test_preds.cpu())
        precision = precision_score(y_test.cpu(), test_preds.cpu())
        recall = recall_score(y_test.cpu(), test_preds.cpu())
        auc = roc_auc_score(y_test.cpu(), test_probs[:, 1].cpu())
        aupr = average_precision_score(y_test.cpu(), test_probs[:, 1].cpu())
        
        print("\nTest Results:")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"AUPR: {aupr:.4f}")
        
        return test_probs[:, 1].cpu().numpy(), test_preds.cpu().numpy()

def save_trained_model(model, dataset_name, model_dir='trained_models'):
    """Save the trained model"""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_path = os.path.join(model_dir, f'{dataset_name}_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'dataset_name': dataset_name,
        'model_config': {
            'emb_dim': 64,
            'conv_dim': 128,
            'head': 6,
            'L': 2
        }
    }, model_path)
    
    print(f"Model saved to {model_path}")
    return model_path

def load_model(model_path, features_shape, device='cpu'):
    """Load a trained model"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model
    model = CLOSEgaps(
        input_num=features_shape[0],
        input_feature_num=features_shape[1],
        emb_dim=checkpoint['model_config']['emb_dim'],
        conv_dim=checkpoint['model_config']['conv_dim'],
        head=checkpoint['model_config']['head'],
        L=checkpoint['model_config']['L']
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def train_dataset(dataset_name, epochs=100, save_model=True):
    """Train model on a specific dataset"""
    print(f"\n{'='*60}")
    print(f"Training on dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Load dataset
    reactions, metabolites_df = load_dataset(dataset_name)
    
    # Create incidence matrix and features
    incidence_matrix, all_metas = create_incidence_matrix(reactions, metabolites_df)
    molecular_features = create_molecular_features(metabolites_df, all_metas)
    
    # Prepare training data
    X_train, X_val, X_test, y_train, y_val, y_test, features = prepare_training_data(
        incidence_matrix, molecular_features
    )
    
    # Train model
    trained_model = train_model(X_train, X_val, y_train, y_val, molecular_features, epochs=epochs)
    
    # Evaluate model
    test_probs, test_preds = evaluate_model(trained_model, X_test, y_test, molecular_features)
    
    # Save model
    if save_model:
        model_path = save_trained_model(trained_model, dataset_name)
    
    # Save results
    results = {
        'dataset': dataset_name,
        'n_metabolites': len(all_metas),
        'n_reactions': len(reactions),
        'n_features': molecular_features.shape[1],
        'test_probs': test_probs,
        'test_preds': test_preds,
        'y_test': y_test.cpu().numpy()
    }
    
    return trained_model, results

def main():
    """Main execution function"""
    print("CLOSEgaps Model Training")
    print("=" * 60)
    
    # Set random seed
    set_random_seed(42)
    
    # Available datasets
    datasets = ['iAF1260b', 'iAF692', 'iJO1366', 'iMM904']
    
    # Train models for each dataset
    all_results = {}
    
    for dataset in datasets:
        try:
            model, results = train_dataset(dataset, epochs=50, save_model=True)
            all_results[dataset] = results
            
        except Exception as e:
            print(f"Error training on dataset {dataset}: {e}")
            continue
    
    # Save all results
    import pickle
    with open('training_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\nTraining completed. Results saved to training_results.pkl")
    print(f"Trained models saved to trained_models/ directory")

if __name__ == "__main__":
    main()
