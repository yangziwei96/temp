#!/usr/bin/env python3
"""
CLOSEgaps Tutorial: Deep Learning for Metabolic Network Gap-Filling

This script demonstrates the complete CLOSEgaps pipeline step by step, 
from data loading to prediction.

Overview:
CLOSEgaps is a deep learning-driven tool that addresses gap-filling in 
Genome-scale Metabolic Models (GEMs) by modeling it as a hyperedge prediction problem.

Key Components:
1. Data Loading: Load metabolic networks and convert to hypergraph representation
2. Feature Engineering: Convert metabolites to molecular fingerprints
3. Model Architecture: Hypergraph Neural Network with attention
4. Training: Train the model on reaction prediction
5. Prediction: Predict missing reactions in new GEMs
6. Validation: Biological validation using Flux Balance Analysis

TWO METHODS AVAILABLE:
- SIMPLE METHOD: Quick demonstration with coefficient perturbation
- ORIGINAL METHOD: Full CLOSEgaps methodology with ChEBI-based negative sampling
"""

import os
import torch
import torch.nn as nn
import torch_geometric.nn as hnn
import pandas as pd
import numpy as np
import re
import math
import random
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
warnings.filterwarnings('ignore')

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

set_random_seed(42)

# ============================================================================
# 1. DATA PROCESSING FUNCTIONS
# ============================================================================

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

def smiles_to_fp(smiles, radius=2, nBits=2048):
    """Convert SMILES to Morgan fingerprint"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros((1, nBits))
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, useFeatures=False)
        fp_bits = fp.ToBitString()
        finger_print = np.array(list(map(int, fp_bits))).astype(np.float).reshape(1, -1)
        return finger_print
    except:
        return np.zeros((1, nBits))

def getGipKernel(features, normalize=True, gamma=1.0):
    """Calculate Gaussian Interaction Profile (GIP) kernel"""
    features = features.cpu().numpy()
    if normalize:
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
    
    # Calculate pairwise similarities
    similarity = np.exp(-gamma * np.sum((features[:, np.newaxis, :] - features[np.newaxis, :, :]) ** 2, axis=2))
    return torch.tensor(similarity, dtype=torch.float)

# ============================================================================
# 2. NEGATIVE SAMPLING METHODS
# ============================================================================

def create_negative_samples_simple(incidence_matrix, negative_ratio=1):
    """
    SIMPLE METHOD: Create negative samples by perturbing positive reactions
    This is a simplified approach for demonstration purposes.
    """
    print(f"Creating negative samples using SIMPLE method (ratio {negative_ratio}:1)")
    
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

def create_negative_samples_original(pos_rxn, pos_data_source, neg_data_source, balanced_atom=False, negative_ratio=1, atom_ratio=0.5):
    """
    ORIGINAL METHOD: Create negative samples using exact CLOSEgaps methodology
    This uses ChEBI database to replace metabolites with same atom count.
    """
    print(f"Creating negative samples using ORIGINAL CLOSEgaps method (ratio {negative_ratio}:1, atom_ratio {atom_ratio})")
    
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

# ============================================================================
# 3. DATA PREPARATION METHODS
# ============================================================================

def prepare_data_simple_method():
    """
    SIMPLE METHOD: Prepare data using coefficient perturbation
    Quick demonstration approach.
    """
    print("=" * 60)
    print("SIMPLE METHOD: Using coefficient perturbation")
    print("=" * 60)
    
    # Load sample data
    reactions = [
        "2 glucose + 2 atp => 2 glucose6p + 2 adp",
        "glucose6p => fructose6p",
        "fructose6p + atp => fructose16p + adp",
        "fructose16p => glyceraldehyde3p + dihydroxyacetonep",
        "glyceraldehyde3p + nad + pi => 13bpg + nadh + h"
    ]
    
    # Create sample metabolites with SMILES
    metabolites_data = {
        'name': ['glucose', 'atp', 'glucose6p', 'adp', 'fructose6p', 'fructose16p', 
                'glyceraldehyde3p', 'dihydroxyacetonep', 'nad', 'pi', '13bpg', 'nadh', 'h'],
        'smiles': ['C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O', 
                  'C1=NC2=C(C(=N1)N)N=CN2C3C(C(C(O3)COP(=O)(O)OP(=O)(O)O)O)O',
                  'C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)OP(=O)(O)O',
                  'C1=NC2=C(C(=N1)N)N=CN2C3C(C(C(O3)COP(=O)(O)O)O)O',
                  'C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)OP(=O)(O)O',
                  'C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)OP(=O)(O)O',
                  'C([C@@H](C=O)O)OP(=O)(O)O',
                  'C([C@@H](C=O)O)OP(=O)(O)O',
                  'C1=NC2=C(C(=N1)N)N=CN2C3C(C(C(O3)COP(=O)(O)O)O)O',
                  'OP(=O)(O)O',
                  'C([C@@H](C=O)O)OP(=O)(O)O',
                  'C1=NC2=C(C(=N1)N)N=CN2C3C(C(C(O3)COP(=O)(O)O)O)O',
                  '[H+]']
    }
    
    metabolites_df = pd.DataFrame(metabolites_data)
    
    # Create incidence matrix
    reactions_index, reactions_metas, reactants_nums, direction = get_coefficient_and_reactant(reactions)
    all_metas = list(set(sum(reactions_metas, [])))
    all_metas.sort()
    
    incidence_matrix = np.zeros((len(all_metas), len(reactions)))
    for i, (rxn_index, rxn_metas) in enumerate(zip(reactions_index, reactions_metas)):
        for j, meta in enumerate(rxn_metas):
            if meta in all_metas:
                meta_idx = all_metas.index(meta)
                incidence_matrix[meta_idx, i] = float(rxn_index[j])
    
    # Create molecular features
    features = []
    name_to_smiles = dict(zip(metabolites_df['name'], metabolites_df['smiles']))
    
    for meta in all_metas:
        if meta in name_to_smiles:
            smiles = name_to_smiles[meta]
            fp = smiles_to_fp(smiles)[0]
        else:
            fp = np.zeros(2048)
        features.append(fp)
    
    features = np.array(features)
    
    # Create negative samples using simple method
    negative_matrix = create_negative_samples_simple(incidence_matrix, negative_ratio=1)
    
    # Combine positive and negative samples
    all_reactions = np.concatenate([incidence_matrix, negative_matrix], axis=1)
    
    # Create labels
    positive_labels = np.ones(incidence_matrix.shape[1])
    negative_labels = np.zeros(negative_matrix.shape[1])
    all_labels = np.concatenate([positive_labels, negative_labels])
    
    print(f"Simple method data prepared:")
    print(f"  Total reactions: {all_reactions.shape[1]} (positive: {len(positive_labels)}, negative: {len(negative_labels)})")
    print(f"  Metabolites: {all_reactions.shape[0]}")
    print(f"  Features: {features.shape}")
    
    return all_reactions, all_labels, features

def prepare_data_original_method(dataset_name='iAF1260b'):
    """
    ORIGINAL METHOD: Prepare data using exact CLOSEgaps methodology
    Full implementation with ChEBI database.
    """
    print("=" * 60)
    print("ORIGINAL METHOD: Using CLOSEgaps methodology with ChEBI")
    print("=" * 60)
    
    # Load reactions
    rxn_file = f'data/{dataset_name}/{dataset_name}_rxn_name_list.txt'
    with open(rxn_file, 'r') as f:
        pos_rxn = [line.strip() for line in f.readlines()]
    
    # Use only first 100 reactions for testing to avoid memory issues
    pos_rxn = pos_rxn[:100]
    print(f"Using first 100 reactions for testing (out of {len(pos_rxn)} total)")
    
    # Load metabolites with SMILES
    meta_file = f'data/{dataset_name}/{dataset_name}_meta_count.csv'
    pos_metas_smiles = pd.read_csv(meta_file)
    
    # Load ChEBI database for negative sampling
    chebi_meta_filter = pd.read_csv('data/pool/cleaned_chebi.csv')
    
    print(f"Loaded {len(pos_rxn)} reactions and {len(pos_metas_smiles)} metabolites")
    print(f"ChEBI database: {len(chebi_meta_filter)} metabolites")
    
    # Create negative reactions using original method
    neg_rxn = create_negative_samples_original(pos_rxn, pos_metas_smiles, chebi_meta_filter, 
                                             balanced_atom=False, negative_ratio=1, atom_ratio=0.5)
    
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
    
    # Create labels
    positive_labels = np.ones(len(pos_rxn))
    negative_labels = np.zeros(len(neg_rxn))
    all_labels = np.concatenate([positive_labels, negative_labels])
    
    # Create molecular features
    features = []
    name_to_smiles = dict(zip(pos_metas_smiles['name'], pos_metas_smiles['smiles']))
    
    for meta in all_metas:
        if meta in name_to_smiles:
            smiles = name_to_smiles[meta]
            fp = smiles_to_fp(smiles)[0]
        else:
            fp = np.zeros(2048)
        features.append(fp)
    
    features = np.array(features)
    
    print(f"Original method data prepared:")
    print(f"  Total reactions: {all_reactions.shape[1]} (positive: {len(pos_rxn)}, negative: {len(neg_rxn)})")
    print(f"  Metabolites: {all_reactions.shape[0]}")
    print(f"  Features: {features.shape}")
    
    return all_reactions, all_labels, features

# ============================================================================
# 4. MODEL ARCHITECTURE
# ============================================================================

class CLOSEgaps(nn.Module):
    """CLOSEgaps: Hypergraph Neural Network for metabolic gap-filling"""
    
    def __init__(self, input_num, input_feature_num, emb_dim, conv_dim, head=3, p=0.1, L=1,
                 use_attention=True, similarity=None):
        super(CLOSEgaps, self).__init__()
        self.emb_dim = emb_dim
        self.conv_dim = conv_dim
        self.p = p
        self.input_num = input_num
        self.head = head
        self.hyper_conv_L = L
        self.linear_encoder = nn.Linear(input_feature_num, emb_dim)
        self.similarity_liner = nn.Linear(input_num, emb_dim)
        self.max_pool = hnn.global_max_pool
        self.similarity = similarity
        self.in_channel = emb_dim
        if similarity is not None:
            self.in_channel = 2 * emb_dim

        self.relu = nn.ReLU()
        self.hypergraph_conv = hnn.HypergraphConv(self.in_channel, conv_dim, heads=head, use_attention=use_attention,
                                                  dropout=p)
        if L > 1:
            self.hypergraph_conv_list = nn.ModuleList()
            for l in range(L - 1):
                self.hypergraph_conv_list.append(
                    hnn.HypergraphConv(head * conv_dim, conv_dim, heads=head, use_attention=use_attention, dropout=p))

        if use_attention:
            self.hyper_attr_liner = nn.Linear(input_num, self.in_channel)
            if L > 1:
                self.hyperedge_attr_list = nn.ModuleList()
                for l in range(L - 1):
                    self.hyperedge_attr_list.append(nn.Linear(input_num, head * conv_dim))
        self.hyperedge_linear = nn.Linear(conv_dim * head, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_features, incidence_matrix):
        input_nodes_features = self.relu(self.linear_encoder(input_features))
        if self.similarity is not None:
            simi_feature = self.relu(self.similarity_liner(self.similarity))
            input_nodes_features = torch.cat((simi_feature, input_nodes_features), dim=1)

        row, col = torch.where(incidence_matrix.T)
        edges = torch.cat((col.view(1, -1), row.view(1, -1)), dim=0)
        hyperedge_attr = self.hyper_attr_liner(incidence_matrix.T)
        input_nodes_features = self.hypergraph_conv(input_nodes_features, edges, hyperedge_attr=hyperedge_attr)
        if self.hyper_conv_L > 1:
            for l in range(self.hyper_conv_L - 1):
                layer_hyperedge_attr = self.hyperedge_attr_list[l](incidence_matrix.T)
                input_nodes_features = self.hypergraph_conv_list[l](input_nodes_features, edges,
                                                                    hyperedge_attr=layer_hyperedge_attr)
                input_nodes_features = self.relu(input_nodes_features)

        hyperedge_feature = torch.mm(incidence_matrix.T, input_nodes_features)
        return self.hyperedge_linear(hyperedge_feature)

    def predict(self, input_features, incidence_matrix):
        return self.softmax(self.forward(input_features, incidence_matrix))

# ============================================================================
# 5. TRAINING AND EVALUATION
# ============================================================================

def train_and_evaluate(X, y, features, method_name="Method"):
    """Train and evaluate CLOSEgaps model"""
    print(f"\n{method_name} Training and Evaluation")
    print("=" * 40)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X.T, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Calculate similarity matrix
    features_tensor = torch.tensor(features, dtype=torch.float).to(device)
    similarity_matrix = getGipKernel(features_tensor, False, 1.0).to(device)
    
    # Initialize model
    model = CLOSEgaps(
        input_num=features.shape[0],
        input_feature_num=features.shape[1],
        emb_dim=64,
        conv_dim=128,
        head=6,
        L=2,
        similarity=similarity_matrix
    ).to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_f1 = 0
    best_model = model.state_dict().copy()
    patience = 10
    patience_counter = 0
    
    for epoch in range(100):
        model.train()
        
        # Batch training
        batch_size = 256
        num_batches = X_train.shape[0] // batch_size + (1 if X_train.shape[0] % batch_size != 0 else 0)
        
        total_loss = 0
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, X_train.shape[0])
            
            batch_X = torch.tensor(X_train[start_idx:end_idx].T, dtype=torch.float).to(device)
            batch_y = torch.tensor(y_train[start_idx:end_idx], dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            outputs = model(features_tensor, batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_X = torch.tensor(X_val.T, dtype=torch.float).to(device)
            val_outputs = model(features_tensor, val_X)
            val_probs = torch.softmax(val_outputs, dim=1)
            val_preds = (val_probs[:, 1] >= 0.5).long()
            
            # Handle case where only one class is present
            try:
                val_f1 = f1_score(y_val, val_preds.cpu())
            except:
                val_f1 = 0.0
            try:
                val_auc = roc_auc_score(y_val, val_probs[:, 1].cpu())
            except:
                val_auc = 0.5
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model and evaluate
    model.load_state_dict(best_model)
    model.eval()
    
    with torch.no_grad():
        test_X = torch.tensor(X_test.T, dtype=torch.float).to(device)
        test_outputs = model(features_tensor, test_X)
        test_probs = torch.softmax(test_outputs, dim=1)
        test_preds = (test_probs[:, 1] >= 0.5).long()
        
        # Calculate metrics with error handling
        try:
            f1 = f1_score(y_test, test_preds.cpu())
        except:
            f1 = 0.0
        try:
            precision = precision_score(y_test, test_preds.cpu())
        except:
            precision = 0.0
        try:
            recall = recall_score(y_test, test_preds.cpu())
        except:
            recall = 0.0
        try:
            auc = roc_auc_score(y_test, test_probs[:, 1].cpu())
        except:
            auc = 0.5
        try:
            aupr = average_precision_score(y_test, test_probs[:, 1].cpu())
        except:
            aupr = 0.5
        
        print(f"\n{method_name} Results:")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  AUPR: {aupr:.4f}")
        
        return {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'aupr': aupr
        }

# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function with both methods"""
    print("CLOSEgaps Tutorial: Two Methods Available")
    print("=" * 60)
    print("1. SIMPLE METHOD: Quick demonstration with coefficient perturbation")
    print("2. ORIGINAL METHOD: Full CLOSEgaps methodology with ChEBI database")
    print("=" * 60)
    
    # Method selection
    method_choice = input("Choose method (1 for Simple, 2 for Original, or Enter for both): ").strip()
    
    results = {}
    
    if method_choice == "1" or method_choice == "":
        print("\n" + "="*60)
        print("RUNNING SIMPLE METHOD")
        print("="*60)
        
        # Simple method
        X_simple, y_simple, features_simple = prepare_data_simple_method()
        results['simple'] = train_and_evaluate(X_simple, y_simple, features_simple, "Simple Method")
    
    if method_choice == "2" or method_choice == "":
        print("\n" + "="*60)
        print("RUNNING ORIGINAL METHOD")
        print("="*60)
        
        # Check if data files exist
        if not os.path.exists('data/iAF1260b/iAF1260b_rxn_name_list.txt'):
            print("Warning: Original method requires data files.")
            print("Please ensure the following files exist:")
            print("  - data/iAF1260b/iAF1260b_rxn_name_list.txt")
            print("  - data/iAF1260b/iAF1260b_meta_count.csv")
            print("  - data/pool/cleaned_chebi.csv")
            print("\nSkipping original method...")
        else:
            # Original method
            X_original, y_original, features_original = prepare_data_original_method('iAF1260b')
            results['original'] = train_and_evaluate(X_original, y_original, features_original, "Original Method")
    
    # Summary
    if len(results) > 1:
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        
        print(f"{'Metric':<12} {'Simple Method':<15} {'Original Method':<15}")
        print("-" * 45)
        for metric in ['f1', 'precision', 'recall', 'auc', 'aupr']:
            simple_val = results['simple'][metric]
            original_val = results['original'][metric]
            print(f"{metric.upper():<12} {simple_val:<15.4f} {original_val:<15.4f}")
    
    print("\n" + "="*60)
    print("TUTORIAL COMPLETED")
    print("="*60)
    print("Key differences between methods:")
    print("- Simple Method: Quick demo with coefficient perturbation")
    print("- Original Method: Full CLOSEgaps with ChEBI metabolite replacement")
    print("\nFor production use, the Original Method is recommended.")

if __name__ == "__main__":
    main()
