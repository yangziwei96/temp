#!/usr/bin/env python3
"""
CLOSEgaps Validation Setup Test

This script tests the validation setup to ensure all components work correctly.
It verifies data loading, model instantiation, and basic functionality.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append('..')

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from CLOSEgaps import CLOSEgaps
        print("PASS: CLOSEgaps imported successfully")
    except ImportError as e:
        print(f"FAIL: Failed to import CLOSEgaps: {e}")
        return False
    
    try:
        from utils import smiles_to_fp, getGipKernel
        print("PASS: utils imported successfully")
    except ImportError as e:
        print(f"FAIL: Failed to import utils: {e}")
        return False
    
    try:
        from sklearn.metrics import f1_score, roc_auc_score
        print("PASS: sklearn metrics imported successfully")
    except ImportError as e:
        print(f"FAIL: Failed to import sklearn: {e}")
        return False
    
    return True

def test_data_loading():
    """Test if datasets can be loaded"""
    print("\nTesting data loading...")
    
    datasets = ['iAF1260b', 'iAF692', 'iJO1366', 'iMM904']
    available_datasets = []
    
    for dataset in datasets:
        try:
            # Check if reaction file exists
            rxn_file = f'../data/{dataset}/{dataset}_rxn_name_list.txt'
            if not os.path.exists(rxn_file):
                print(f"FAIL: Missing reaction file: {rxn_file}")
                continue
            
            # Check if metabolite file exists
            meta_file = f'../data/{dataset}/{dataset}_meta_count.csv'
            if not os.path.exists(meta_file):
                print(f"FAIL: Missing metabolite file: {meta_file}")
                continue
            
            # Try to load data
            with open(rxn_file, 'r') as f:
                reactions = [line.strip() for line in f.readlines()]
            
            metabolites_df = pd.read_csv(meta_file)
            
            print(f"PASS: {dataset}: {len(reactions)} reactions, {len(metabolites_df)} metabolites")
            available_datasets.append(dataset)
            
        except Exception as e:
            print(f"FAIL: Error loading {dataset}: {e}")
    
    return available_datasets

def test_model_instantiation():
    """Test if CLOSEgaps model can be instantiated"""
    print("\nTesting model instantiation...")
    
    try:
        from CLOSEgaps import CLOSEgaps
        
        # Create dummy data
        n_metabolites = 100
        n_features = 2048
        
        # Create dummy features
        features = torch.randn(n_metabolites, n_features)
        
        # Create dummy similarity matrix
        similarity = torch.randn(n_metabolites, n_metabolites)
        
        # Instantiate model
        model = CLOSEgaps(
            input_num=n_metabolites,
            input_feature_num=n_features,
            emb_dim=64,
            conv_dim=128,
            head=6,
            L=2,
            similarity=similarity
        )
        
        print(f"PASS: Model instantiated successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        dummy_input = torch.randn(n_metabolites, 10)  # 10 reactions
        with torch.no_grad():
            output = model(features, dummy_input)
        
        print(f"PASS: Forward pass successful, output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"FAIL: Model instantiation failed: {e}")
        return False

def test_utils_functions():
    """Test utility functions"""
    print("\nTesting utility functions...")
    
    try:
        from utils import smiles_to_fp, getGipKernel
        
        # Test SMILES to fingerprint
        test_smiles = "CCO"  # Ethanol
        fp = smiles_to_fp(test_smiles)
        print(f"PASS: SMILES to fingerprint: {fp.shape}")
        
        # Test GIP kernel
        features = torch.randn(50, 2048)
        kernel = getGipKernel(features, True, 1.0)
        print(f"PASS: GIP kernel: {kernel.shape}")
        
        return True
        
    except Exception as e:
        print(f"FAIL: Utility functions failed: {e}")
        return False

def test_data_processing():
    """Test data processing functions"""
    print("\nTesting data processing...")
    
    try:
        # Import processing functions from cross_validation
        from cross_validation import get_coefficient_and_reactant, create_incidence_matrix_from_reactions
        
        # Test reaction parsing
        test_reactions = [
            "2 glucose + 2 ATP => 2 glucose-6-phosphate + 2 ADP",
            "pyruvate + NADH <=> lactate + NAD+"
        ]
        
        reactions_index, reactions_metas, reactants_nums, direction = get_coefficient_and_reactant(test_reactions)
        print(f"PASS: Reaction parsing: {len(reactions_index)} reactions parsed")
        
        # Test incidence matrix creation
        metabolites_df = pd.DataFrame({
            'name': ['glucose', 'ATP', 'glucose-6-phosphate', 'ADP', 'pyruvate', 'NADH', 'lactate', 'NAD+'],
            'smiles': ['C([C@@H]1[C@H]([C@@H](C(O1)O)O)O)O'] * 8  # Dummy SMILES
        })
        
        # Get all unique metabolites from reactions
        all_metas = list(set(['glucose', 'ATP', 'glucose-6-phosphate', 'ADP', 'pyruvate', 'NADH', 'lactate', 'NAD+']))
        all_metas.sort()
        
        incidence_matrix = create_incidence_matrix_from_reactions(test_reactions, all_metas)
        print(f"PASS: Incidence matrix: {incidence_matrix.shape}")
        
        return True
        
    except Exception as e:
        print(f"FAIL: Data processing failed: {e}")
        return False

def main():
    """Run all tests"""
    print("CLOSEgaps Validation Setup Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\nFAIL: Import test failed. Please check dependencies.")
        return
    
    # Test data loading
    available_datasets = test_data_loading()
    if not available_datasets:
        print("\nFAIL: No datasets available. Please check data directory.")
        return
    
    # Test model instantiation
    if not test_model_instantiation():
        print("\nFAIL: Model instantiation failed.")
        return
    
    # Test utility functions
    if not test_utils_functions():
        print("\nFAIL: Utility functions failed.")
        return
    
    # Test data processing
    if not test_data_processing():
        print("\nFAIL: Data processing failed.")
        return
    
    print("\n" + "=" * 50)
    print("PASS: All tests passed! Validation setup is ready.")
    print(f"Available datasets: {', '.join(available_datasets)}")
    print("\nYou can now run:")
    print("  python train_models.py")
    print("  python cross_validation.py")

if __name__ == "__main__":
    main()
