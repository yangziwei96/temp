(venv) B20474@CTOLWE00010:/mnt/data1/Share/yang-intern/GEM-P$ # Check what Python packages are available through apt
apt search python3-pandas
apt search python3-scikit-learn
apt search python3-rdkit

# Check if any of these packages are already installed system-wide
python3 -c "import pandas; print('pandas available')" 2>/dev/null || echo "pandas not available"
python3 -c "import sklearn; print('sklearn available')" 2>/dev/null || echo "sklearn not available"
python3 -c "import rdkit; print('rdkit available')" 2>/dev/null || echo "rdkit not available"
ソート中... 完了
全文検索... 完了  
python3-pandas/jammy,jammy 1.3.5+dfsg-3 all
  data structures for "relational" or "labeled" data

python3-pandas-lib/jammy 1.3.5+dfsg-3 amd64
  low-level implementations and bindings for pandas

ソート中... 完了
全文検索... 完了  
ソート中... 完了
全文検索... 完了  
python3-rdkit/jammy 202109.2-1build1 amd64
  Collection of cheminformatics and machine-learning software

pandas not available
sklearn not available
rdkit not available
(venv) B20474@CTOLWE00010:/mnt/data1/Share/yang-intern/GEM-P$ 
