venv) B20474@CTOLWE00010:/mnt/data1/Share/yang-intern/GEM-P$ # Try installing packages that might be available in your environment
pip install --no-deps pandas scikit-learn matplotlib seaborn tqdm joblib

# Or try installing from system Python
python3 -m pip install --user pandas scikit-learn matplotlib seaborn tqdm joblib
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
ERROR: Could not find a version that satisfies the requirement pandas (from versions: none)
ERROR: No matching distribution found for pandas
ERROR: Can not perform a '--user' install. User site-packages are not visible in this virtualenv.
(venv) B20474@CTOLWE00010:/mnt/data1/Share/yang-intern/GEM-P$ # Check what's already installed in your venv
pip list | grep -E "(pandas|sklearn|matplotlib|seaborn|tqdm|joblib|rdkit)"

# Try importing packages to see if they're already available
python3 -c "import pandas; print('pandas version:', pandas.__version__)" 2>/dev/null || echo "pandas not in venv"
python3 -c "import sklearn; print('sklearn version:', sklearn.__version__)" 2>/dev/null || echo "sklearn not in venv"
python3 -c "import matplotlib; print('matplotlib version:', matplotlib.__version__)" 2>/dev/null || echo "matplotlib not in venv"
pandas not in venv
sklearn not in venv
matplotlib not in venv
(venv) B20474@CTOLWE00010:/mnt/data1/Share/yang-intern/GEM-P$ # Try installing with different pip options
pip install --no-cache-dir --no-deps pandas
pip install --no-cache-dir --no-deps scikit-learn
pip install --no-cache-dir --no-deps matplotlib
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
ERROR: Could not find a version that satisfies the requirement pandas (from versions: none)
ERROR: No matching distribution found for pandas
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/scikit-learn/
WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/scikit-learn/
WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/scikit-learn/
WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/scikit-learn/
WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/scikit-learn/
ERROR: Could not find a version that satisfies the requirement scikit-learn (from versions: none)
ERROR: No matching distribution found for scikit-learn
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/matplotlib/
WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/matplotlib/
WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/matplotlib/
WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/matplotlib/
WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/matplotlib/
ERROR: Could not find a version that satisfies the requirement matplotlib (from versions: none)
ERROR: No matching distribution found for matplotlib
(venv) B20474@CTOLWE00010:/mnt/data1/Share/yang-intern/GEM-P$ 
