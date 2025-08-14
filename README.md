Great! The packages are available in the Ubuntu repositories, but you can't use sudo. Since you're in a virtual environment, let's try a different approach:

## Solution: Install packages without sudo using pip with system packages

Try installing the packages using pip but pointing to the system Python packages:

```bash
# Try installing packages that might be available in your environment
pip install --no-deps pandas scikit-learn matplotlib seaborn tqdm joblib

# Or try installing from system Python
python3 -m pip install --user pandas scikit-learn matplotlib seaborn tqdm joblib
```

## Alternative: Check if packages are already available in your virtual environment

```bash
# Check what's already installed in your venv
pip list | grep -E "(pandas|sklearn|matplotlib|seaborn|tqdm|joblib|rdkit)"

# Try importing packages to see if they're already available
python3 -c "import pandas; print('pandas version:', pandas.__version__)" 2>/dev/null || echo "pandas not in venv"
python3 -c "import sklearn; print('sklearn version:', sklearn.__version__)" 2>/dev/null || echo "sklearn not in venv"
python3 -c "import matplotlib; print('matplotlib version:', matplotlib.__version__)" 2>/dev/null || echo "matplotlib not in venv"
```

## If that doesn't work, try this workaround:

```bash
# Try installing with different pip options
pip install --no-cache-dir --no-deps pandas
pip install --no-cache-dir --no-deps scikit-learn
pip install --no-cache-dir --no-deps matplotlib
```

Try the first command (`pip install --no-deps pandas scikit-learn matplotlib seaborn tqdm joblib`) and see if it works without requiring external downloads.
