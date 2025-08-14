The issue is that your firewall is blocking **all external package repository connections**, not just HTTPS. Let me give you more diagnostic tests to find a working solution:

## Additional Diagnostic Tests

Run these commands on your lab server:

```bash
# Test HTTP connectivity to mirrors
curl -I http://pypi.tuna.tsinghua.edu.cn
curl -I http://mirrors.aliyun.com

# Test if you can reach any external servers
curl -I http://www.baidu.com
curl -I http://www.google.com

# Check if there's a local package repository
ls /etc/apt/sources.list.d/
cat /etc/apt/sources.list

# Check if conda is available
which conda
conda --version

# Test if you can install basic packages that might already be cached
pip install --no-deps numpy
pip install --no-deps pandas

# Check what packages are already installed
pip list | grep -E "(torch|numpy|pandas|scikit|rdkit)"

# Test if you can reach internal servers
curl -I http://localhost
curl -I http://127.0.0.1
```

## Alternative Installation Strategy

If all external connections are blocked, try this approach:

```bash
# Try installing packages one by one, starting with the most basic
pip install numpy pandas scikit-learn tqdm matplotlib seaborn joblib

# Then try PyTorch packages separately
pip install torch
pip install torch-scatter torch-sparse
pip install torch-geometric

# Finally try the specialized packages
pip install rdkit cobra optlang
```

Run the diagnostic tests first and tell me the results - this will help determine if there's any way to get packages from external sources or if we need to use a completely offline approach.
