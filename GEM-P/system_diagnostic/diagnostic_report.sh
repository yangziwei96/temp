#!/bin/bash

echo "=== GEM-P Package Installation Diagnostic Report ==="
echo "Generated on: $(date)"
echo "User: $(whoami)"
echo "Hostname: $(hostname)"
echo ""

echo "=== Python Environment ==="
echo "Python version: $(python3 --version)"
echo "Pip version: $(pip --version)"
echo "Virtual environment: $VIRTUAL_ENV"
echo ""

echo "=== Network Connectivity Tests ==="
echo "Ping to 8.8.8.8:"
ping -c 3 8.8.8.8 2>/dev/null | tail -2 || echo "Ping failed"
echo ""

echo "DNS resolution for pypi.org:"
nslookup pypi.org 2>/dev/null | head -5 || echo "DNS lookup failed"
echo ""

echo "HTTPS connectivity tests:"
curl -I https://pypi.org 2>/dev/null | head -1 || echo "HTTPS to pypi.org: FAILED"
curl -I https://pypi.tuna.tsinghua.edu.cn 2>/dev/null | head -1 || echo "HTTPS to Tsinghua mirror: FAILED"
echo ""

echo "HTTP connectivity tests:"
curl -I http://pypi.tuna.tsinghua.edu.cn 2>/dev/null | head -1 || echo "HTTP to Tsinghua mirror: FAILED"
echo ""

echo "=== Package Installation Tests ==="
echo "Testing pip install with different mirrors:"
echo "1. Default PyPI:"
timeout 30 pip install pandas 2>&1 | head -15 || echo "Default PyPI: TIMEOUT or FAILED"
echo ""

echo "2. Tsinghua mirror (HTTPS):"
timeout 30 pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple/ pandas 2>&1 | head -15 || echo "Tsinghua HTTPS: TIMEOUT or FAILED"
echo ""

echo "3. Tsinghua mirror (HTTP):"
timeout 30 pip install --index-url http://pypi.tuna.tsinghua.edu.cn/simple/ pandas 2>&1 | head -15 || echo "Tsinghua HTTP: TIMEOUT or FAILED"
echo ""

echo "4. Aliyun mirror:"
timeout 30 pip install --index-url http://mirrors.aliyun.com/pypi/simple/ pandas 2>&1 | head -15 || echo "Aliyun: TIMEOUT or FAILED"
echo ""

echo "5. With trusted hosts:"
timeout 30 pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org pandas 2>&1 | head -15 || echo "Trusted hosts: TIMEOUT or FAILED"
echo ""

echo "6. Conda test (if available):"
timeout 30 conda install pandas 2>&1 | head -15 2>/dev/null || echo "Conda: NOT AVAILABLE"
echo ""

echo "=== GPU and CUDA Information ==="
echo "CUDA version: $(nvcc --version 2>/dev/null | grep 'release' | cut -d' ' -f6 || echo 'CUDA not found')"
echo "GPU devices: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 'No GPU found')"
echo "Number of GPUs: $(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo '0')"
echo ""

echo "=== PyTorch CUDA Support ==="
python3 -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null || echo "PyTorch CUDA test failed"
echo ""

echo "=== Memory Information ==="
echo "System RAM: $(free -h | grep Mem | awk '{print $2}')"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 'N/A')"
echo ""

echo "=== Storage Information ==="
echo "Available disk space: $(df -h /mnt/data1/Share/yang-intern/ | tail -1 | awk '{print $4}')"
echo "Current directory space: $(du -sh . | cut -f1)"
echo ""

echo "=== Current Package Status ==="
echo "Installed packages in virtual environment:"
pip list | grep -E "(torch|numpy|pandas|sklearn|matplotlib|rdkit|cobra)" || echo "No relevant packages found"
echo ""

echo "=== System Package Availability ==="
echo "Available system packages:"
apt search python3-pandas 2>/dev/null | head -3 || echo "apt search failed"
apt search python3-scikit-learn 2>/dev/null | head -3 || echo "apt search failed"
apt search python3-rdkit 2>/dev/null | head -3 || echo "apt search failed"
echo ""

echo "=== Required Packages Missing ==="
python3 -c "import pandas; print('pandas: OK')" 2>/dev/null || echo "pandas: MISSING"
python3 -c "import sklearn; print('scikit-learn: OK')" 2>/dev/null || echo "scikit-learn: MISSING"
python3 -c "import matplotlib; print('matplotlib: OK')" 2>/dev/null || echo "matplotlib: MISSING"
python3 -c "import seaborn; print('seaborn: OK')" 2>/dev/null || echo "seaborn: MISSING"
python3 -c "import rdkit; print('rdkit: OK')" 2>/dev/null || echo "rdkit: MISSING"
python3 -c "import cobra; print('cobra: OK')" 2>/dev/null || echo "cobra: MISSING"
python3 -c "import optlang; print('optlang: OK')" 2>/dev/null || echo "optlang: MISSING"
python3 -c "import tqdm; print('tqdm: OK')" 2>/dev/null || echo "tqdm: MISSING"
python3 -c "import joblib; print('joblib: OK')" 2>/dev/null || echo "joblib: MISSING"
python3 -c "import torch_geometric; print('torch-geometric: OK')" 2>/dev/null || echo "torch-geometric: MISSING"
echo ""

echo "=== System Information ==="
echo "OS: $(lsb_release -d 2>/dev/null | cut -f2 || uname -a)"
echo "Architecture: $(uname -m)"
echo "Kernel: $(uname -r)"
echo ""

echo "=== User Permissions ==="
echo "Can use sudo: $(sudo -n true 2>/dev/null && echo 'YES' || echo 'NO')"
echo "User groups: $(groups)"
echo ""

echo "=== Package Manager Status ==="
echo "apt update test:"
sudo -n apt update 2>/dev/null | head -3 || echo "apt update failed (requires sudo)"
echo ""

echo "=== Network Proxy Information ==="
echo "HTTP_PROXY: $HTTP_PROXY"
echo "HTTPS_PROXY: $HTTPS_PROXY"
echo "http_proxy: $http_proxy"
echo "https_proxy: $https_proxy"
echo ""

echo "=== Additional Package Details ==="
echo "PyTorch Geometric version needed: >=2.1.0"
echo "Current torch version: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Unknown')"
echo ""

echo "=== Alternative Installation Methods ==="
echo "Conda available: $(which conda 2>/dev/null && echo 'YES' || echo 'NO')"
echo "pip config location: $(pip config list 2>/dev/null | head -1 || echo 'No pip config')"
echo ""

echo "=== End of Report ==="