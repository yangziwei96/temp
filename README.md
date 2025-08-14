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
echo ""

echo "=== System Information ==="
echo "OS: $(lsb_release -d 2>/dev/null | cut -f2 || uname -a)"
echo "Arc
