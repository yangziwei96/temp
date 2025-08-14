I see the issue - your lab server has no internet connectivity (100% packet loss to pypi.org). This is a common situation in corporate/academic environments with strict firewall policies. Let me help you install the requirements using alternative methods.

```bash
pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn torch>=1.12.1 torch-geometric>=2.1.0 torch-scatter>=2.0.9 torch-sparse>=0.6.15 pandas>=1.3.5 numpy>=1.21.5 scikit-learn>=1.0.2 rdkit>=2022.03.5 tqdm>=4.62.1 matplotlib>=3.5.0 seaborn>=0.11.0 cobra>=0.22.1 optlang>=1.5.2 joblib>=1.2.0
```
Let me try a different approach. Since you have no internet connectivity, let's check what packages are already available in your environment and what we can work with:

```bash
pip list
```
Let me try using conda instead, which might have better connectivity or different mirror settings:

```bash
conda install -c conda-forge torch torch-geometric torch-scatter torch-sparse pandas numpy scikit-learn rdkit tqdm matplotlib seaborn cobra optlang joblib
```
Since you're in a restricted network environment, here are the most practical solutions:

## Option 1: Use Alternative Package Indexes
Try these commands one by one:

```bash
# Try Tsinghua mirror (often works in China/Asia)
pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple/ -r requirements.txt

# Try Aliyun mirror
pip install --index-url https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt

# Try Douban mirror
pip install --index-url https://pypi.douban.com/simple/ -r requirements.txt
```

## Option 2: Install with Trusted Hosts
```bash
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.tuna.tsinghua.edu.cn -r requirements.txt
```

## Option 3: Use Conda with Different Channels
```bash
# Try conda-forge channel
conda install -c conda-forge torch torch-geometric rdkit cobra

# Try pytorch channel for PyTorch packages
conda install -c pytorch torch torch-geometric torch-scatter torch-sparse
```

## Option 4: Manual Download and Transfer
If none of the above work, you'll need to:

1. **On a machine with internet access** (your local machine):
   ```bash
   pip download -r requirements.txt -d ./packages/
   ```

2. **Transfer the packages folder** to your lab server (via USB, network share, etc.)

3. **On your lab server**:
   ```bash
   pip install ./packages/*.whl
   ```

## Option 5: Check if Packages are Already Available
Some lab servers have local package repositories. Try:
```bash
# Check if there's a local pip configuration
pip config list

# Check if conda has different channels configured
conda config --show channels
```

## Option 6: Minimal Installation
If you can't get all packages, try installing the most critical ones first:
```bash
pip install numpy pandas scikit-learn
```

Would you like me to help you try any of these specific approaches? Also, do you have access to:
1. A local machine with internet where you could download packages?
2. Any shared network drives or USB access to transfer files?
3. System administrator access to configure proxy settings?
