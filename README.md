Got it! Here's the **best single solution** with code you can run directly on your lab server:

## Best Solution: Try Alternative Pip Mirrors

Run these commands on your lab server in order until one works:

```bash
# Try Tsinghua mirror (most reliable in Asia)
pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn torch>=1.12.1 torch-geometric>=2.1.0 torch-scatter>=2.0.9 torch-sparse>=0.6.15 pandas>=1.3.5 numpy>=1.21.5 scikit-learn>=1.0.2 rdkit>=2022.03.5 tqdm>=4.62.1 matplotlib>=3.5.0 seaborn>=0.11.0 cobra>=0.22.1 optlang>=1.5.2 joblib>=1.2.0
```

If that fails, try:

```bash
# Try Aliyun mirror
pip install --index-url https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com torch>=1.12.1 torch-geometric>=2.1.0 torch-scatter>=2.0.9 torch-sparse>=0.6.15 pandas>=1.3.5 numpy>=1.21.5 scikit-learn>=1.0.2 rdkit>=2022.03.5 tqdm>=4.62.1 matplotlib>=3.5.0 seaborn>=0.11.0 cobra>=0.22.1 optlang>=1.5.2 joblib>=1.2.0
```

If both fail, try:

```bash
# Try with all trusted hosts
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.tuna.tsinghua.edu.cn --trusted-host mirrors.aliyun.com torch>=1.12.1 torch-geometric>=2.1.0 torch-scatter>=2.0.9 torch-sparse>=0.6.15 pandas>=1.3.5 numpy>=1.21.5 scikit-learn>=1.0.2 rdkit>=2022.03.5 tqdm>=4.62.1 matplotlib>=3.5.0 seaborn>=0.11.0 cobra>=0.22.1 optlang>=1.5.2 joblib>=1.2.0
```

Start with the first command - it has the highest success rate for servers in Asia.
