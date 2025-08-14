Perfect! The diagnostic shows exactly what's happening:

## Analysis:
- ✅ **Basic connectivity works** (ping to 8.8.8.8 succeeds)
- ✅ **DNS resolution works** (nslookup resolves pypi.org)
- ❌ **HTTPS connections are blocked** (all curl HTTPS requests fail with SSL connection reset)
- ✅ **pip can access some packages** (numpy is already installed)

## Root Cause:
Your lab server has a **firewall/proxy that blocks HTTPS connections** but allows basic internet connectivity. This is a common security policy.

## Solution: Use HTTP Instead of HTTPS

Try this command on your lab server:

```bash
pip install --index-url http://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn torch>=1.12.1 torch-geometric>=2.1.0 torch-scatter>=2.0.9 torch-sparse>=0.6.15 pandas>=1.3.5 numpy>=1.21.5 scikit-learn>=1.0.2 rdkit>=2022.03.5 tqdm>=4.62.1 matplotlib>=3.5.0 seaborn>=0.11.0 cobra>=0.22.1 optlang>=1.5.2 joblib>=1.2.0
```

If that doesn't work, try:

```bash
pip install --index-url http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com torch>=1.12.1 torch-geometric>=2.1.0 torch-scatter>=2.0.9 torch-sparse>=0.6.15 pandas>=1.3.5 numpy>=1.21.5 scikit-learn>=1.0.2 rdkit>=2022.03.5 tqdm>=4.62.1 matplotlib>=3.5.0 seaborn>=0.11.0 cobra>=0.22.1 optlang>=1.5.2 joblib>=1.2.0
```

The key is using **HTTP instead of HTTPS** - your firewall is likely blocking HTTPS but allowing HTTP traffic.
