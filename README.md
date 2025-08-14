=== GEM-P Package Installation Diagnostic Report ===
Generated on: 2025年  8月 14日 木曜日 13:58:22 JST
User: B20474
Hostname: CTOLWE00010

=== Python Environment ===
Python version: Python 3.10.6
Pip version: pip 22.0.2 from /mnt/data1/Share/yang-intern/venv/lib/python3.10/site-packages/pip (python 3.10)
Virtual environment: /mnt/data1/Share/yang-intern/venv

=== Network Connectivity Tests ===
Ping to 8.8.8.8:
3 packets transmitted, 3 received, 0% packet loss, time 2003ms
rtt min/avg/max/mdev = 2.045/2.093/2.133/0.036 ms

DNS resolution for pypi.org:
Server:		127.0.0.53
Address:	127.0.0.53#53

Non-authoritative answer:
Name:	pypi.org

HTTPS connectivity tests:

HTTP connectivity tests:

=== Package Installation Tests ===
Testing pip install with different mirrors:
1. Default PyPI:
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
ERROR: Could not find a version that satisfies the requirement pandas (from versions: none)
ERROR: No matching distribution found for pandas

2. Tsinghua mirror (HTTPS):
Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple/
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
ERROR: Could not find a version that satisfies the requirement pandas (from versions: none)
ERROR: No matching distribution found for pandas

3. Tsinghua mirror (HTTP):
Looking in indexes: http://pypi.tuna.tsinghua.edu.cn/simple/
WARNING: The repository located at pypi.tuna.tsinghua.edu.cn is not a trusted or secure host and is being ignored. If this repository is available via HTTPS we recommend you use HTTPS instead, otherwise you may silence this warning and allow it anyway with '--trusted-host pypi.tuna.tsinghua.edu.cn'.
ERROR: Could not find a version that satisfies the requirement pandas (from versions: none)
ERROR: No matching distribution found for pandas

4. Aliyun mirror:
Looking in indexes: http://mirrors.aliyun.com/pypi/simple/
WARNING: The repository located at mirrors.aliyun.com is not a trusted or secure host and is being ignored. If this repository is available via HTTPS we recommend you use HTTPS instead, otherwise you may silence this warning and allow it anyway with '--trusted-host mirrors.aliyun.com'.
ERROR: Could not find a version that satisfies the requirement pandas (from versions: none)
ERROR: No matching distribution found for pandas

5. With trusted hosts:
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
ERROR: Could not find a version that satisfies the requirement pandas (from versions: none)
ERROR: No matching distribution found for pandas

6. Conda test (if available):
./diagnostic_report.sh: 行 56: conda: コマンドが見つかりません

=== GPU and CUDA Information ===
CUDA version: V11.7.99
GPU devices: NVIDIA GeForce RTX 4080
Number of GPUs: 1

=== PyTorch CUDA Support ===
PyTorch CUDA available: True
CUDA version: 11.8
GPU count: 1

=== Memory Information ===
System RAM: 125Gi
GPU Memory: 16376

=== Storage Information ===
Available disk space: 1.6T
Current directory space: 155M

=== Current Package Status ===
Installed packages in virtual environment:
numpy                    2.1.2
torch                    2.7.1+cu118
torch_cluster            1.6.3+pt27cu118
torch_scatter            2.1.2+pt27cu118
torch_sparse             0.6.18+pt27cu118
torch_spline_conv        1.2.2+pt27cu118
torchaudio               2.7.1+cu118
torchvision              0.22.1+cu118

=== System Package Availability ===
Available system packages:
ソート中...
全文検索...
python3-pandas/jammy,jammy 1.3.5+dfsg-3 all
ソート中...
全文検索...
ソート中...
全文検索...
python3-rdkit/jammy 202109.2-1build1 amd64

=== Required Packages Missing ===
pandas: MISSING
scikit-learn: MISSING
matplotlib: MISSING
seaborn: MISSING
rdkit: MISSING
cobra: MISSING
optlang: MISSING
tqdm: MISSING
joblib: MISSING

=== System Information ===
OS: Ubuntu 22.04.1 LTS
Architecture: x86_64
Kernel: 5.15.0-58-generic

=== User Permissions ===
Can use sudo: NO
User groups: B20474 techrg shared-users

=== Package Manager Status ===
apt update test:

=== Network Proxy Information ===
HTTP_PROXY: 
HTTPS_PROXY: 
http_proxy: 
https_proxy: 

=== Additional Package Details ===
PyTorch Geometric version needed: >=2.1.0
Current torch version: 2.7.1+cu118

=== Alternative Installation Methods ===
Conda available: NO
pip config location: 

=== End of Report ===
