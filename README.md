(venv) B20474@CTOLWE00010:/mnt/data1/Share/yang-intern/GEM-P$ # Test HTTP connectivity to mirrors
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

curl -I http://127.0.0.1internal serversscikit|rdkit)"ready be cached
curl: (56) Recv failure: 接続が相手からリセットされました
curl: (56) Recv failure: 接続が相手からリセットされました
curl: (56) Recv failure: 接続が相手からリセットされました
curl: (56) Recv failure: 接続が相手からリセットされました
cloud_r_project_org_bin_linux_ubuntu.list  cudnn-local-ubuntu2204-8.6.0.163.list  google-cloud-sdk.list  mizilla.list
cuda-ubuntu2204-x86_64.list                docker.list                            microsoft-prod.list    nvidia-container-toolkit.list
# deb cdrom:[Ubuntu 22.04.1 LTS _Jammy Jellyfish_ - Release amd64 (20220809.1)]/ jammy main restricted

# See http://help.ubuntu.com/community/UpgradeNotes for how to upgrade to
# newer versions of the distribution.
deb http://jp.archive.ubuntu.com/ubuntu/ jammy main restricted
# deb-src http://jp.archive.ubuntu.com/ubuntu/ jammy main restricted

#  # Major bug fix updates produced after the final release of the
#  # distribution.
deb http://jp.archive.ubuntu.com/ubuntu/ jammy-updates main restricted
# deb-src http://jp.archive.ubuntu.com/ubuntu/ jammy-updates main restricted

#  # N.B. software from this repository is ENTIRELY UNSUPPORTED by the Ubuntu
#  # team. Also, please note that software in universe WILL NOT receive any
#  # review or updates from the Ubuntu security team.
deb http://jp.archive.ubuntu.com/ubuntu/ jammy universe
# deb-src http://jp.archive.ubuntu.com/ubuntu/ jammy universe
deb http://jp.archive.ubuntu.com/ubuntu/ jammy-updates universe
# deb-src http://jp.archive.ubuntu.com/ubuntu/ jammy-updates universe

#  # N.B. software from this repository is ENTIRELY UNSUPPORTED by the Ubuntu
#  # team, and may not be under a free licence. Please satisfy yourself as to
#  # your rights to use the software. Also, please note that software in
#  # multiverse WILL NOT receive any review or updates from the Ubuntu
#  # security team.
deb http://jp.archive.ubuntu.com/ubuntu/ jammy multiverse
# deb-src http://jp.archive.ubuntu.com/ubuntu/ jammy multiverse
deb http://jp.archive.ubuntu.com/ubuntu/ jammy-updates multiverse
# deb-src http://jp.archive.ubuntu.com/ubuntu/ jammy-updates multiverse

#  # N.B. software from this repository may not have been tested as
#  # extensively as that contained in the main release, although it includes
#  # newer versions of some applications which may provide useful features.
#  # Also, please note that software in backports WILL NOT receive any review
#  # or updates from the Ubuntu security team.
deb http://jp.archive.ubuntu.com/ubuntu/ jammy-backports main restricted universe multiverse
# deb-src http://jp.archive.ubuntu.com/ubuntu/ jammy-backports main restricted universe multiverse

deb http://security.ubuntu.com/ubuntu jammy-security main restricted
# deb-src http://security.ubuntu.com/ubuntu jammy-security main restricted
deb http://security.ubuntu.com/ubuntu jammy-security universe
# deb-src http://security.ubuntu.com/ubuntu jammy-security universe
deb http://security.ubuntu.com/ubuntu jammy-security multiverse
# deb-src http://security.ubuntu.com/ubuntu jammy-security multiverse

# This system was installed using small removable media
# (e.g. netinst, live or single CD). The matching "deb cdrom"
# entries were disabled at the end of the installation process.
# For information about how to configure apt package sources,
# see the sources.list(5) manual.
conda: コマンドが見つかりません
Requirement already satisfied: numpy in /mnt/data1/Share/yang-intern/venv/lib/python3.10/site-packages (2.1.2)
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
ERROR: Could not find a version that satisfies the requirement pandas (from versions: none)
ERROR: No matching distribution found for pandas
numpy                    2.1.2
torch                    2.7.1+cu118
torch_cluster            1.6.3+pt27cu118
torch_scatter            2.1.2+pt27cu118
torch_sparse             0.6.18+pt27cu118
torch_spline_conv        1.2.2+pt27cu118
torchaudio               2.7.1+cu118
torchvision              0.22.1+cu118
curl: (7) Failed to connect to localhost port 80 after 0 ms: 接続を拒否されました
curl: (7) Failed to connect to 127.0.0.1 port 80 after 0 ms: 接続を拒否されました
(venv) B20474@CTOLWE00010:/mnt/data1/Share/yang-intern/GEM-P$ # Try installing packages one by one, starting with the most basic
pip install numpy pandas scikit-learn tqdm matplotlib seaborn joblib

# Then try PyTorch packages separately
pip install torch
pip install torch-scatter torch-sparse
pip install torch-geometric

# Finally try the specialized packages
pip install rdkit cobra optlang
Requirement already satisfied: numpy in /mnt/data1/Share/yang-intern/venv/lib/python3.10/site-packages (2.1.2)
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/pandas/
ERROR: Could not find a version that satisfies the requirement pandas (from versions: none)
ERROR: No matching distribution found for pandas
Requirement already satisfied: torch in /mnt/data1/Share/yang-intern/venv/lib/python3.10/site-packages (2.7.1+cu118)
Requirement already satisfied: nvidia-cuda-cupti-cu11==11.8.87 in /mnt/data1/Share/yang-intern/venv/lib/python3.10/site-packages (from torch) (11.8.87)
Requirement already satisfied: nvidia-curand-cu11==10.3.0.86 in /mnt/data1/Share/yang-intern/venv/lib/python3.10/site-packages (from torch) (10.3.0.86)
Requirement already satisfied: nvidia-cusparse-cu11==11.7.5.86 in /mnt/data1/Share/yang-intern/venv/lib/python3.10/site-packages (from torch) (11.7.5.86)
Requirement already satisfied: fsspec in /mnt/data1/Share/yang-intern/venv/lib/python3.10/site-packages (from torch) (2024.6.1)
Requirement already satisfied: networkx in /mnt/data1/Share/yang-intern/venv/lib/python3.10/site-packages (from torch) (3.3)
Requirement already satisfied: typing-extensions>=4.10.0 in /mnt/data1/Share/yang-intern/venv/lib/python3.10/site-packages (from torch) (4.12.2)
Requirement already satisfied: nvidia-cuda-runtime-cu11==11.8.89 in /mnt/data1/Share/yang-intern/venv/lib/python3.10/site-packages (from torch) (11.8.89)
Requirement already satisfied: triton==3.3.1 in /mnt/data1/Share/yang-intern/venv/lib/python3.10/site-packages (from torch) (3.3.1)
Requirement already satisfied: filelock in /mnt/data1/Share/yang-intern/venv/lib/python3.10/site-packages (from torch) (3.13.1)
Requirement already satisfied: nvidia-cusolver-cu11==11.4.1.48 in /mnt/data1/Share/yang-intern/venv/lib/python3.10/site-packages (from torch) (11.4.1.48)
Requirement already satisfied: sympy>=1.13.3 in /mnt/data1/Share/yang-intern/venv/lib/python3.10/site-packages (from torch) (1.13.3)
Requirement already satisfied: nvidia-nccl-cu11==2.21.5 in /mnt/data1/Share/yang-intern/venv/lib/python3.10/site-packages (from torch) (2.21.5)
Requirement already satisfied: nvidia-cudnn-cu11==9.1.0.70 in /mnt/data1/Share/yang-intern/venv/lib/python3.10/site-packages (from torch) (9.1.0.70)
Requirement already satisfied: nvidia-cuda-nvrtc-cu11==11.8.89 in /mnt/data1/Share/yang-intern/venv/lib/python3.10/site-packages (from torch) (11.8.89)
Requirement already satisfied: nvidia-cublas-cu11==11.11.3.6 in /mnt/data1/Share/yang-intern/venv/lib/python3.10/site-packages (from torch) (11.11.3.6)
Requirement already satisfied: nvidia-nvtx-cu11==11.8.86 in /mnt/data1/Share/yang-intern/venv/lib/python3.10/site-packages (from torch) (11.8.86)
Requirement already satisfied: nvidia-cufft-cu11==10.9.0.58 in /mnt/data1/Share/yang-intern/venv/lib/python3.10/site-packages (from torch) (10.9.0.58)
Requirement already satisfied: jinja2 in /mnt/data1/Share/yang-intern/venv/lib/python3.10/site-packages (from torch) (3.1.4)
Requirement already satisfied: setuptools>=40.8.0 in /mnt/data1/Share/yang-intern/venv/lib/python3.10/site-packages (from triton==3.3.1->torch) (59.6.0)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /mnt/data1/Share/yang-intern/venv/lib/python3.10/site-packages (from sympy>=1.13.3->torch) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in /mnt/data1/Share/yang-intern/venv/lib/python3.10/site-packages (from jinja2->torch) (2.1.5)
Requirement already satisfied: torch-scatter in /mnt/data1/Share/yang-intern/venv/lib/python3.10/site-packages (2.1.2+pt27cu118)
Requirement already satisfied: torch-sparse in /mnt/data1/Share/yang-intern/venv/lib/python3.10/site-packages (0.6.18+pt27cu118)
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/scipy/
WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/scipy/
WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/scipy/
WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/scipy/
WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/scipy/
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/torch-sparse/
WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/torch-sparse/
WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/torch-sparse/
WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/torch-sparse/
WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/torch-sparse/
INFO: pip is looking at multiple versions of torch-scatter to determine which version is compatible with other requirements. This could take a while.
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/torch-scatter/
WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/torch-scatter/
WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/torch-scatter/
WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/torch-scatter/
WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/torch-scatter/
ERROR: Could not find a version that satisfies the requirement scipy (from torch-sparse) (from versions: none)
ERROR: No matching distribution found for scipy
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/torch-geometric/
WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/torch-geometric/
WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/torch-geometric/
WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/torch-geometric/
WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/torch-geometric/
ERROR: Could not find a version that satisfies the requirement torch-geometric (from versions: none)
ERROR: No matching distribution found for torch-geometric
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/rdkit/
WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/rdkit/
WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/rdkit/
WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/rdkit/
WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(104, '接続が相手からリセットされました'))': /simple/rdkit/
ERROR: Could not find a version that satisfies the requirement rdkit (from versions: none)
ERROR: No matching distribution found for rdkit
(venv) B20474@CTOLWE00010:/mnt/data1/Share/yang-intern/GEM-P$ 
