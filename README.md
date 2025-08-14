(venv) B20474@CTOLWE00010:/mnt/data1/Share/yang-intern/GEM-P$ # Test basic connectivity
ping -c 3 8.8.8.8

# Test DNS resolution
nslookup pypi.org

# Test HTTP connectivity to different mirrors
curl -I https://pypi.org
curl -I https://pypi.tuna.tsinghua.edu.cn
curl -I https://mirrors.aliyun.com

# Test pip configuration
pip config list

# Check if you're behind a proxy
echo $http_proxy
pip install --verbose --index-url https://pypi.tuna.tsinghua.edu.cn/simple/ numpy
PING 8.8.8.8 (8.8.8.8) 56(84) bytes of data.
64 bytes from 8.8.8.8: icmp_seq=1 ttl=112 time=2.18 ms
64 bytes from 8.8.8.8: icmp_seq=2 ttl=112 time=2.24 ms
64 bytes from 8.8.8.8: icmp_seq=3 ttl=112 time=2.29 ms

--- 8.8.8.8 ping statistics ---
3 packets transmitted, 3 received, 0% packet loss, time 2003ms
rtt min/avg/max/mdev = 2.177/2.235/2.293/0.047 ms
Server:         127.0.0.53
Address:        127.0.0.53#53

Non-authoritative answer:
Name:   pypi.org
Address: 151.101.64.223
Name:   pypi.org
Address: 151.101.128.223
Name:   pypi.org
Address: 151.101.192.223
Name:   pypi.org
Address: 151.101.0.223
Name:   pypi.org
Address: 2a04:4e42:200::223
Name:   pypi.org
Address: 2a04:4e42:400::223
Name:   pypi.org
Address: 2a04:4e42::223
Name:   pypi.org
Address: 2a04:4e42:600::223

curl: (35) OpenSSL SSL_connect: 接続が相手からリセットされました in connection to pypi.org:443 
curl: (35) OpenSSL SSL_connect: 接続が相手からリセットされました in connection to pypi.tuna.tsinghua.edu.cn:443 
curl: (35) OpenSSL SSL_connect: 接続が相手からリセットされました in connection to mirrors.aliyun.com:443 




Using pip 22.0.2 from /mnt/data1/Share/yang-intern/venv/lib/python3.10/site-packages/pip (python 3.10)
Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple/
Requirement already satisfied: numpy in /mnt/data1/Share/yang-intern/venv/lib/python3.10/site-packages (2.1.2)
(venv) B20474@CTOLWE00010:/mnt/data1/Share/yang-intern/GEM-P$ 
