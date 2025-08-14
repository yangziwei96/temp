The connection reset errors indicate your lab server has strict firewall/proxy restrictions that are actively blocking pip connections. Here's a **diagnostic test** to understand your network situation:

## Network Diagnostic Test

Run these commands on your lab server to test connectivity:

```bash
# Test basic connectivity
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
echo $https_proxy
echo $HTTP_PROXY
echo $HTTPS_PROXY

# Test with different pip options
pip install --verbose --index-url https://pypi.tuna.tsinghua.edu.cn/simple/ numpy
```

## Most Likely Solutions Based on Results:

**If ping works but curl fails**: You need to configure pip to use a proxy
```bash
pip install --proxy http://your-proxy:port package_name
```

**If all external connections fail**: Your server is completely isolated - you'll need to download packages locally and transfer them

**If some mirrors work**: Use the working mirror for all installations

Run the diagnostic test and tell me the results - this will tell us exactly what's blocking your connections and how to work around it.
