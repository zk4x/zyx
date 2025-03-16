# How to optimize GPU compute kernels in zyx?

We can classify kernels into three categories:
1. kernels without loops (only global and local work size) - these are rare (due to kernelizer being good at fusion) and simple to optimize. For now just having work size search is enough.
2. kernels with large global work size (roughly >32)
    This includes matmul and convolution
3. kernels with small global work size (about <32)
    These are pure reduce kernels
