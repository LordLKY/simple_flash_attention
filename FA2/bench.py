import math
import time
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load
from pathlib import Path

# Load the CUDA kernel as a python module
cpp_source = Path(__file__).parent / "main.cpp"
cu_sorce = Path(__file__).parent / "flash.cu"
minimal_attn = load(name='minimal_attn', sources=[cpp_source, cu_sorce], extra_cuda_cflags=['-O2'])

# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 16
n_head = 12
seq_len = 64
head_embd = 64

q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

# Our minimal flash attention aims to be faster than this by avoiding HBM read/writes of N^2 matrices.
def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

normal_attn_result = manual_attn(q, k, v)
flash_attn_result = minimal_attn.forward(q, k, v)
print('attn values sanity check:', torch.allclose(flash_attn_result, normal_attn_result, rtol=0, atol=1e-02))

manual_attn(q, k, v)  # warmup
start_time = time.perf_counter()
for i in range(20):
    manual_attn(q, k, v)
end_time = time.perf_counter()
print(f'normal attention time: {end_time - start_time}s')

minimal_attn.forward(q, k, v)  # warmup
start_time = time.perf_counter()
for i in range(20):
    minimal_attn.forward(q, k, v)
end_time = time.perf_counter()
print(f'flash attention time: {end_time - start_time}s')
