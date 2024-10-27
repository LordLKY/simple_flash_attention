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

batch_size = 8
n_head = 8
seq_len = 512
head_embd = 64

q = torch.randn(batch_size, n_head, 1, head_embd).cuda()  # decoding mode, seq_len = 1
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

def flash_attn(q, k, v):
    k = k.view(batch_size, n_head, seq_len, head_embd // 4, 4)
    # reshape k to (batch_size, n_head, head_embd // 4, seq_len, 4) for better locality
    k = k.permute(0, 1, 3, 2, 4).contiguous()
    # reshape v to (batch_size, n_head, head_embd, seq_len) for better locality
    v = v.transpose(-2, -1).contiguous()
    return minimal_attn.forward(q, k, v)

normal_attn_result = manual_attn(q, k, v)
flash_attn_result = flash_attn(q, k, v)
print('attn values sanity check:', torch.allclose(flash_attn_result, normal_attn_result, rtol=0, atol=1e-02))

manual_attn(q, k, v)  # warmup
start_time = time.perf_counter()
for i in range(10):
    manual_attn(q, k, v)
end_time = time.perf_counter()
print(f'normal attention time: {end_time - start_time}s')

minimal_attn.forward(q, k, v)  # warmup
start_time = time.perf_counter()
for i in range(10):
    minimal_attn.forward(q, k, v)
end_time = time.perf_counter()
print(f'flash attention time: {end_time - start_time}s')
