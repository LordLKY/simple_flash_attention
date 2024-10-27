# Simple Flash-Attention
Simple version of Flash-Attention(include V1&amp;V2&amp;Flash-Decoding)

## About

This is a simple version of classic Flash-Attention series, including FA-V1, FA-V2 and Flash-Decoding. Both of them are implemented in about 100 lines of CUDA (only forward). The code is simple and easy to understand. It may have a long way to go to reach the remarkable efficiency as official FA but it's suitable for those who starts learning FA and struggles to master CUDA (like me).Besides it contains Flash-Decoding which is a better choice in decoding-phase of LLM.

Flash-Attention V1/V2 is widely used in the training or prefilling phase of LLM. The skectch map of FA-V1 is shown below.

![sketch map of FA-V1](https://github.com/LordLKY/simple_flash_attention/blob/main/asset/2.png)

But in decoding-phase (when the sequence length of Q is very small, 1 for instance), Flash-Decoding is more reasonable since it does parallel computing on the dimension of sequence length. The [kernel of vLLM](https://github.com/vllm-project/vllm/blob/main/csrc/attention/attention_kernels.cu) is based on it.

Thanks to [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal) (FA-V1 in this repo is almost same as it) which inspires me to build this repo.

## Build & Usage

### Requirement

- CUDA 11.x or 12.x
- pytorch
- Ninja (for cross compilation)

```bash
git clone https://github.com/LordLKY/simple_flash_attention.git
cd FlashDecoding (or FA1/FA2)
python bench.py
```

All 3 directories have the same architecture (flash.cu: kernel, main.cpp: function registration, bench.py: test & benchmark).

## Test

3 types of FA is compared with native pytorch implementation of MHA (on RTX 4060).

### Prefill
With Q/K/V (batch_size=16, n_head=12, seq_len=64, d=64), FA-V2 is about 7 times faster than pytorch MHA while FA-V1 is much slower (by exchanging the order of loops, FA-V2 is far more efficient and resonable than FA-V1).

### Decode
With Q (batch_size=8, n_head=8, seq_len=1, d=64) and K/V(seq_len=512), Flash-Decoding is about 9 times faster than pytorch MHA. Considerable speedup can be achieved as seq_len of K/V increase.

## Notice

- The code is not optimized for extreme performance. It's just a simple version for learning
- Only some regular seq_length & embedded-dimension are supported
- Only did rough benchmark

### Reference
- [FlashAttention V1](https://arxiv.org/abs/2205.14135)
- [FlashAttention V2](https://arxiv.org/abs/2307.08691)
- [FlashDecoding](https://pytorch.org/blog/flash-decoding/)
- [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal)