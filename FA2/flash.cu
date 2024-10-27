#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* O) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    // Offset into Q,K,V,O - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh

    // Define SRAM for Q,K,V,O,S
    extern __shared__ float sram[];
    int tile_size_q = Br * d;  // size of Qi, Oi
    int tile_size_kv = Bc * d;  // size of Kj, Vj
    float* Qi = sram;
    float* Kj = &sram[tile_size_q];
    float* Vj = &sram[tile_size_q + tile_size_kv];
    float* Oi = &sram[tile_size_q + (2 * tile_size_kv)];
    float* S = &sram[(2 * tile_size_q) + (2 * tile_size_kv)];

    for(int i = 0; i < Tr; i++){
        // Load Qi to SRAM
        for (int x = 0; x < d; x++) {
            Qi[(tx * d) + x] = Q[qkv_offset + (tile_size_q * i) + (tx * d) + x];
        }
        float row_m_prev = -INFINITY;
        float row_l_prev = 0;

        for(int j = 0; j < Tc; j++){
            // Load Kj, Vj to SRAM
            for(int k = tx; k < Bc; k += blockDim.x){
                for (int x = 0; x < d; x++) {
                    Kj[(k * d) + x] = K[qkv_offset + (tile_size_kv * j) + (k * d) + x];
                    Vj[(k * d) + x] = V[qkv_offset + (tile_size_kv * j) + (k * d) + x];
                }
            }
            __syncthreads();

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // Compute new Oi
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                Oi[(tx * d) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * Oi[(tx * d) + x]) \
                    + (__expf(row_m - row_m_new) * pv));
            }
            row_m_prev = row_m_new;
            row_l_prev = row_l_new;
        }

        // write Oi to HBM
        for(int x = 0; x < d; x++){
            O[qkv_offset + (tile_size_q * i) + (tx * d) + x] = Oi[(tx * d) + x];
        }
        __syncthreads();
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // best condition is: Bc == Br
    const int Bc = 32; const int Br = 32;

    // B: batch size / nh: number of heads / N: sequence length / d: dimension of each head
    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);

    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Initialize O to HBM
    auto O = torch::zeros_like(Q);
    torch::Device device(torch::kCUDA);

    // Calculate SRAM size needed per block
    const int sram_size = (2 * Bc * d * sizeof(float)) + (2 * Br * d * sizeof(float)) + (2 * Bc * Br * sizeof(float));
    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(Br);  // Bc threads per block

    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        O.data_ptr<float>()
    );
    return O;
}