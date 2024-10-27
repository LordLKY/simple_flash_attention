#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define FLOAT4(pointer) reinterpret_cast<float4 *>(pointer)

__global__
void block_forward(float* Q, float* K, float* V, const int N, const int d,
                    const float softmax_scale, float* mid_O, float *mid_r) {
    int tx = threadIdx.x; int wid = tx % 32;
    int bx = blockIdx.x; int by = blockIdx.y; int bz = blockIdx.z;
    int B_S = blockDim.x;

    extern __shared__ float sram[];
    float *Q_s = sram;
    float *S = &sram[d];

    int q_offset = (bx * gridDim.y + by) * d;
    int k_offset = (bx * gridDim.y + by) * d * N + bz * B_S * 4;
    int mid_o_offset = (bx * gridDim.y + by) * d * gridDim.z + bz * d;
    int mid_r_offset = (bx * gridDim.y + by) * 2 * gridDim.z + bz;

    // Load Q into SRAM
    for(int i = tx; i < d; i += B_S){
        Q_s[i] = *(Q + q_offset + i);
    }

    __syncthreads();

    // Calculate S
    // with overlapping (hope it will work)
    float4 vec0, vec1, vec2;
    int iter = 0;
    float dot = 0.0;
    vec1 = FLOAT4(K + k_offset)[tx];
    for(; iter < d / 4 - 1; iter++) {
        vec0 = FLOAT4(Q_s)[iter];
        if(iter & 1){
            vec1 = FLOAT4(K + k_offset + (iter + 1) * N * 4)[tx];
            dot += vec2.x * vec0.x + vec2.y * vec0.y + vec2.z * vec0.z + vec2.w * vec0.w;
        } else {
            vec2 = FLOAT4(K + k_offset + (iter + 1) * N * 4)[tx];
            dot += vec1.x * vec0.x + vec1.y * vec0.y + vec1.z * vec0.z + vec1.w * vec0.w;
        }
    }
    vec0 = FLOAT4(Q_s)[iter];
    if(iter & 1){
        dot += vec2.x * vec0.x + vec2.y * vec0.y + vec2.z * vec0.z + vec2.w * vec0.w;
    } else {
        dot += vec1.x * vec0.x + vec1.y * vec0.y + vec1.z * vec0.z + vec1.w * vec0.w;
    }
    dot *= softmax_scale;

    // __syncthreads(); // cub will do this

    // Reduce to calculate softmax
    using BlockReduce = cub::BlockReduce<float, 256>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float max_element = BlockReduce(temp_storage).Reduce(dot, cub::Max());
    if(tx == 0){mid_r[mid_r_offset] = max_element;}
    __syncthreads();
    S[tx] = dot; 
    S[tx] = __expf(dot - mid_r[mid_r_offset]);
    float exp_element = S[tx];
    // __syncthreads(); // cub will do this
    float sum = BlockReduce(temp_storage).Sum(exp_element);
    if(tx == 0){mid_r[mid_r_offset + gridDim.z] = sum;}
    __syncthreads();
    S[tx] /= mid_r[mid_r_offset + gridDim.z];

    // Calculate O
    int t_v_offset = (bx * gridDim.y + by) * d * N + N * (tx / 32) + bz * B_S;
    for(int k = 0; k < d; k += (B_S / 32), t_v_offset += (B_S / 32 * N)){
        __syncthreads();
        dot = 0.0;
        for(int i = wid; i < B_S; i += 32){
            dot += S[i] * V[t_v_offset + i];
        }

        __syncthreads();
        int j = 16;
        while(j >= 1){
            dot += __shfl_xor_sync(0xffffffff, dot, j, 32);
            j = j >> 1;
        }
        if(wid == 0){mid_O[mid_o_offset + k + tx / 32] = dot;}
    }
}

__global__
void block_reduce(float *mid_r, float *mid_O, const int N, const int d, const int N_P_S,
                  float *O){
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;
    int mid_o_offset = (bx * gridDim.y + by) * d * N_P_S;
    int mid_r_offset = (bx * gridDim.y + by) * 2 * N_P_S;

    extern __shared__ float sram[];
    float *O_s = sram;

    float max_element = mid_r[mid_r_offset];
    float sum = mid_r[mid_r_offset + N_P_S];
    if(tx < d){O_s[tx] = mid_O[mid_o_offset + tx];}

    // online softmax
    for(int i = 1; i < N_P_S; i++){
        float prev_max_element = max_element;
        float prev_sum = sum;
        max_element = max(mid_r[mid_r_offset + i], max_element);
        sum = (__expf(-max_element + prev_max_element) * prev_sum) + \
            (__expf(-max_element + mid_r[mid_r_offset + i]) * mid_r[mid_r_offset + i + N_P_S]);
        if(tx < d){
            O_s[tx] = (prev_sum / sum) * O_s[tx] * __expf(-max_element + prev_max_element) + \
                (mid_r[mid_r_offset + i + N_P_S] / sum) * mid_O[mid_o_offset + i * d + tx] * \
                __expf(-max_element + mid_r[mid_r_offset + i]);
        }
    }

    if(tx < d){O[(bx * gridDim.y + by) * d + tx] = O_s[tx];}
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // in decoding-phase, we can assume that Tr = Br = 1
    // and there is no need to use Bc/Tc
    // for simplicity, a thread deals with one dot(Q_row, K_col)

    // Q: [B, nh, 1, d]
    // K: [B, nh, d/4, N, 4]
    // V: [B, nh, d, N]

    // B: batch size / nh: number of heads / N: sequence length / d: dimension of each head
    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = V.size(3); const int d = V.size(2);

    const int BLOCK_SIZE = 256; // aka num_threads_per_block
    const int N_PARALLEL_SIZE = N / BLOCK_SIZE;

    const float softmax_scale = 1.0 / sqrt(d);

    // Initialize O to HBM
    auto O = torch::zeros_like(Q);
    torch::Device device(torch::kCUDA);

    // middle result (max & sum of the partitions)
    float *mid_r = nullptr, *mid_O = nullptr;
    cudaMalloc(&mid_O, sizeof(float) * B * nh * d * N_PARALLEL_SIZE);
    cudaMalloc(&mid_r, sizeof(float) * B * nh * N_PARALLEL_SIZE * 2);

    // Calculate SRAM size needed per block
    // K, V doesn't need to be stored in SRAM since they are accessed only once
    const int sram_size1 = (d * sizeof(float)) + (BLOCK_SIZE * sizeof(float));
    dim3 grid_dim1(B, nh, N_PARALLEL_SIZE);  // batch_size x num_heads
    dim3 block_dim1(BLOCK_SIZE);  // Bc threads per block

    block_forward<<<grid_dim1, block_dim1, sram_size1>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, softmax_scale, mid_O, mid_r
    );

    // reduce the max and sum of the partitions
    const int sram_size2 = d * sizeof(float);
    dim3 grid_dim2(B, nh);
    dim3 block_dim2(d);

    block_reduce<<<grid_dim2, block_dim2, sram_size2>>>(
        mid_r, mid_O, N, d, N_PARALLEL_SIZE, O.data_ptr<float>()
    );

    // free the memory
    cudaFree(mid_O);
    cudaFree(mid_r);
    return O;
}