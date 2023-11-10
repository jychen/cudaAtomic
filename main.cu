#include <stdio.h>
#include <stdint.h>

#define CUDA_OK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void do_atomics_kernel(uint32_t *counters, uint32_t ncounters, uint32_t nops) {
    uint32_t i = (blockDim.x * blockIdx.x + threadIdx.x) % ncounters;
    int p = 1;
    while (nops--) {
        p = atomicAdd(counters + i, p);
        if (++i >= ncounters) {
            i = 0;
        }
    }
}

void test_atomic_throughput(size_t ncounters, uint32_t nthreads, uint32_t nops) {
    uint32_t *d_counters;
    cudaEvent_t start_event, stop_event;
    float time_ms;

    CUDA_OK(cudaMalloc(&d_counters, 4 * ncounters));
    CUDA_OK(cudaMemset(d_counters, 0, 4 * ncounters));
    CUDA_OK(cudaEventCreate(&start_event));
    CUDA_OK(cudaEventCreate(&stop_event));
    CUDA_OK(cudaEventRecord(start_event));
    do_atomics_kernel<<<(nthreads - 1)/1024 + 1, 1024>>>(d_counters, ncounters, nops);
    CUDA_OK(cudaEventRecord(stop_event));
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaEventElapsedTime(&time_ms, start_event, stop_event));
    printf("Counters %ld, Threads %d, Ops %d, Time(ms) %f, Throughtput %f\n", 
        ncounters, nthreads, nops, time_ms, (float)nthreads * (float)nops / time_ms);
    CUDA_OK(cudaFree(d_counters));
    CUDA_OK(cudaEventDestroy(start_event));
    CUDA_OK(cudaEventDestroy(stop_event));
}

int main() {
    test_atomic_throughput(1, 68*2048, 2048);
    test_atomic_throughput(1, 68*2048, 1024*64);
    test_atomic_throughput(8, 68*2048, 1024*64);
    test_atomic_throughput(32, 68*2048, 1024*64);
    test_atomic_throughput(256, 68*2048, 1024*64);
    test_atomic_throughput(1024, 68*2048, 1024*64);
    test_atomic_throughput(1024*16, 68*2048, 1024*64);
    test_atomic_throughput(1024*1024, 68*2048, 1024*64);
    test_atomic_throughput(1024*1024*256, 68*2048, 1024*64);
    test_atomic_throughput(1024*1024*1024, 68*2048, 1024*64);
    return 0;
}