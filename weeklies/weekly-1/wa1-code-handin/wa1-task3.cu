#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include "helper.h"

#define GPU_RUNS 100

__global__ void funcKernel(float* X, float *Y) {
    const unsigned int gid = threadIdx.x+256*blockIdx.x;
    Y[gid] = pow((X[gid]/(X[gid]-2.3)), 3);
}

int main(int argc, char** argv) {
    unsigned int N;
    
    { // reading the number of elements 
      if (argc != 1) { 
        printf("Num Args is: %d instead of 0. Exiting!\n", argc); 
        exit(1);
      }
    }

    N = 100000000;

    // use the first CUDA device:
    cudaSetDevice(0);

    unsigned int mem_size = N*sizeof(float);

    // allocate host memory
    float* h_in  = (float*) malloc(mem_size);
    float* h_out = (float*) malloc(mem_size);
    float* h_out2 = (float*) malloc(mem_size);

    // initialize the memory
    for(unsigned int i=0; i<N; ++i) {
        h_in[i] = (float)(i+1);
    }

    // allocate device memory
    float* d_in;
    float* d_out;
    cudaMalloc((void**)&d_in,  mem_size);
    cudaMalloc((void**)&d_out, mem_size);

    // copy host memory to device
    cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);
  
    { // execute the kernel a number of times;
      // to measure performance use a large N, e.g., 200000000,
      // and increase GPU_RUNS to 100 or more. 
    
        double elapsed; struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        for(int n = 0; n < N; n++) {
            float tmp_n = (float)(n+1);
            h_out2[n] = pow((tmp_n/(tmp_n-2.3)), 3);
        }
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (1.0 * (t_diff.tv_sec*1e6+t_diff.tv_usec));
        double gigabytespersec = (2.0 * N * 4.0) / (elapsed * 1000.0);
        printf("CPU: The kernel took on average %f microseconds. GB/sec: %f \n", elapsed, gigabytespersec);
    }

    { 
        const int blockSize = 256;
        const int blockCount = (N+blockSize-1)/blockSize;
        double elapsed; struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL);

        for(int r = 0; r < GPU_RUNS; r++) {
            funcKernel<<< blockCount, blockSize>>>(d_in, d_out);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (1.0 * (t_diff.tv_sec*1e6+t_diff.tv_usec)) / GPU_RUNS;
        double gigabytespersec = (2.0 * N * 4.0) / (elapsed * 1000.0);
        printf("GPU: The kernel took on average %f microseconds. GB/sec: %f \n", elapsed, gigabytespersec);
    }
        
    // check for errors
    gpuAssert( cudaPeekAtLastError() );

    // copy result from ddevice to host
    cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

    // print result

    for(unsigned int i=0; i<N; ++i) {
        float actual   = h_out2[i];
        float expected = h_out[i]; 
        if( abs(actual)-abs(expected) > 0.001 ) {
            printf("Invalid result at index %d, actual: %f, expected: %f. \n", i, actual, expected);
            exit(3);
        }
    }
    printf("Successful Validation.\n");

    // clean-up memory
    free(h_in);       free(h_out);
    cudaFree(d_in);   cudaFree(d_out);
}
