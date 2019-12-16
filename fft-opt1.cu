// One of the possible optimizations to the implementation in fft.cu
// 
// The algorithm is as follows:
//
// for (s = 0 to log2(N)) ........ loop1
//      m = 2^i;
//      statements
//      for (j=0;j<N;j+=m) ....... loop2
//          for (k=0 to m/2) ..... loop3
//              statements
//
// The second and third loop are complementary to each other in the sense that 
// while loop2 runs for n/m values of j, loop3 runs for m/2 values of k. With 
// larger values of m, loop3 does more work, while for smaller values of m, loop2
// does more work, so in this optimization, we plan to separate out the two cases
// and achieve parallelization of both outer and inner loops
//
// The point of separation needs to be experimentally arrived at, i.e, we need to 
// test out different cases and choose the best of the lot. As a first commit in
// the optimization, we have adopted a naive approach and check the value of m/2.
// If the value of m/2 is less than the maximum number of threads that can be 
// spawned by a block, we parallelize loop2, and loop3 otherwise.
//
//

#include <stdio.h>
#include <cmath>
#include <cuda.h>

typedef float2 Complex;

#define THREADS 32
#define MAX_NO_OF_THREADS_PER_BLOCK 1024

const long long ARRAY_SIZE = 65536; 
const long long ARRAY_BYTES = ARRAY_SIZE * sizeof(Complex);

// Bit reversal re-ordering, first step of FFT
__global__ void bit_reverse_reorder(Complex *d_rev, Complex *d_a, int s) {
	  int id = blockIdx.x * blockDim.x + threadIdx.x;
    int rev = __brev(id) >> (32-s);

    if(id < ARRAY_SIZE)
        d_rev[rev] = d_a[id];
}

// Work of the innermost loop, common to both parallelization
__device__ void inplace_fft(Complex *a, int j, int k, int m){
    
    if (j+k+m/2 < ARRAY_SIZE){
        
        Complex w, t, u;

        // w^k (w is root of unity)
        w.x = __cosf((2*M_PI*k)/m);
        w.y = -__sinf((2*M_PI*k)/m);

        // u = a[j+k]
        u.x = a[j+k].x;
        u.y = a[j+k].y;

        // t = w*a[j+k+m/2];
        t.x = w.x*a[j+k+m/2].x - w.y*a[j+k+m/2].y;
        t.y = w.x*a[j+k+m/2].y + w.y*a[j+k+m/2].x;

        // a[j+k] = u+t;
        a[j+k].x = u.x + t.x;
        a[j+k].y = u.y + t.y;

        // a[j+k+m/2] = u-t;
        a[j+k+m/2].x = u.x - t.x;
        a[j+k+m/2].y = u.y - t.y;

    }
}

// Parallelization of loop2
__global__ void fft_outer(Complex *a, int m){
    int j = (blockIdx.x * blockDim.x + threadIdx.x)*m;
    if (j < ARRAY_SIZE){
        for (int k=0;k<m/2;k++){
            inplace_fft(a,j,k,m);
        }
    }    
}

// Parallelization of loop3
__global__ void fft_inner(Complex *a, int j, int m){
    int k = (blockIdx.x * blockDim.x + threadIdx.x);
    if (k < m/2)
        inplace_fft(a,j,k,m);
}

float magnitude(float2 a)
{
    return sqrt(a.x*a.x + a.y*a.y);
}

int main() {
    // Creating files to write output to
    FILE *fptr;
    fptr = fopen("fft-opt1-output.dat", "wr");

    // Host arrays for input and output
    Complex h_a[ARRAY_SIZE];
    Complex h_rev[ARRAY_SIZE];

    // Input signal, complex part remains zero.
    // Signal is of the form sin(2*M_PI*f*x/N) or cos(2*M_PI*f*x/N)
    // N is sample size and is always a power of 2 
    for(int i = 0; i < ARRAY_SIZE; i++) {
    	h_a[i].x = cos((6.0*M_PI*i)/ARRAY_SIZE);
        h_a[i].y = 0.0;
    }
        
    // No of bits required to represent N
    int s = (int)ceil(log2(ARRAY_SIZE));

    // Device arrays
    Complex *d_a, *d_rev;

    // Memory Allocation
    cudaMalloc((void**) &d_a, ARRAY_BYTES);
    cudaMalloc((void**) &d_rev, ARRAY_BYTES);

    // Copy host array to device
    cudaMemcpy(d_a, h_a, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // First step in FFT, bit-reverse reordering
    // Swap an element at index 'i', with the element 
    // which is the bit-string reversal of 'i' 
    bit_reverse_reorder<<<(ARRAY_SIZE+THREADS-1)/THREADS, THREADS>>>(d_rev, d_a, s);
    
    cudaDeviceSynchronize();

    // FFT driver code
    for (int i=1;i<=s;i++){

        // m = 2^i
        int m = 1 << i;

        if (m/2 < MAX_NO_OF_THREADS_PER_BLOCK){

            fft_outer<<<((ARRAY_SIZE/m)+THREADS-1)/THREADS,THREADS>>>(d_rev,m);    

        } else {
            
            for (int j=0;j<ARRAY_SIZE;j+=m){

                fft_inner<<<((m/2)+THREADS-1)/THREADS,THREADS>>>(d_rev,j,m);

            }
        }
    }

    // Copy result array from device to host
    cudaMemcpy(h_rev,d_rev,ARRAY_BYTES,cudaMemcpyDeviceToHost);

    // Writing output to files
    fprintf(fptr, "i\t\ta.magn\t\ta.real\t\t\ta.img\n");
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        fprintf(fptr,"%d\t\t%f\t\t%f\t\t%f\n", i, magnitude(h_rev[i]), h_rev[i].x, h_rev[i].y);
    }

    // Free allocated device memory
    cudaFree(d_a);
    cudaFree(d_rev);

    return 0;
}