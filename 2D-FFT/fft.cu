#include <stdio.h>
#include <cmath>
#include <cuda.h>

typedef float2 Complex;

#define THREADS 32
#define MAX_NO_OF_THREADS_PER_BLOCK 1024

// TODO: Make sure the implementations works for non square matrices
const long long ARRAY_SIZE = 1024; 
const long long ARRAY_BYTES = ARRAY_SIZE * sizeof(Complex);

// Bit reversal re-ordering, first step of FFT
__global__ void bit_reverse_reorder_row(Complex *d_rev, int i, Complex *d_a, int s) {
	  int id = blockIdx.x * blockDim.x + threadIdx.x;
    int rev = __brev(id) >> (32-s);

    if(id < ARRAY_SIZE)
        d_rev[i][rev] = d_a[i][id];
}

// Work of the innermost loop, common to both parallelization
__device__ void inplace_fft_row(Complex *a, int i, int j, int k, int m){
    
    if (j+k+m/2 < ARRAY_SIZE){
        
        Complex w, t, u;

        // w^k (w is root of unity)
        w.x = __cosf((2*M_PI*k)/m);
        w.y = -__sinf((2*M_PI*k)/m);

        // u = a[j+k]
        u.x = a[i][j+k].x;
        u.y = a[i][j+k].y;

        // t = w*a[j+k+m/2];
        t.x = w.x*a[i][j+k+m/2].x - w.y*a[i][j+k+m/2].y;
        t.y = w.x*a[i][j+k+m/2].y + w.y*a[i][j+k+m/2].x;

        // a[j+k] = u+t;
        a[i][j+k].x = u.x + t.x;
        a[i][j+k].y = u.y + t.y;

        // a[j+k+m/2] = u-t;
        a[i][j+k+m/2].x = u.x - t.x;
        a[i][j+k+m/2].y = u.y - t.y;

    }
}

// Parallelization of loop2
__global__ void fft_outer_row(Complex *a, int i, int m){
    int j = (blockIdx.x * blockDim.x + threadIdx.x)*m;
    if (j < ARRAY_SIZE){
        for (int k=0;k<m/2;k++){
            inplace_fft(a,i,j,k,m);
        }
    }    
}

// Parallelization of loop3
__global__ void fft_inner_row(Complex *a, int i, int j, int m){
    int k = (blockIdx.x * blockDim.x + threadIdx.x);
    if (k < m/2)
        inplace_fft(a,i,j,k,m);
}

// TODO: Check for possibilities for reuse or modification of code to work for row and column

// Bit reversal re-ordering, first step of FFT
__global__ void bit_reverse_reorder_column(Complex *d_rev, int i, Complex *d_a, int s) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
  int rev = __brev(id) >> (32-s);

  if(id < ARRAY_SIZE)
      d_rev[rev][i] = d_a[id][i];
}

// Work of the innermost loop, common to both parallelization
__device__ void inplace_fft_column(Complex *a, int i, int j, int k, int m){
  
  if (j+k+m/2 < ARRAY_SIZE){
      
      Complex w, t, u;

      // w^k (w is root of unity)
      w.x = __cosf((2*M_PI*k)/m);
      w.y = -__sinf((2*M_PI*k)/m);

      // u = a[j+k]
      u.x = a[j+k][i].x;
      u.y = a[j+k][i].y;

      // t = w*a[j+k+m/2];
      t.x = w.x*a[j+k+m/2][i].x - w.y*a[j+k+m/2][i].y;
      t.y = w.x*a[j+k+m/2][i].y + w.y*a[j+k+m/2][i].x;

      // a[j+k] = u+t;
      a[j+k][i].x = u.x + t.x;
      a[j+k][i].y = u.y + t.y;

      // a[j+k+m/2] = u-t;
      a[j+k+m/2][i].x = u.x - t.x;
      a[j+k+m/2][i].y = u.y - t.y;

  }
}

// Parallelization of loop2
__global__ void fft_outer_column(Complex *a, int i, int m){
  int j = (blockIdx.x * blockDim.x + threadIdx.x)*m;
  if (j < ARRAY_SIZE){
      for (int k=0;k<m/2;k++){
          inplace_fft_column(a,i,j,k,m);
      }
  }    
}

// Parallelization of loop3
__global__ void fft_inner_column(Complex *a, int i, int j, int m){
  int k = (blockIdx.x * blockDim.x + threadIdx.x);
  if (k < m/2)
      inplace_fft_column(a,i,j,k,m);
}

float magnitude(float2 a)
{
    return sqrt(a.x*a.x + a.y*a.y);
}

int main() {

    //Measuring performance
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Creating files to write output to
    FILE *fptr;
    fptr = fopen("fft-opt1-output.dat", "wr");

    // Host arrays for input and output
    Complex h_a[ARRAY_SIZE][ARRAY_SIZE];
    Complex h_rev[ARRAY_SIZE][ARRAY_SIZE];

    // Input signal, complex part remains zero.
    // Signal is of the form sin(2*M_PI*f*x/N) or cos(2*M_PI*f*x/N)
    // N is sample size and is always a power of 2 
    for(int i = 0; i < ARRAY_SIZE; i++) {
        for(int j = 0; j < ARRAY_SIZE; j++)
        {
            h_a[i][j].x = cos((6.0*M_PI*i)/ARRAY_SIZE);
            h_a[i][j].y = 0.0;
        }
    }
        
    // No of bits required to represent N
    int s = (int)ceil(log2(ARRAY_SIZE));

    // Device arrays
    Complex *d_a, *d_rev;

    // Memory Allocation
    cudaMalloc((void**) &d_a, ARRAY_BYTES * ARRAY_BYTES);
    cudaMalloc((void**) &d_rev, ARRAY_BYTES * ARRAY_BYTES);

    // Copy host array to device
    cudaMemcpy(d_a, h_a, ARRAY_BYTES * ARRAY_BYTES, cudaMemcpyHostToDevice);

    //Start of performance measurement
    cudaEventRecord(start);

    // First step in FFT, bit-reverse reordering
    // Swap an element at index 'i', with the element 
    // which is the bit-string reversal of 'i' 
    for(int i = 0; i < ARRAY_SIZE; i++)
        bit_reverse_reorder_row<<<((ARRAY_BYTES * ARRAY_BYTES)+THREADS-1)/THREADS, THREADS>>>(d_rev, i, d_a, s);
    
    cudaDeviceSynchronize();

    // FFT driver code for row wise FFT
    for(int i = 0; i < ARRAY_SIZE; i++)
    {
        for (int j=1;j<=s;j++){

            // m = 2^j
            int m = 1 << j;

            if (m/2 < MAX_NO_OF_THREADS_PER_BLOCK){

                fft_outer_row<<<((ARRAY_SIZE/m)+THREADS-1)/THREADS,THREADS>>>(d_rev,i,m);    

            } else {
                
                for (int k=0;k<ARRAY_SIZE;k+=m){

                    fft_inner_row<<<((m/2)+THREADS-1)/THREADS,THREADS>>>(d_rev,i,k,m);

                }
            }
        }
    }

    for(int i = 0; i < ARRAY_SIZE; i++)
        bit_reverse_reorder_column<<<((ARRAY_BYTES * ARRAY_BYTES)+THREADS-1)/THREADS, THREADS>>>(d_rev, i, d_a, s);
    
    cudaDeviceSynchronize();

    // FFT driver code for column wise FFT
    // TODO: Check for optimisations since accesses here are column wise
    for(int i = 0; i < ARRAY_SIZE; i++)
    {
        for (int j=1;j<=s;j++){

            // m = 2^j
            int m = 1 << j;

            if (m/2 < MAX_NO_OF_THREADS_PER_BLOCK){

                fft_outer_column<<<((ARRAY_SIZE/m)+THREADS-1)/THREADS,THREADS>>>(d_rev,i,m);    

            } else {
                
                for (int k=0;k<ARRAY_SIZE;k+=m){

                    fft_inner_column<<<((m/2)+THREADS-1)/THREADS,THREADS>>>(d_rev,i,k,m);

                }
            }
        }
    }

    //End of performance measurement
    cudaEventRecord(stop);

    //Block CPU execution until the event "stop" is recorded
    cudaEventSynchronize(stop);

    //Print the time taken in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("The total time taken is %f milliseconds\n", milliseconds);

    // Copy result array from device to host
    cudaMemcpy(h_rev,d_rev,ARRAY_BYTES * ARRAY_BYTES,cudaMemcpyDeviceToHost);

    // Writing output to files
    fprintf(fptr, "i\t\ta.magn\t\ta.real\t\t\ta.img\n");
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        for(int j = 0; j < ARRAY_SIZE; j++)
        {
            fprintf(fptr,"%d\t\t%f\t\t%f\t\t%f\n", i, magnitude(h_rev[i][j]), h_rev[i][j].x, h_rev[i][j].y);
        }
    }

    // Free allocated device memory
    cudaFree(d_a);
    cudaFree(d_rev);

    return 0;
}