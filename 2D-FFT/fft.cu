#include <stdio.h>
#include <cmath>
#include <cuda.h>

typedef float2 Complex;

#define THREADS 32
#define MAX_NO_OF_THREADS_PER_BLOCK 1024

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

// TODO: Make sure the implementations works for non square matrices
const long long ARRAY_SIZE = 1024; 
const long long ARRAY_BYTES = ARRAY_SIZE * sizeof(Complex);

// Bit reversal re-ordering, first step of FFT
__global__ void bit_reverse_reorder(Complex *d_rev, Complex *d_a, int s) 
{
    int row_id = blockIdx.y;
    int col_id = threadIdx.x;
    int row_size = ARRAY_SIZE;
    int rev = __brev(col_id) >> (32-s);
    
    // d_rev[row_id][rev] = d_a[row_id][col_id]
    d_rev[row_id * row_size + rev] = d_a[row_id * row_size + col_id];
}

// Optimized kernel for calculating FFT of all rows simultaneously

__global__ void fft_2d(Complex *A, int s)
{
    int row_id = blockIdx.y;
    int col_id = threadIdx.x;
    int row_size = ARRAY_SIZE;

    // Shared memory allocated indepedent to each block
    __shared__ Complex Work_Array[ARRAY_SIZE];
    
    // Work_Array[col_id] = A[row_id][col_id];
    Work_Array[col_id].x = A[(row_id * row_size) + col_id].x;
    Work_Array[col_id].y = A[(row_id * row_size) + col_id].y;

    // Work_Array[col_id + row_size/2] = A[row_id][col_id + row_size/2];
    Work_Array[col_id + row_size/2].x = A[(row_id * row_size) + (col_id + row_size/2)].x;
    Work_Array[col_id + row_size/2].y = A[(row_id * row_size) + (col_id + row_size/2)].y;

    __syncthreads();

    for(int i=1; i<=s; i++)
    {
        int m = 1 << i;

        int j = col_id * m;

        for(int k=0; k < m/2; k++)
        {
            if(j + k + m/2 < row_size)
            {
                Complex w, t, u;

                // w^k (w is root of unity)
                w.x = __cosf((2*M_PI*k)/m);
                w.y = -__sinf((2*M_PI*k)/m);

                // u = a[j+k]
                u.x = Work_Array[j+k].x;
                u.y = Work_Array[j+k].y;

                // t = w*a[j+k+m/2];
                t.x = w.x*Work_Array[j+k+m/2].x - w.y*Work_Array[j+k+m/2].y;
                t.y = w.x*Work_Array[j+k+m/2].y + w.y*Work_Array[j+k+m/2].x;

                // a[j+k] = u+t;
                Work_Array[j+k].x = u.x + t.x;
                Work_Array[j+k].y = u.y + t.y;

                // a[j+k+m/2] = u-t;
                Work_Array[j+k+m/2].x = u.x - t.x;
                Work_Array[j+k+m/2].y = u.y - t.y;

            }
        }

        __syncthreads();
    }

    // Copying data back from shared memory to global memory

    // A[row_id][col_id] = Work_Array[col_id];
    A[(row_id * row_size) + col_id].x = Work_Array[col_id].x;
    A[(row_id * row_size) + col_id].y = Work_Array[col_id].y;

    // A[row_id][col_id + row_size/2] =  Work_Array[col_id + row_size/2];
    A[(row_id * row_size) + (col_id + row_size/2)].x =  Work_Array[col_id + row_size/2].x;
    A[(row_id * row_size) + (col_id + row_size/2)].y =  Work_Array[col_id + row_size/2].y;

}

__global__ void MatrixTranspose(Complex *d_out, Complex *d_in)
{
    __shared__ Complex tile[TILE_DIM][TILE_DIM+1];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y+j][threadIdx.x] = d_in[(y+j)*width + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        d_out[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

float magnitude(float2 a)
{
    return sqrt(a.x*a.x + a.y*a.y);
}

int main() 
{

    //Measuring performance
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Creating files to write output to
    FILE *fptr;
    fptr = fopen("fft-2d-output.dat", "wr");

    // Host arrays for input and output
    Complex h_a[ARRAY_SIZE][ARRAY_SIZE];
    Complex h_rev[ARRAY_SIZE][ARRAY_SIZE];

    // Input signal, complex part remains zero.
    // Signal is of the form sin(2*M_PI*f*x/N) or cos(2*M_PI*f*x/N)
    // N is sample size and is always a power of 2 
    for(int i = 0; i < ARRAY_SIZE; i++) 
    {
        for(int j = 0; j < ARRAY_SIZE; j++)
        {
            h_a[i][j].x = cos((6.0*M_PI*j)/ARRAY_SIZE);
            h_a[i][j].y = 0.0;
        }
    }
        
    // No of bits required to represent N
    int s = (int)ceil(log2(ARRAY_SIZE));

    // Device arrays
    Complex *d_a, *d_rev, *d_rev1;

    // Memory Allocation
    cudaMalloc((void**) &d_a, ARRAY_BYTES * ARRAY_SIZE);
    cudaMalloc((void**) &d_rev, ARRAY_BYTES * ARRAY_SIZE);
    cudaMalloc((void**) &d_rev1, ARRAY_BYTES * ARRAY_SIZE);

    // Copy host array to device
    cudaMemcpy(d_a, *h_a, ARRAY_BYTES * ARRAY_SIZE, cudaMemcpyHostToDevice);

    //Start of performance measurement
    cudaEventRecord(start);

    // First step in FFT, bit-reverse reordering
    // Swap an element at index 'i', with the element 
    // which is the bit-string reversal of 'i' 
        
    bit_reverse_reorder<<<dim3(1, ARRAY_SIZE, 1), dim3(ARRAY_SIZE, 1, 1)>>>(d_rev, d_a, s);
    
    cudaDeviceSynchronize();

    // FFT driver code for row wise FFT
    // Assigning every row of the 2d fft to a block of threads and calling N/2 threads for each block
    fft_2d <<<dim3(1, ARRAY_SIZE, 1), dim3(ARRAY_SIZE/2, 1, 1)>>> (d_rev, s);

    // TODO : Code for matrix transpose - DONE

    cudaMemset(d_rev1, 0, ARRAY_BYTES*ARRAY_SIZE);

    // Taking transpose of d_rev and storing in d_rev1
    MatrixTranspose<<<dim3(ARRAY_SIZE/TILE_DIM, ARRAY_SIZE/TILE_DIM, 1), dim3(TILE_DIM, BLOCK_ROWS, 1)>>>(d_rev1, d_rev);

    // Column wise bit-reverse reordering of d_rev1, and storing in d_rev
    bit_reverse_reorder<<<dim3(1, ARRAY_SIZE, 1), dim3(ARRAY_SIZE, 1, 1)>>>(d_rev, d_rev1, s);
    
    cudaDeviceSynchronize();

    // FFT driver code for column wise FFT
    
    fft_2d <<<dim3(1, ARRAY_SIZE, 1), dim3(ARRAY_SIZE/2, 1, 1)>>> (d_rev, s);

    //End of performance measurement
    cudaEventRecord(stop);

    //Block CPU execution until the event "stop" is recorded
    cudaEventSynchronize(stop);

    //Print the time taken in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("The total time taken is %f milliseconds\n", milliseconds);

    // Copy result array from device to host
    cudaMemcpy(*h_rev, d_rev, ARRAY_BYTES * ARRAY_SIZE, cudaMemcpyDeviceToHost);

    // Writing output to files
    fprintf(fptr, "[i][j]\t\ta.magn\t\ta.real\t\t\ta.img\n");
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        for(int j = 0; j < ARRAY_SIZE; j++)
        {
            fprintf(fptr,"[%d][%d]\t\t%f\t\t%f\t\t%f\n", i, j, magnitude(h_rev[i][j]), h_rev[i][j].x, h_rev[i][j].y);
        }
    }

    // Free allocated device memory
    cudaFree(d_a);
    cudaFree(d_rev);
    cudaFree(d_rev1);

    return 0;
}