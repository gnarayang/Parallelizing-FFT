// Include header files
#include <bits/stdc++.h>
#include <cuda.h>
#include <cmath>

#define ll long long int
#define THREADS 32

typedef float2 Complex;

const long long ARRAY_SIZE = 1024;
const long long ARRAY_BYTES = ARRAY_SIZE * sizeof(Complex);

// Parallelized reordering (Doesn't this count as pre-processing?)
__global__ void bit_reverse_reorder (Complex *d_rev, Complex *d_a, int s){
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    int rev = __brev(id) >> (32-s);
    if (id < ARRAY_SIZE)
        d_rev[rev] = d_a[id];
}

// FFT driver kernel code
__global__ void fft(Complex *a, int j, int m){
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if(k < m/2 && j+k+m/2 < ARRAY_SIZE){
        
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

float magnitude(float2 a)
{
    return sqrt(a.x*a.x + a.y*a.y);
}

int main(int argc, char *argv[]) {
    // Creating files to write output to
    FILE *fptr;
    fptr = fopen("fft-output.dat", "wr");

    //Creating Complex arrays for data 
    Complex h_a[ARRAY_SIZE], h_rev[ARRAY_SIZE]; 
    Complex *d_a, *d_rev;

    // Input signal (of the form sin((2*M_PI*f*x)/N)) where N is the sample size
    // imaginary part of the signal by default is 0
    for(int i = 0; i < ARRAY_SIZE; i++) {
    	h_a[i].x = sin((10*M_PI*i)/ARRAY_SIZE);
        h_a[i].y = 0.0;
    }
        
    // No. of bits in the sample size, used for bit reversal reordering
    int s = (int)ceil(log2(ARRAY_SIZE));

    //Allocate memory for the device arrays
    cudaMalloc((void**) &d_a, ARRAY_BYTES);
    cudaMalloc((void**) &d_rev, ARRAY_BYTES);

    //Copy all elements of sample array from host to device
    cudaMemcpy(d_a, h_a, ARRAY_BYTES, cudaMemcpyHostToDevice);

    //Reorder the sample as first step of FFT
    bit_reverse_reorder<<<(ARRAY_SIZE+THREADS-1)/THREADS, THREADS>>>(d_rev, d_a, s);

    //Synchronise devices before jumping to fft
    cudaDeviceSynchronize();

    // Naive fft parallelization (TODO: improve upon the efficiency)
    for (int i=0;i<=s;i++){

        int m = 1 << i;
        
        for(int j=0;j<ARRAY_SIZE;j+=m){

            // Performing in-place fft
            fft<<<((m/2)+THREADS-1)/THREADS,THREADS>>>(d_rev,j,m);    
        
        }    
    }

    // Copy result array to host
    cudaMemcpy(h_rev, d_rev, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // Writing output to files
    fprintf(fptr, "i\t\ta.magn\t\ta.real\t\t\ta.img\n");
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        fprintf(fptr,"%d\t\t%f\t\t%f\t\t%f\n", i, magnitude(h_rev[i]), h_rev[i].x, h_rev[i].y);
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_rev);
    
    // TODO
    // Use h_rev to plot magnitude (sqrt(h_rev[i].x^2 + h_rev[i].y^2)) vs frequency (i)
}