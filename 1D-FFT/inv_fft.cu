#include <stdio.h>
#include <cmath>
#include <cuda.h>

typedef float2 Complex;

#define THREADS 32
#define MAX_NO_OF_THREADS_PER_BLOCK 1024

long long ARRAY_SIZE; 
long long ARRAY_BYTES;

__global__ void bit_reverse_reorder(Complex *d_rev, Complex *d_a, int s, long long ARRAY_SIZE) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
    int rev = __brev(id) >> (32-s);

    if(id < ARRAY_SIZE)
        d_rev[rev] = d_a[id];
}

__global__ void swap_real_and_imaginary(Complex *d_rev, long long ARRAY_SIZE) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < ARRAY_SIZE) {
        float temp = d_rev[id].x;
        d_rev[id].x = d_rev[id].y;
        d_rev[id].y = temp;
    }
}

__global__ void clip(Complex *d_rev, long long cut_off_frequency, long long ARRAY_SIZE) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id > cut_off_frequency && id < (ARRAY_SIZE - cut_off_frequency)) {
        d_rev[id].x = 0.0f;
        d_rev[id].y = 0.0f;
    }
}

__device__ void inplace_fft(Complex *a, int j, int k, int m, long long ARRAY_SIZE){
    
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

__global__ void fft_outer(Complex *a, int m, long long ARRAY_SIZE){
    int j = (blockIdx.x * blockDim.x + threadIdx.x)*m;
    if (j < ARRAY_SIZE){
        for (int k=0;k<m/2;k++){
            inplace_fft(a,j,k,m,ARRAY_SIZE);
        }
    }    
}

__global__ void fft_inner(Complex *a, int j, int m, long long ARRAY_SIZE){
    int k = (blockIdx.x * blockDim.x + threadIdx.x);
    if (k < m/2)
        inplace_fft(a,j,k,m,ARRAY_SIZE);
}

float magnitude(float2 a)
{
    return sqrt(a.x*a.x + a.y*a.y);
}

int to_int(char str[])
{
    int cut_off_frequency = 0;
    int i = 0;
    while(str[i] != '\0')
    {
        cut_off_frequency += str[i] - 48;
        cut_off_frequency *= 10;

        i++;
    }

    return cut_off_frequency/10;
}

int main(int argc, char *argv[]) 
{
    int cut_off_frequency;

    if(argc >= 2)
        cut_off_frequency = to_int(argv[1]);
    
    else
        cut_off_frequency = 1000;

    // printf("%d\n", cut_off_frequency);
    

    //Measuring performance
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //Creating file to read input from
    FILE *input;
    input = fopen("../utils/input.dat", "r");

    int fs;

    fscanf(input, "%d", &fs);

    fscanf(input, "%Ld", &ARRAY_SIZE);

    ARRAY_BYTES = ARRAY_SIZE * sizeof(Complex);

    // Creating files to write output to
    FILE *fptr;
    FILE *output;

    fptr = fopen("../utils/fft-opt1-output.dat", "wr");
    output = fopen("../utils/convert_to_wav.dat", "wr");
    
    Complex *h_a = (Complex *)malloc(ARRAY_BYTES);
    Complex *h_rev = (Complex *)malloc(ARRAY_BYTES);

    for(int i = 0; i < ARRAY_SIZE; i++) 
    {
   	    fscanf(input, "%f", &h_a[i].x);
        // h_a[i].x = sin((6.0*M_PI*i)/ARRAY_SIZE);
        // printf("%f\n", h_a[i].x);
        h_a[i].y = 0.0;
    }
    	
    int s = (int)ceil(log2(ARRAY_SIZE));

    Complex *d_a, *d_rev, *d_rev1;

    cudaMalloc((void**) &d_a, ARRAY_BYTES);
    cudaMalloc((void**) &d_rev, ARRAY_BYTES);
    cudaMalloc((void**) &d_rev1, ARRAY_BYTES);

    cudaMemcpy(d_a, h_a, ARRAY_BYTES, cudaMemcpyHostToDevice);

    //Start of performance measurement
    cudaEventRecord(start);

    bit_reverse_reorder<<<(ARRAY_SIZE+THREADS-1)/THREADS, THREADS>>>(d_rev, d_a, s, ARRAY_SIZE);
    
    cudaDeviceSynchronize();

    for (int i=1;i<=s;i++){
        int m = 1 << i;
        if (m/2 < MAX_NO_OF_THREADS_PER_BLOCK){
            fft_outer<<<((ARRAY_SIZE/m)+THREADS-1)/THREADS,THREADS>>>(d_rev,m,ARRAY_SIZE);
        } else {
            for (int j=0;j<ARRAY_SIZE;j+=m){
                fft_inner<<<((m/2)+THREADS-1)/THREADS,THREADS>>>(d_rev,j,m,ARRAY_SIZE);
            }
        }
    }

    cudaDeviceSynchronize();

    // Clipping
    clip<<<(ARRAY_SIZE+THREADS-1)/THREADS, THREADS>>>(d_rev, cut_off_frequency, ARRAY_SIZE);

    // Beginning of inverse FFT

    swap_real_and_imaginary<<<(ARRAY_SIZE+THREADS-1)/THREADS, THREADS>>>(d_rev, ARRAY_SIZE);

    cudaDeviceSynchronize();

    bit_reverse_reorder<<<(ARRAY_SIZE+THREADS-1)/THREADS, THREADS>>>(d_rev1, d_rev, s, ARRAY_SIZE);

    cudaDeviceSynchronize();
    
    for (int i=1;i<=s;i++)
    {
        int m = 1 << i;
        if (m < sqrt(ARRAY_SIZE / 4))
        {
            fft_outer<<<((ARRAY_SIZE/m)+THREADS-1)/THREADS,THREADS>>>(d_rev1,m,ARRAY_SIZE);
        } 
        else 
        {
            for (int j=0;j<ARRAY_SIZE;j+=m)
            {
                fft_inner<<<((m/2)+THREADS-1)/THREADS,THREADS>>>(d_rev1,j,m,ARRAY_SIZE);
            }
        }
    }

    swap_real_and_imaginary<<<(ARRAY_SIZE+THREADS-1)/THREADS, THREADS>>>(d_rev1, ARRAY_SIZE);

    // End of performance measurement
    cudaEventRecord(stop);

    // Block CPU execution until the event "stop" is recorded
    cudaEventSynchronize(stop);

    //Print the time taken in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("The total time taken is %f milliseconds\n", milliseconds);

    cudaMemcpy(h_rev,d_rev1,ARRAY_BYTES,cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_rev);
    cudaFree(d_rev1);
    
    // for (int i=0;i<ARRAY_SIZE;i++){
    //     printf ("%f %f\n", h_rev[i].x/ARRAY_SIZE, h_rev[i].y/ARRAY_SIZE);
    // }

    // Writing output to files
    fprintf(output, "%d\n", fs);
    fprintf(fptr, "i\t\ta.magn\t\ta.real\t\t\ta.img\n");
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        fprintf(output, "%Ld\n", (long long)h_rev[i].x/ARRAY_SIZE);
        fprintf(fptr,"%d\t\t%f\t\t%f\t\t%f\n", i, magnitude(h_rev[i])/ARRAY_SIZE, h_rev[i].x/ARRAY_SIZE, h_rev[i].y/ARRAY_SIZE);
    }

    return 0;
}