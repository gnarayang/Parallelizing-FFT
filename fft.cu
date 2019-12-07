// Include header files
#include <bits/stdc++.h>
#include <cuda.h>
#include <cmath>

#define ll long long int

// const complex<double> J(0, 1);
// Maybe use float2 instead of complex?
// we can use static inline functions for addition, multiplication and inverse of complex numbers
typedef float2 Complex;

const int THREADS = 32;
const long long ARRAY_SIZE = 1024;
const long long ARRAY_BYTES = ARRAY_SIZE * sizeof(Complex);

__global__ void bit_reverse_reorder (Complex *d_rev, Complex *d_a, int s){
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    int rev = __brev(id) >> (32-s);
    if (id < ARRAY_SIZE)
        d_rev[rev] = d_a[id];
}

using namespace std;

// wave_to_sample
ll bit_reverse_host(ll n, int s){
    ll rev = 0;
    ll count = s;
    while(n){
        rev <<= 1;
        rev |= (n&1);
        n >>= 1;
        count--;
    }
    rev <<= count;
    return rev;
}

Complex* reverse_bits(Complex *a, int s) {
    Complex *a_rev;
    for (int i=0;i<a.size();i++){
        a_rev[bit_reverse_host(i,s)] = a[i];
    }
    return a_rev;
}

void fft(Complex *a)
{
    ll n = a.size();

    // DOUBT:
    // isn't the algo the following:
    // for(s=1 to log2(n)){
    //     w_m = exp(-2pi*j/m)
    //     for(j=0;j<n;j+=m){
    //          w = 1;
    //          for(k=0 to m/2-1){
    //              t = w * a[j+k+m/2];
    //              u = a[j+k];
    //              a[j+k] = u+t;              
    //              a[j+k+m/2] = u-t;
    //              w *= w_m;
    //          }
    //     }           
    // }

    for(ll s = 1; s < log2(n); s++)
    {
        ll m = pow(2,s);

        // Make changes here
        complex<double> w(1,0);

        complex<double> wm = exp(J * 2 * M_1_PI/ m);

        for(ll i = 0; i < m/2; i++)
        {
            for(ll k = i; k < n; k+=m)
            {
                complex<double> t = w * a[k + m/2];
                complex<double> u = a[k];

                a[k] = u + t;
                a[k+m/2] = u - t;
            }
            w = w * wm;
        }
    }
}

int main(int argc, char *argv[]) {
    
    // brev_sample = reverse_bits(sample);

    //Creating Complex arrays for data 
    Complex h_a[ARRAY_SIZE], h_rev[ARRAY_SIZE]; 
    Complex *d_a, *d_rev;

    for(int i = 0; i < ARRAY_SIZE; i++) {
    	h_a[i].x = sin((10*M_PI*i)/ARRAY_SIZE);
        h_a[i].y = 0.0;
    }
    	
    int s = (int)ceil(log2(ARRAY_SIZE));

    cudaMalloc((void**) &d_a, ARRAY_BYTES);
    cudaMalloc((void**) &d_rev, ARRAY_BYTES);

    cudaMemcpy(d_a, h_a, ARRAY_BYTES, cudaMemcpyHostToDevice);

    bit_reverse_reorder<<<(int)ceil(ARRAY_SIZE/THREADS), THREADS>>>(d_rev, d_a, s);

    cudaMemcpy(h_rev, d_rev, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_rev);

    // TODO
    // Parallelise FFT function
    
    fft(brev_sample);

}