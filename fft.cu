// Include header files
#include <bits/stdc++.h>
#include <cuda.h>

#define ll long long int

const complex<double> J(0, 1);

using namespace std;

// wave_to_sample

vector<complex<double>> reverse_bits(vector<complex<double>> a)
{
    double temp;
    vector<complex<double>> a_rev;
    for(ll i = 0; i < a.size(); i++)
    {
        if(a[i].real == 0)
        {
            a_rev[i].real = 0;
            continue;
        }
        a_rev[i].real = 1;
        if(a[i].imag == 0)
        {
            a_rev[i].imag = 0;
            continue;
        }
        a_rev[i].imag = 1;
        for(ll k = 0; k < log2(a[i].real); i++)
        {
            a_rev[i].real = a_rev[i].real*2 + a[i].real%2;
            a[i].real = a[i].real/2;
        }
        for(ll k = 0; k < log2(a[i].imag); i++)
        {
            a_rev[i].imag = a_rev[i].imag*2 + a[i].imag%2;
            a[i].imag = a[i].imag/2;
        }
    }
    return(a_rev);
}

void fft(vector<complex<double>> &a)
{
    ll n = a.size();
    for(ll s = 1; s < log2(n); s++)
    {
        ll m = pow(2,s);

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
    
    vector<complex<double>> sample;
    // TODO 
    // Implement a function wave_to_sample
    // Return it to vector samples

    vector<complex<double>> brev_sample;
    // TODO
    // Call brev to reverse the bits
    brev_sample = reverse_bits(sample);

    // TODO
    // Parallelise FFT function
    
    fft(brev_sample);

}