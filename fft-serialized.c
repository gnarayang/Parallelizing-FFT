#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<unistd.h> 

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

#define ARRAY_SIZE 65536

typedef struct Complex{
    float re,im;
} Complex;

const long long ARRAY_BYTES = ARRAY_SIZE*sizeof(Complex);

int bit_reverse(int n, int s){
    int rev=0, count=s;
    while(n){
        rev = rev << 1;
        rev |= n & 1;
        n = n >> 1;
        count--;
    }
    rev = rev << count;
    return rev;
}

void bit_reverse_reorder(Complex *a, Complex *rev, int s){
    for(int i=0;i<ARRAY_SIZE;i++){        
        rev[bit_reverse(i,s)] = a[i];
    }
}

void fft(Complex *a, int s){
    for(int i=1;i<=s;i++){
        
        int m = 1 << i;
        Complex w_m;
        w_m.re = cosf((2*M_PI)/m);
        w_m.im = -sinf((2*M_PI)/m); 

        for (int j=0;j<ARRAY_SIZE;j+=m){
            
            Complex w;
            w.re = 1;
            w.im = 0;

            for (int k=0;k<m/2;k++){

                Complex t, u;
                float w_x = w.re, w_y = w.im;

                t.re = w_x*a[j+k+m/2].re - w_y*a[j+k+m/2].im;
                t.im = w_x*a[j+k+m/2].im + w_y*a[j+k+m/2].re;

                u.re = a[j+k].re;
                u.im = a[j+k].im;

                a[j+k].re = u.re + t.re;
                a[j+k].im = u.im + t.im;

                a[j+k+m/2].re = u.re - t.re;
                a[j+k+m/2].im = u.im - t.im;

                w.re = w_x*w_m.re - w_y*w_m.im;
                w.im = w_x*w_m.im + w_y*w_m.re;                
            }
        }
    }
}

void main(){
    FILE *fptr;
    fptr = fopen("fft-serialized-output.dat", "wr");
    if (fptr == NULL) {
        printf("Error!");
        exit(1);
    }
    Complex *a, *rev;
    int s = (int)ceil(log2(ARRAY_SIZE));
    a = (Complex *)malloc(ARRAY_BYTES);
    rev = (Complex *)malloc(ARRAY_BYTES);
    for (int i=0;i<ARRAY_SIZE;i++){
        a[i].re = sinf((12*M_PI*i)/ARRAY_SIZE);
        a[i].im = 0;
    }
    bit_reverse_reorder(a,rev,s);
    fft(rev,s);
    fprintf(fptr, "i\t\trev.real\t\trev.img\t\ta.real\t\ta.img\n");
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        fprintf(fptr,"%d\t\t%f\t\t%f\t\t%f\t\t%f\n", i, rev[i].re, rev[i].im, a[i].re, a[i].im);
    }
    
    // printf ("%f %f %f %f\n",rev[6].re, rev[6].im, rev[ARRAY_SIZE-6].re, rev[ARRAY_SIZE-6].im);
}