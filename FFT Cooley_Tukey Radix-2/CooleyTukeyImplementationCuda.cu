//GPU Programming - Project
//Name: Cavaughn Browne
//Parallel Programming Date: 12/5/2016

//Reads N sets of data from a file called data.dat and processes them using 
//the FFT-Cooley Tukey Algorithm. 

//compile with these lines with the data.dat file in the same directory
//module load cuda
//nvcc -arch=compute_35 -code=sm_35 CooleyTukeyImplementationCuda.cu -o a.out

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


#define PI 3.141592653589793

//A ComplexNum consists of a two doubles: real and imag 
struct ComplexNum 
{
	double real;
	double imag;
};

void FFT(ComplexNum* signals, int N);

__global__ void FFT_Kernel(ComplexNum* signals_d, ComplexNum* XkResults, int N);

int main(int argc, char **argv)
{
	
	FILE *fp;
	fp = fopen("data.dat", "r");
	
	//Takes the first argument from stdin and stores it in N
	int N = atoi(argv[1]);
	int j;
	struct ComplexNum x[N]; //to store the signals in the file 
	
	j = 0;
	
	//Read from the file while j < N and not the end of the file
	while(j < N && !feof(fp))
	{
		fscanf(fp, "%lf", &x[j].real);
		fscanf(fp, "%lf", &x[j].imag);
		j++;
	}
	
	//fill the rest of the array with 0(s) if j < N
	if(j < N)
	{
		for(; j < N; j++)
		{
			x[j].real = 0;
			x[j].imag = 0;
		}
	}
	
	printf("TOTAL PROCESSED SAMPLES: %d\n", N);
	
	//calculate the FFT
	FFT(x, N);
	
}

/*
Invokes the FFT_Kernel to calculate the FFT of the 
"signals" of number N. After the first 8 results are
printed.
*/
void FFT(ComplexNum* signals, int N)
{
	int size = N * 2* sizeof(double);
	int threads;
	int blocks;
	ComplexNum* signals_d;
	ComplexNum* XkResults_d;
	ComplexNum XkResults_h[N];
	float time, commTime, commTime2;
	cudaEvent_t start, stop, start2, stop2; //For timing
	clock_t st, diff, st2, diff2;

	
	//calculate the number of blocks and threads to use
	if(N < 1024)
	{
		threads = N % 1024;
		blocks = N/threads;
	}
	else if(N % 1024 == 0)
	{
		threads = 1024;
		blocks = N/threads;

	}
	else
	{
		threads = 1024;
		blocks = (N/threads) + 1;
	}
	
	//Cuda Time
	cudaEventCreate(&start2); //Creates the start2 time event
	cudaEventCreate(&stop2) ; //Creates the stop2 time event
	cudaEventRecord(start2, 0) ; //Records the start2 time
	
	//C-time 
	//store the current clock time in st 
	st2 = clock();
	
	//Memory Allocation and Transfer
	cudaMalloc((void**)&signals_d, size);
	cudaMemcpy(signals_d, signals, size, cudaMemcpyHostToDevice);
	
	cudaMalloc((void**)&XkResults_d, size);
	
	//End of timing period
	//C-time
	//find the difference between the st clock time and current clock time
	diff2 = clock() - st2; 
	
	//Cuda-time
	cudaEventRecord(stop2, 0); //Records the stop time
	cudaEventSynchronize(stop2); 
	//the elapsed time between start and stop is stored in time in 
	//milliseconds
	cudaEventElapsedTime(&commTime, start2, stop2); 
	cudaEventDestroy(start2);
	cudaEventDestroy(stop2);
	
	
	
	//New Timing period
	//Cuda-time
	cudaEventCreate(&start); //Creates the start time event
	cudaEventCreate(&stop) ; //Creates the stop time event
	cudaEventRecord(start, 0) ; //Records the start time
	
	//C-time 
	//store the current clock time in st 
	st = clock();
	
	FFT_Kernel<<< blocks, threads >>>(signals_d, XkResults_d, N);
	
	//C-time
	//find the difference between the st clock time and current clock time
	diff = clock() - st; 
	
	//Cuda-time
	cudaEventRecord(stop, 0); //Records the stop time
	cudaEventSynchronize(stop); 
	//the elapsed time between start and stop is stored in time in 
	//milliseconds
	cudaEventElapsedTime(&time, start, stop); 
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	//Start timing again
	//Cuda Time
	cudaEventCreate(&start2); //Creates the start2 time event
	cudaEventCreate(&stop2) ; //Creates the stop2 time event
	cudaEventRecord(start2, 0) ; //Records the start2 time
	
	//C-time 
	//store the current clock time in st 
	st2 = clock();
	
	
	//copy the results from the device back to the host
	cudaMemcpy(XkResults_h, XkResults_d, size, cudaMemcpyDeviceToHost);
	
	//Free memory on device
	cudaFree(signals_d);
	cudaFree(XkResults_d);
	
	//End of timing period
	//C-time
	//find the difference between the st clock time and current clock time
	diff2 += (clock() - st2); 
	
	//Cuda-time
	cudaEventRecord(stop2, 0); //Records the stop time
	cudaEventSynchronize(stop2); 
	//the elapsed time between start and stop is stored in time in 
	//milliseconds
	cudaEventElapsedTime(&commTime2, start2, stop2); 
	cudaEventDestroy(start2);
	cudaEventDestroy(stop2);
	
	
	//prints the first 8
	int p;
	for (p = 0; p < 8; p++)
	{
		printf("XR[%d] : %lf\nXI[%d] : %lf\n", p, XkResults_h[p].real, p, XkResults_h[p].imag);

	}

	printf("Cuda Communication Time:  %3.1f ms \n", commTime + commTime2);
	printf("C Communication Time:  %3.1f ms \n\n", diff2);

	
	printf("Cuda Calculation Time:  %3.1f ms \n", time);
	printf("C Calculation Time: %3.1f ms \n", diff);
	
}

/*
	Each thread calculates one Xk result by
	summing up even and odd parts for m = 0 
	to m < N/2 of the FFT equation
*/
__global__
void FFT_Kernel(ComplexNum* signals_d, ComplexNum* XkResults, int N)
{
	struct ComplexNum Xk;
	struct ComplexNum evenP;
	struct ComplexNum oddP;
	double c, s, realPart, imgPart;
	int m, k;
	
	//thread will compute Xk....k = threadIdx.x
	k = blockIdx.x * blockDim.x + threadIdx.x;
	double theta = (-2 * PI * k) / (N / 2);
	 if(k < N)
	 {
		evenP.real = 0;
		evenP.imag = 0;
		oddP.real = 0;
		oddP.imag = 0;
		
		for ( m = 0; m < N / 2; m++)
		{
			
			c = cos(theta * m);
			s = sin(theta * m);
			
			//Even Index Part computation
			realPart = (signals_d[2 * m].real *c) - ((signals_d[2 * m].imag * s));
			evenP.real += realPart;
			imgPart = (signals_d[2 * m].real *s) + ((signals_d[2 * m].imag * c));
			evenP.imag += imgPart;

			//Odd Index Part Computation
			realPart = (signals_d[(2 * m) + 1].real *c) - ((signals_d[(2 * m) + 1].imag * s));
			oddP.real += realPart;
			imgPart = (signals_d[(2 * m) + 1].real *s) + ((signals_d[(2 * m) + 1].imag * c));
			oddP.imag += imgPart;
		}

		//Add the real and the odd part sums and store the result.
		Xk.real = evenP.real + (cos(theta / 2) * oddP.real) - (sin(theta / 2) * oddP.imag);
		Xk.imag = evenP.imag + (cos(theta / 2) * oddP.imag) + (sin(theta / 2) * oddP.real);
		
		XkResults[k] = Xk;
		
	}
		
}