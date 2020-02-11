/*  
    Autor:   Matej Poljuha
    Kolegij: Paralelno programiranje na heterogenim sustavima
    Tema:    Izračun vrijednosti određenog integrala trapeznom formulom - CPU+GPU varijanta
*/

#include <assert.h>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <chrono>
#include "hip/hip_runtime.h"

using namespace std;

#define HIP_ASSERT(x) (assert((x) == hipSuccess))

#define TIMEPOINT chrono::high_resolution_clock::now()

#define OPERATIONS_PER_THREAD 1024.0
#define NUMBER_OF_THREADS ceil((n+1)/OPERATIONS_PER_THREAD)                 // Broj zadanih segmenata + 1 (zbog rubnih točaka) / zadani broj operacija koje izvodi 1 thread

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32
#define THREADS_PER_BLOCK_Z 1

__host__ __device__ double f (double x);
__global__ void GPU_trapeze_formula(double*  y, const double*  a, const double*  h, const long n);
int grid_dimensions (long n);
int valid_segment(double a, double b, long n);
void benchmarking (chrono::_V2::high_resolution_clock::time_point, chrono::_V2::high_resolution_clock::time_point);

int main (void){
    long n;                                                                 // Broj podsegmenata
    double a, b, sum = 0;
    
    double *host_a;
    double *host_h;
    double *host_y;

    double *device_a;
    double *device_h;
    double *device_y;

    cout << "Podintegralna funkcija: " << "sin(x)" << endl;

    do {
        cout << "\nUnos granica integracije,\na = ";                        // Zadavanje parametara
        cin >> a;                                                           // Donja granica integrala
        cout << "b = ";
        cin >> b;                                                           // Gornja granica integrala
        cout << "Unos broja segmenata\nn = ";
        cin >> n;                                                           // Broj segmenata za trapeznu formulu
    } while (valid_segment(a, b, n));
    
    host_a = (double*)malloc(sizeof(double));                               // Alokacija memorije na CPU
    host_h = (double*)malloc(sizeof(double));
    host_y = (double*)malloc(NUMBER_OF_THREADS * sizeof(double));           // n+1 veličina polja jer n segmenata ima n+1 rubnih točaka

    int dimension = grid_dimensions(n);                                     // Računa dimenziju grida
    *host_a = a;
    *host_h = (b-a)/n;

    /*
        Računanje vrijednosti integrala (poziv funkcije)
    */
    
    cout << "\nDobivena vrijednost integrala je: " << std::fixed << std::setprecision(80) << endl;

    auto total_time_start = TIMEPOINT;                                                  // Početak mjerenja ukupnog vremena izvođenja algoritma

    auto cpu_2_gpu_start = TIMEPOINT;                                                   // Početak mjerenja vremena prijenosa podataka sa CPU-a na GPU

    HIP_ASSERT(hipMalloc((void**)&device_a, sizeof(double)));
    HIP_ASSERT(hipMalloc((void**)&device_h, sizeof(double)));                           // Alokacija memorije na GPU-u
    HIP_ASSERT(hipMalloc((void**)&device_y, NUMBER_OF_THREADS * sizeof(double)));
    
    HIP_ASSERT(hipMemcpy(device_a, host_a, sizeof(double), hipMemcpyHostToDevice));     // Prijenos podataka na GPU
    HIP_ASSERT(hipMemcpy(device_h, host_h, sizeof(double), hipMemcpyHostToDevice));

    auto cpu_2_gpu_end = TIMEPOINT;                                                     // Kraj mjerenja vremena prijenosa podataka sa CPU-a na GPU

    int num_of_blocks_x = (dimension % THREADS_PER_BLOCK_X == 0)?(dimension/THREADS_PER_BLOCK_X):(dimension/THREADS_PER_BLOCK_X + 1);
    int num_of_blocks_y = (dimension % THREADS_PER_BLOCK_Y == 0)?(dimension/THREADS_PER_BLOCK_Y):(dimension/THREADS_PER_BLOCK_Y + 1);

    auto calc_start = TIMEPOINT;                                                        // Početak mjerenja vremena izvođenja algoritma GPU-u

    hipLaunchKernelGGL(                                                                 // Poziv funkcije na GPU-u
        GPU_trapeze_formula, 
        dim3(num_of_blocks_x, num_of_blocks_y),
        dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
        0, 0,
        device_y, device_a, device_h, n
    );

    HIP_ASSERT(hipMemcpy(host_y, device_y, NUMBER_OF_THREADS * sizeof(double), hipMemcpyDeviceToHost));

    auto calc_end = TIMEPOINT;                                                          // Kraj mjerenja vremena izvođenja algoritma GPU-u

    for (int i = 0; i < NUMBER_OF_THREADS; i++){                                        // Izračun trapezne formule na osnovu dobivenih podataka
        sum += host_y[i];
    }

    cout << sum << endl;

    auto total_time_end = TIMEPOINT;                                                    // Kraj mjerenja ukupnog vremena izvođenja algoritma
    
    /*
        Ispis vremena izvođenja
    */
    cout << "\nVrijeme alociranja i prijenosa podataka na GPU: ";
    benchmarking(cpu_2_gpu_start, cpu_2_gpu_end);

    cout << "\nVrijeme računanja (na GPU-u): ";
    benchmarking(calc_start, calc_end);

    cout << "\nUKUPNO VRIJEME IZVOĐENJA: ";
    benchmarking(total_time_start, total_time_end);
    
    /*
        Oslobađanje memorije
    */
    HIP_ASSERT(hipFree(device_a));
    HIP_ASSERT(hipFree(device_h));
    HIP_ASSERT(hipFree(device_y));

    free(host_a);
    free(host_h);
    free(host_y);
    
    return 0;
}

__host__ __device__ double f (double x){
    /*
        Podintegralna funkcija
    */
    return sin(x);
}

__global__ void GPU_trapeze_formula(double*  y, const double*  a, const double*  h, const long n){
    int blockID = hipBlockIdx_x + hipBlockIdx_y * hipGridDim_x;
    int threadID = blockID * (hipBlockDim_x * hipBlockDim_y) + hipThreadIdx_x + (hipThreadIdx_y * hipBlockDim_x);

    if (threadID < NUMBER_OF_THREADS){
        double sum = 0;                                                                     // Suma elemenata koje računa jedan thread
        long batch = threadID * (long long) OPERATIONS_PER_THREAD;                          // Varijabla koja pozicionira thread na početak segmenta koji on treba obraditi 
        for (long i = batch; i < min(batch + (long)OPERATIONS_PER_THREAD, n+1); i++)
        {
            double x = *a + i * (*h);
            if(i == 0 || i == n)
            {
                sum += *h/2.0 * f(x);
            }
            else{
                sum += *h * f(x);
            }
        }
        y[threadID] = sum;                                                                  // Izlaz
    }
}

int grid_dimensions (long n){                                                // Vraća n dimenziju grida (grid je n x n dimenzija)
    return ceil(sqrt(NUMBER_OF_THREADS));
}

int valid_segment(double a, double b, long n){
    /*
        Priručna funkcija za provjeru valjanosti početnih podataka (funkcionalna samo za neke funkcije, detektira jedino intervale prekinutosti > 0.001)
    */

    double i = a;
    if (a >= b){                                                            // Prvi uneseni broj (a) se smatra početkom segmenta i mora biti manji od broja b
        cout << "\nPočetak segmenta mora biti manji od kraja segmenta." << endl;
        return 1;
    }
    while (i <= b){                                                         // Funkcija mora biti neprekinuta na segmentu
        if(isnan(f(i))){
            cout << "\nFunkcija nije neprekinuta na segmentu, ponovi unos." << endl;
            return 1;
        }
        else{
            i += 0.001;
        }
    }
    if (n < 0){                                                             // Broj segmenata mora biti pozitivan broj
        cout << "\nBroj segmenata mora biti pozitivan cijeli broj." << endl;
        return 1;
    }
    return 0;
}

void benchmarking (chrono::_V2::high_resolution_clock::time_point start, chrono::_V2::high_resolution_clock::time_point end){
    using namespace chrono;

    /*
        Točno vrijeme u mikrosekundama
    */
    auto exact_vrijeme = duration_cast<microseconds>(end - start);          // Točno vrijeme izvođenja u mikrosekundama
    cout << exact_vrijeme.count() << " mikrosekundi" << endl;               // Ispis točnog vremena izvođenja u mikrosekundama

    /*
        Human readable vrijeme
    */
    cout << "Human readable format:        ";

    auto minute = duration_cast<minutes>(exact_vrijeme);                    // Iz vremena u mikrosekundama uzima broj cijelih sekundi
    cout << minute.count() << " min ";
    exact_vrijeme -= duration_cast<microseconds>(minute);                   // Vrijeme u sekundama pretvara natrag u mikrosekunde i oduzima od ukupnog vremena u mikrosekundama

    auto sekunde = duration_cast<seconds>(exact_vrijeme);
    cout << sekunde.count() << " s ";
    exact_vrijeme -= duration_cast<microseconds>(sekunde);

    auto milisekunde = duration_cast<milliseconds>(exact_vrijeme);
    cout << milisekunde.count() << " ms ";
    exact_vrijeme -= duration_cast<microseconds>(milisekunde);

    cout << exact_vrijeme.count() << " μs" << endl;
}