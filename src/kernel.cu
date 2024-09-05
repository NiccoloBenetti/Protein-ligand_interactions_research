#include <cuda_runtime.h>
#include <cmath>

// Kernel CUDA bidimensionale per calcolare le distanze tra atomi
 __global__ void calculateDistancesKernel2D(float* posA_x, float* posA_y, float* posA_z,
                                           float* posB_x, float* posB_y, float* posB_z,
                                           float* distances, int numA, int numB) {
    // Calcola gli indici bidimensionali del thread all'interno della griglia
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Indice per posA (molecola A)
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // Indice per posB (molecola B)

    // Verifica che gli indici siano validi
    if (i < numA && j < numB) {
        // Calcolo della distanza euclidea tra l'atomo i di molA e l'atomo j di molB
        float dx = posA_x[i] - posB_x[j];
        float dy = posA_y[i] - posB_y[j];
        float dz = posA_z[i] - posB_z[j];
        distances[i * numB + j] = sqrtf(dx * dx + dy * dy + dz * dz);
    }
}

// Funzione wrapper per chiamare il kernel CUDA bidimensionale
extern "C" void launchDistanceKernel2D(float* d_posA_x, float* d_posA_y, float* d_posA_z,
                                       float* d_posB_x, float* d_posB_y, float* d_posB_z,
                                       float* d_distances, int numA, int numB, int blockSizeX, int blockSizeY) {
    // Definisci la dimensione del blocco e della griglia
    dim3 threadsPerBlock(blockSizeX, blockSizeY);  // Blocchi 2D di thread
    dim3 blocksPerGrid((numA + blockSizeX - 1) / blockSizeX, 
                       (numB + blockSizeY - 1) / blockSizeY);  // Griglia 2D di blocchi

    // Lancia il kernel CUDA bidimensionale
    calculateDistancesKernel2D<<<blocksPerGrid, threadsPerBlock>>>(d_posA_x, d_posA_y, d_posA_z,
                                                                   d_posB_x, d_posB_y, d_posB_z,
                                                                   d_distances, numA, numB);
}
