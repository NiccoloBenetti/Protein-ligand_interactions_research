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

__global__ void calculateHydrogenBondKernel(float* donor_x, float* donor_y, float* donor_z,
                                            float* hydrogen_x, float* hydrogen_y, float* hydrogen_z,
                                            float* acceptor_x, float* acceptor_y, float* acceptor_z,
                                            float* distances, float* angles, int numDonors, int numAcceptors) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Indice per il donatore
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // Indice per l'accettore

    if (i < numDonors && j < numAcceptors) {
        // Calcolo della distanza euclidea tra donatore e accettore
        float dx = donor_x[i] - acceptor_x[j];
        float dy = donor_y[i] - acceptor_y[j];
        float dz = donor_z[i] - acceptor_z[j];
        distances[i * numAcceptors + j] = sqrtf(dx * dx + dy * dy + dz * dz);

        // Calcolo dell'angolo tra donatore, idrogeno e accettore
        float hx = hydrogen_x[i], hy = hydrogen_y[i], hz = hydrogen_z[i];
        float dhx = donor_x[i] - hx, dhy = donor_y[i] - hy, dhz = donor_z[i] - hz;
        float ahx = acceptor_x[j] - hx, ahy = acceptor_y[j] - hy, ahz = acceptor_z[j] - hz;

        float dotProduct = dhx * ahx + dhy * ahy + dhz * ahz;
        float mag_dh = sqrtf(dhx * dhx + dhy * dhy + dhz * dhz);
        float mag_ah = sqrtf(ahx * ahx + ahy * ahy + ahz * ahz);
        angles[i * numAcceptors + j] = acosf(dotProduct / (mag_dh * mag_ah)) * 180.0f / M_PI;
    }
}

    // Funzione wrapper per chiamare il kernel CUDA per il calcolo dei legami a idrogeno
    extern "C" void launchHydrogenBondKernel(float* d_donor_x, float* d_donor_y, float* d_donor_z,
                                         float* d_hydrogen_x, float* d_hydrogen_y, float* d_hydrogen_z,
                                         float* d_acceptor_x, float* d_acceptor_y, float* d_acceptor_z,
                                         float* d_distances, float* d_angles,
                                         int numDonors, int numAcceptors, int blockSizeX, int blockSizeY) {
    // Definisci la dimensione del blocco e della griglia
    dim3 threadsPerBlock(blockSizeX, blockSizeY);  // Blocchi 2D di thread
    dim3 blocksPerGrid((numDonors + blockSizeX - 1) / blockSizeX, 
                       (numAcceptors + blockSizeY - 1) / blockSizeY);  // Griglia 2D di blocchi

    // Lancia il kernel CUDA bidimensionale per il calcolo dei legami a idrogeno
    calculateHydrogenBondKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_donor_x, d_donor_y, d_donor_z,
        d_hydrogen_x, d_hydrogen_y, d_hydrogen_z,
        d_acceptor_x, d_acceptor_y, d_acceptor_z,
        d_distances, d_angles,
        numDonors, numAcceptors);
}


