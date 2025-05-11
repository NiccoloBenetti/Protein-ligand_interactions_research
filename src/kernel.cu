#include <cuda_runtime.h>
#include <cmath>
#include "main.hpp"

// ---------------------------------------------------------------------------------------------- LAUNCHERS ------------------------------------------------------------------------------------------------------------

 __global__ void calculateHydrophobicBondKernel(float* posA_x, float* posA_y, float* posA_z,
                                           float* posB_x, float* posB_y, float* posB_z,
                                           float* distances, int numA, int numB) {
    // Calcola gli indici bidimensionali del thread all'interno della griglia
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Indice per posA (molecola A)
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // Indice per posB (molecola B)
    int distance;

    // Verifica che gli indici siano validi
    if (i < numA && j < numB) {
        // Calcolo della distanza euclidea tra l'atomo i di molA e l'atomo j di molB
        float dx = posA_x[i] - posB_x[j];
        float dy = posA_y[i] - posB_y[j];
        float dz = posA_z[i] - posB_z[j];
        distance = sqrtf(dx * dx + dy * dy + dz * dz);
        
        if(distance <= DISTANCE_HYDROPHOBIC) { // Salva i risultati solo se rispettano i requisiti
            distances[i * numB + j] = distance;
        } else{
            distances[i * numB + j] = -1.0f;  // Nessuna interazione
        }
    }
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
        float distance = sqrtf(dx * dx + dy * dy + dz * dz);

        // Calcolo dell'angolo tra donatore, idrogeno e accettore
        float hx = hydrogen_x[i], hy = hydrogen_y[i], hz = hydrogen_z[i];
        float dhx = donor_x[i] - hx, dhy = donor_y[i] - hy, dhz = donor_z[i] - hz;
        float ahx = acceptor_x[j] - hx, ahy = acceptor_y[j] - hy, ahz = acceptor_z[j] - hz;

        float dotProduct = dhx * ahx + dhy * ahy + dhz * ahz;
        float mag_dh = sqrtf(dhx * dhx + dhy * dhy + dhz * dhz);
        float mag_ah = sqrtf(ahx * ahx + ahy * ahy + ahz * ahz);
        float angle = acosf(dotProduct / (mag_dh * mag_ah)) * 180.0f / M_PI;

        // Salva le distanze e gli angoli solo se soddisfano i criteri
        if (distance <= DISTANCE_HYDROGENBOND && angle >= MIN_ANGLE_HYDROGENBOND && angle <= MAX_ANGLE_HYDROGENBOND) {
            distances[i * numAcceptors + j] = distance;
            angles[i * numAcceptors + j] = angle;
        } else {
            distances[i * numAcceptors + j] = -1.0f;  // Usa un valore negativo per indicare nessuna interazione
        }
    }
}

__global__ void calculateHalogenBondKernel(float* donor_x, float* donor_y, float* donor_z,
                                           float* halogen_x, float* halogen_y, float* halogen_z,
                                           float* acceptor_x, float* acceptor_y, float* acceptor_z,
                                           float* any_x, float* any_y, float* any_z,
                                           float* distances, float* firstAngles, float* secondAngles,
                                           int numDonors, int numAcceptors) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Indice per i donatori
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // Indice per gli accettori

    if (i < numDonors && j < numAcceptors) {
        // Calcolo della distanza euclidea tra il donatore e l'accettore
        float dx = donor_x[i] - acceptor_x[j];
        float dy = donor_y[i] - acceptor_y[j];
        float dz = donor_z[i] - acceptor_z[j];
        float distance = sqrtf(dx * dx + dy * dy + dz * dz);

        // Calcolo degli angoli
        float hx = halogen_x[i], hy = halogen_y[i], hz = halogen_z[i];
        float ax = acceptor_x[j], ay = acceptor_y[j], az = acceptor_z[j];
        float anyx = any_x[j], anyy = any_y[j], anyz = any_z[j];

        // Primo angolo: tra donatore, alogeno e accettore
        float dhx = donor_x[i] - hx, dhy = donor_y[i] - hy, dhz = donor_z[i] - hz;
        float ahx = ax - hx, ahy = ay - hy, ahz = az - hz;
        float dotProduct1 = dhx * ahx + dhy * ahy + dhz * ahz;
        float mag_dh = sqrtf(dhx * dhx + dhy * dhy + dhz * dhz);
        float mag_ah = sqrtf(ahx * ahx + ahy * ahy + ahz * ahz);
        float firstAngle = acosf(dotProduct1 / (mag_dh * mag_ah)) * 180.0f / M_PI;

        // Secondo angolo: tra accettore, alogeno e "any"
        float ahhx = ax - hx, ahhy = ay - hy, ahhz = az - hz;
        float aax = anyx - ax, aay = anyy - ay, aaz = anyz - az;
        float dotProduct2 = ahhx * aax + ahhy * aay + ahhz * aaz;
        float mag_ahh = sqrtf(ahhx * ahhx + ahhy * ahhy + ahhz * ahhz);
        float mag_aa = sqrtf(aax * aax + aay * aay + aaz * aaz);
        float secondAngle = acosf(dotProduct2 / (mag_ahh * mag_aa)) * 180.0f / M_PI;

        // Salva le distanze e gli angoli solo se soddisfano i criteri
        if (distance <= DISTANCE_HALOGENBOND && firstAngle >= MIN_ANGLE1_HALOGENBOND && firstAngle <= MAX_ANGLE1_HALOGENBOND && 
            secondAngle >= MIN_ANGLE2_HALOGENBOND && secondAngle <= MAX_ANGLE2_HALOGENBOND) {
            distances[i * numAcceptors + j] = distance;
            firstAngles[i * numAcceptors + j] = firstAngle;
            secondAngles[i * numAcceptors + j] = secondAngle;
        } else {
            distances[i * numAcceptors + j] = -1.0f;  // Usa un valore negativo per indicare nessuna interazione
        }
    }
}

__global__ void calculateCationAnionKernel(float* cation_x, float* cation_y, float* cation_z,
                                           float* anion_x, float* anion_y, float* anion_z,
                                           float* distances, int numCations, int numAnions) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Indice per i cationi
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // Indice per gli anioni

    if (i < numCations && j < numAnions) {
        // Calcolo della distanza tra catione e anione
        float dx = cation_x[i] - anion_x[j];
        float dy = cation_y[i] - anion_y[j];
        float dz = cation_z[i] - anion_z[j];
        float distance = sqrtf(dx * dx + dy * dy + dz * dz);

        // Verifica se la distanza è inferiore alla distanza massima per l'interazione ionica
        if (distance <= DISTANCE_IONIC) {
            distances[i * numAnions + j] = distance;
        } else {
            distances[i * numAnions + j] = -1.0f;  // Nessuna interazione
        }
    }
}

__global__ void calculateCationRingKernel(float* cation_x, float* cation_y, float* cation_z,
                                          float* ring_centroid_x, float* ring_centroid_y, float* ring_centroid_z,
                                          float* ring_normal_x, float* ring_normal_y, float* ring_normal_z,
                                          float* distances, float* angles, int numCations, int numRings) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Indice per i cationi
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // Indice per gli anelli aromatici

    if (i < numCations && j < numRings) {
        // Calcolo della distanza tra catione e il centro dell'anello aromatico
        float dx = cation_x[i] - ring_centroid_x[j];
        float dy = cation_y[i] - ring_centroid_y[j];
        float dz = cation_z[i] - ring_centroid_z[j];
        float distance = sqrtf(dx * dx + dy * dy + dz * dz);

        // Verifica se la distanza è inferiore alla distanza massima
        if (distance <= DISTANCE_IONIC) {
            // Calcolo dell'angolo tra il catione e il vettore normale all'anello
            float dotProduct = dx * ring_normal_x[j] + dy * ring_normal_y[j] + dz * ring_normal_z[j];
            float magnitude_cation = sqrtf(dx * dx + dy * dy + dz * dz);
            float magnitude_normal = sqrtf(ring_normal_x[j] * ring_normal_x[j] +
                                           ring_normal_y[j] * ring_normal_y[j] +
                                           ring_normal_z[j] * ring_normal_z[j]);
            float angle = acosf(dotProduct / (magnitude_cation * magnitude_normal)) * 180.0f / M_PI;

            // Verifica se l'angolo è compreso nell'intervallo richiesto
            if (angle >= MIN_ANGLE_IONIC && angle <= MAX_ANGLE_IONIC) {
                distances[i * numRings + j] = distance;
                angles[i * numRings + j] = angle;
            } else {
                distances[i * numRings + j] = -1.0f;  // Nessuna interazione
            }
        } else {
            distances[i * numRings + j] = -1.0f;  // Nessuna interazione
        }
    }
}

__global__ void calculateMetalBondKernel(float* posA_x, float* posA_y, float* posA_z,
                                           float* posB_x, float* posB_y, float* posB_z,
                                           float* distances, int numA, int numB) {
    // Calcola gli indici bidimensionali del thread all'interno della griglia
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Indice per posA (molecola A)
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // Indice per posB (molecola B)
    int distance;

    // Verifica che gli indici siano validi
    if (i < numA && j < numB) {
        // Calcolo della distanza euclidea tra l'atomo i di molA e l'atomo j di molB
        float dx = posA_x[i] - posB_x[j];
        float dy = posA_y[i] - posB_y[j];
        float dz = posA_z[i] - posB_z[j];
        distance = sqrtf(dx * dx + dy * dy + dz * dz);
        
        if(distance <= DISTANCE_METAL) { // Salva i risultati solo se rispettano i requisiti
            distances[i * numB + j] = distance;
        } else{
            distances[i * numB + j] = -1.0f;  // Nessuna interazione
        }
    }
}

// ---------------------------------------------------------------------------------------------- LAUNCHERS ------------------------------------------------------------------------------------------------------------


// Funzione wrapper per chiamare il kernel CUDA bidimensionale
extern "C" void launchHydrophobicBondKernel(float* d_posA_x, float* d_posA_y, float* d_posA_z,
                                       float* d_posB_x, float* d_posB_y, float* d_posB_z,
                                       float* d_distances, int numA, int numB, int blockSizeX, int blockSizeY, cudaStream_t stream) {
    // Definisci la dimensione del blocco e della griglia
    dim3 threadsPerBlock(blockSizeX, blockSizeY);  // Blocchi 2D di thread
    dim3 blocksPerGrid((numA + blockSizeX - 1) / blockSizeX, 
                       (numB + blockSizeY - 1) / blockSizeY);  // Griglia 2D di blocchi

    // Lancia il kernel CUDA bidimensionale nel stream specificato
    calculateHydrophobicBondKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_posA_x, d_posA_y, d_posA_z,
                                                                             d_posB_x, d_posB_y, d_posB_z,
                                                                             d_distances, numA, numB);
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

extern "C" void launchHalogenBondKernel(float* d_donor_x, float* d_donor_y, float* d_donor_z,
                                        float* d_halogen_x, float* d_halogen_y, float* d_halogen_z,
                                        float* d_acceptor_x, float* d_acceptor_y, float* d_acceptor_z,
                                        float* d_any_x, float* d_any_y, float* d_any_z,
                                        float* d_distances, float* d_firstAngles, float* d_secondAngles,
                                        int numDonors, int numAcceptors, int blockSizeX, int blockSizeY, cudaStream_t stream) {
    // Definisci la dimensione dei blocchi e della griglia
    dim3 threadsPerBlock(blockSizeX, blockSizeY);
    dim3 blocksPerGrid((numDonors + blockSizeX - 1) / blockSizeX, 
                       (numAcceptors + blockSizeY - 1) / blockSizeY);

    // Lancia il kernel per il calcolo dei legami di alogeni nel stream specificato
    calculateHalogenBondKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_donor_x, d_donor_y, d_donor_z,
        d_halogen_x, d_halogen_y, d_halogen_z,
        d_acceptor_x, d_acceptor_y, d_acceptor_z,
        d_any_x, d_any_y, d_any_z,
        d_distances, d_firstAngles, d_secondAngles,
        numDonors, numAcceptors);
}

extern "C" void launchIonicInteractionsKernel_CationAnion(float* d_cation_x, float* d_cation_y, float* d_cation_z,
                                                          float* d_anion_x, float* d_anion_y, float* d_anion_z,
                                                          float* d_distances, int numCations, int numAnions, 
                                                          int blockSizeX, int blockSizeY) {
    dim3 threadsPerBlock(blockSizeX, blockSizeY);
    dim3 blocksPerGrid((numCations + blockSizeX - 1) / blockSizeX, 
                       (numAnions + blockSizeY - 1) / blockSizeY);

    // Lancia il kernel per Cationi-Anioni
    calculateCationAnionKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_cation_x, d_cation_y, d_cation_z,
        d_anion_x, d_anion_y, d_anion_z,
        d_distances, numCations, numAnions);
}

extern "C" void launchIonicInteractionsKernel_CationRing(float* d_cation_x, float* d_cation_y, float* d_cation_z,
                                                         float* d_ring_centroid_x, float* d_ring_centroid_y, float* d_ring_centroid_z,
                                                         float* d_ring_normal_x, float* d_ring_normal_y, float* d_ring_normal_z,
                                                         float* d_distances, float* d_angles, int numCations, int numRings, 
                                                         int blockSizeX, int blockSizeY) {
    dim3 threadsPerBlock(blockSizeX, blockSizeY);
    dim3 blocksPerGrid((numCations + blockSizeX - 1) / blockSizeX, 
                       (numRings + blockSizeY - 1) / blockSizeY);

    // Lancia il kernel per Cationi-Anelli Aromatici
    calculateCationRingKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_cation_x, d_cation_y, d_cation_z,
        d_ring_centroid_x, d_ring_centroid_y, d_ring_centroid_z,
        d_ring_normal_x, d_ring_normal_y, d_ring_normal_z,
        d_distances, d_angles, numCations, numRings);
}

extern "C" void launchMetalBondKernel(float* d_posA_x, float* d_posA_y, float* d_posA_z,
                                       float* d_posB_x, float* d_posB_y, float* d_posB_z,
                                       float* d_distances, int numA, int numB, int blockSizeX, int blockSizeY, cudaStream_t stream) {
    // Definisci la dimensione del blocco e della griglia
    dim3 threadsPerBlock(blockSizeX, blockSizeY);  // Blocchi 2D di thread
    dim3 blocksPerGrid((numA + blockSizeX - 1) / blockSizeX, 
                       (numB + blockSizeY - 1) / blockSizeY);  // Griglia 2D di blocchi

    // Lancia il kernel CUDA bidimensionale nel stream specificato
    calculateMetalBondKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_posA_x, d_posA_y, d_posA_z,
                                                                             d_posB_x, d_posB_y, d_posB_z,
                                                                             d_distances, numA, numB);
}






