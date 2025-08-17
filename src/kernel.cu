#include <cuda_runtime.h>
#include <cmath>
#include "main.hpp"

// ---------------------------------------------------------------------------------------------- KERNELS ------------------------------------------------------------------------------------------------------------

__global__ void calculateHydrophobicBondKernel(float* posA_x, float* posA_y, float* posA_z,
                                               float* posB_x, float* posB_y, float* posB_z,
                                               float* distances, int numA, int numB)
{
    // Mappatura coalescente: B -> X, A -> Y
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // indice su B
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // indice su A

    if (i < numA && j < numB) {
        float dx = posA_x[i] - posB_x[j];
        float dy = posA_y[i] - posB_y[j];
        float dz = posA_z[i] - posB_z[j];
        float distance = sqrtf(dx * dx + dy * dy + dz * dz);

        distances[i * numB + j] = (distance <= DISTANCE_HYDROPHOBIC) ? distance : -1.0f;
    }
}

// --- DOPO (COERENTE: B→X=j, A→Y=i) ---
#ifndef HBOND_MIN_COS
// 130° di default; per 120° usa -DHBOND_MIN_COS=-0.5f
#define HBOND_MIN_COS -0.64278764f
#endif

#ifndef HBOND_RELAX_IF_NO_H
// metti -DHBOND_RELAX_IF_NO_H=1 se vuoi il fallback "solo distanza"
#define HBOND_RELAX_IF_NO_H 0
#endif

__global__ void calculateHydrogenBondKernel(const float* __restrict__ donor_x,
                                            const float* __restrict__ donor_y,
                                            const float* __restrict__ donor_z,
                                            const float* __restrict__ hydrogen_x,
                                            const float* __restrict__ hydrogen_y,
                                            const float* __restrict__ hydrogen_z,
                                            const float* __restrict__ acceptor_x,
                                            const float* __restrict__ acceptor_y,
                                            const float* __restrict__ acceptor_z,
                                            float* __restrict__ distances,
                                            int numDonors, int numAcceptors)
{
    // B -> X (acceptors), A -> Y (donors)  **come negli altri kernel**
    const int j = blockIdx.x * blockDim.x + threadIdx.x;  // accettore
    const int i = blockIdx.y * blockDim.y + threadIdx.y;  // donatore
    if (i >= numDonors || j >= numAcceptors) return;

    const int idx = i * numAcceptors + j;

    // ---- distanza^2 e early-out ----
    const float dx = donor_x[i] - acceptor_x[j];
    const float dy = donor_y[i] - acceptor_y[j];
    const float dz = donor_z[i] - acceptor_z[j];
    const float d2 = dx*dx + dy*dy + dz*dz;
    const float thr2 = (float)(DISTANCE_HYDROGENBOND * DISTANCE_HYDROGENBOND);
    if (d2 > thr2) { distances[idx] = -1.0f; return; }

    // ---- vettori per l’angolo D–H…H–A ----
    const float hx = hydrogen_x[i], hy = hydrogen_y[i], hz = hydrogen_z[i];
    float dhx = donor_x[i]    - hx, dhy = donor_y[i]    - hy, dhz = donor_z[i]    - hz;
    float ahx = acceptor_x[j] - hx, ahy = acceptor_y[j] - hy, ahz = acceptor_z[j] - hz;

    // Fallback opzionale se l’H non è disponibile/valido (p.es. acque senza H espliciti)
    const float dh2 = dhx*dhx + dhy*dhy + dhz*dhz;
#if HBOND_RELAX_IF_NO_H
    if (dh2 < 1e-12f) { distances[idx] = sqrtf(d2); return; }
#else
    if (dh2 < 1e-12f) { distances[idx] = -1.0f; return; }
#endif

    const float ah2 = ahx*ahx + ahy*ahy + ahz*ahz;
    const float inv_dh = rsqrtf(dh2 + 1e-20f);
    const float inv_ah = rsqrtf(ah2 + 1e-20f);
    float cosang = (dhx*ahx + dhy*ahy + dhz*ahz) * (inv_dh * inv_ah);
    cosang = fminf(1.f, fmaxf(-1.f, cosang));

    // Angolo minimo: cosθ ≤ HBOND_MIN_COS (130° default)
    if (cosang > HBOND_MIN_COS) { distances[idx] = -1.0f; return; }

    distances[idx] = sqrtf(d2);
}


__global__ void calculateHalogenBondKernel(float* donor_x, float* donor_y, float* donor_z,
                                           float* halogen_x, float* halogen_y, float* halogen_z,
                                           float* acceptor_x, float* acceptor_y, float* acceptor_z,
                                           float* any_x, float* any_y, float* any_z,
                                           float* distances,int numDonors, int numAcceptors)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // donatore
    const int j = blockIdx.y * blockDim.y + threadIdx.y;  // accettore
    if (i >= numDonors || j >= numAcceptors) return;

    const int idx = i * numAcceptors + j;

    // ---- distanza^2 e early-out ----
    const float dx = donor_x[i] - acceptor_x[j];
    const float dy = donor_y[i] - acceptor_y[j];
    const float dz = donor_z[i] - acceptor_z[j];
    const float d2 = dx*dx + dy*dy + dz*dz;
    const float thr2 = (float)(DISTANCE_HALOGENBOND * DISTANCE_HALOGENBOND);
    if (d2 > thr2) { distances[idx] = -1.0f; return; }

    // Vettori per l’angolo 1: D–Hal … Hal–A
    const float hx = halogen_x[i], hy = halogen_y[i], hz = halogen_z[i];
    const float ax = acceptor_x[j], ay = acceptor_y[j], az = acceptor_z[j];
    const float anyx = any_x[j], anyy = any_y[j], anyz = any_z[j];

    const float dhx = donor_x[i]-hx,  dhy = donor_y[i]-hy,  dhz = donor_z[i]-hz;
    const float ahx = ax - hx,        ahy = ay - hy,        ahz = az - hz;

    float inv_dh = rsqrtf(dhx*dhx + dhy*dhy + dhz*dhz + 1e-20f);
    float inv_ah = rsqrtf(ahx*ahx + ahy*ahy + ahz*ahz + 1e-20f);
    float cos1 = (dhx*ahx + dhy*ahy + dhz*ahz) * (inv_dh * inv_ah);
    cos1 = fminf(1.f, fmaxf(-1.f, cos1));

    // 130–180° => cosθ ≤ cos(130°)
    if (cos1 > -0.64278764f) { distances[idx] = -1.0f; return; }

    // Vettori per l’angolo 2: Hal–A … A–Any
    const float ahhx = ax - hx, ahhy = ay - hy, ahhz = az - hz;
    const float aax  = anyx - ax, aay  = anyy - ay, aaz  = anyz - az;

    float inv_ahh = rsqrtf(ahhx*ahhx + ahhy*ahhy + ahhz*ahhz + 1e-20f);
    float inv_aa  = rsqrtf(aax*aax   + aay*aay   + aaz*aaz   + 1e-20f);
    float cos2 = (ahhx*aax + ahhy*aay + ahhz*aaz) * (inv_ahh * inv_aa);
    cos2 = fminf(1.f, fmaxf(-1.f, cos2));

    // 80–140° => cos(140°) ≤ cosθ ≤ cos(80°)
    if (!(cos2 >= -0.76604444f && cos2 <= 0.17364818f)) { distances[idx] = -1.0f; return; }

    // ---- sqrt/acos solo per i validi ----
    distances[idx] = sqrtf(d2);
}


__global__ void calculateCationAnionKernel(float* cation_x, float* cation_y, float* cation_z,
                                           float* anion_x,  float* anion_y,  float* anion_z,
                                           float* distances, int numCations, int numAnions)
{
    // B -> X (anions) ; A -> Y (cations)
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // anione
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // catione

    if (i < numCations && j < numAnions) {
        float dx = cation_x[i] - anion_x[j];
        float dy = cation_y[i] - anion_y[j];
        float dz = cation_z[i] - anion_z[j];
        float distance = sqrtf(dx * dx + dy * dy + dz * dz);

        distances[i * numAnions + j] = (distance <= DISTANCE_IONIC) ? distance : -1.0f;
    }
}

__global__ void calculateCationRingKernel(float* cation_x, float* cation_y, float* cation_z,
                                          float* ring_centroid_x, float* ring_centroid_y, float* ring_centroid_z,
                                          float* ring_normal_x,   float* ring_normal_y,   float* ring_normal_z,
                                          float* distances, float* angles, int numCations, int numRings)
{
    // B -> X (rings) ; A -> Y (cations)
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // anello
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // catione

    if (i < numCations && j < numRings) {
        float dx = cation_x[i] - ring_centroid_x[j];
        float dy = cation_y[i] - ring_centroid_y[j];
        float dz = cation_z[i] - ring_centroid_z[j];
        float distance = sqrtf(dx * dx + dy * dy + dz * dz);

        if (distance <= DISTANCE_IONIC) {
            float dotProduct = dx * ring_normal_x[j] + dy * ring_normal_y[j] + dz * ring_normal_z[j];
            float magnitude_cation = sqrtf(dx * dx + dy * dy + dz * dz);
            float magnitude_normal = sqrtf(ring_normal_x[j] * ring_normal_x[j] +
                                           ring_normal_y[j] * ring_normal_y[j] +
                                           ring_normal_z[j] * ring_normal_z[j]);
            float angle = acosf(dotProduct / (magnitude_cation * magnitude_normal)) * 180.0f / M_PI;

            if (!(angle >= MIN_ANGLE_IONIC && angle <= MAX_ANGLE_IONIC) ||
                angle == MIN_ANGLE_IONIC || angle == MAX_ANGLE_IONIC)
            {
                distances[i * numRings + j] = distance;
                angles[i * numRings + j]    = angle;
            } else {
                distances[i * numRings + j] = -1.0f;
            }
        } else {
            distances[i * numRings + j] = -1.0f;
        }
    }
}

__global__ void calculatePiStackingKernel(
    const float* __restrict__ centroidA_x, const float* __restrict__ centroidA_y, const float* __restrict__ centroidA_z,
    const float* __restrict__ normalA_x,   const float* __restrict__ normalA_y,   const float* __restrict__ normalA_z,
    const float* __restrict__ centroidB_x, const float* __restrict__ centroidB_y, const float* __restrict__ centroidB_z,
    const float* __restrict__ normalB_x,   const float* __restrict__ normalB_y,   const float* __restrict__ normalB_z,
    float* __restrict__ distances,
    float* __restrict__ planesAngles,
    float* __restrict__ normalCentroidAnglesA,
    float* __restrict__ normalCentroidAnglesB,
    int numA, int numB)
{
    // B -> X (j), A -> Y (i)  **COERENTE CON TUTTO IL PROGETTO**
    const int j = blockIdx.x * blockDim.x + threadIdx.x; // ring B
    const int i = blockIdx.y * blockDim.y + threadIdx.y; // ring A
    if (i >= numA || j >= numB) return;

    const int idx = i * numB + j;

    auto poison = [&](void){
        distances[idx]              = -1.0f;
        planesAngles[idx]           = 999.0f;
        normalCentroidAnglesA[idx]  = 999.0f;
        normalCentroidAnglesB[idx]  = 999.0f;
    };

    // ---- distanza^2 e early-out con soglia T-shape ----
    const float cax = centroidA_x[i], cay = centroidA_y[i], caz = centroidA_z[i];
    const float cbx = centroidB_x[j], cby = centroidB_y[j], cbz = centroidB_z[j];
    const float vx = cbx - cax, vy = cby - cay, vz = cbz - caz;
    const float v2 = vx*vx + vy*vy + vz*vz;
    const float thr2 = (float)(DISTANCE_TSHAPE * DISTANCE_TSHAPE);
    if (v2 > thr2) { poison(); return; }

    // ---- normali normalizzate ----
    float nax = normalA_x[i], nay = normalA_y[i], naz = normalA_z[i];
    float nbx = normalB_x[j], nby = normalB_y[j], nbz = normalB_z[j];
    const float inv_nA = rsqrtf(nax*nax + nay*nay + naz*naz + 1e-20f);
    const float inv_nB = rsqrtf(nbx*nbx + nby*nby + nbz*nbz + 1e-20f);
    nax *= inv_nA; nay *= inv_nA; naz *= inv_nA;
    nbx *= inv_nB; nby *= inv_nB; nbz *= inv_nB;

    // cos(piano–piano), usiamo |cos|
    float cos_nn = nax*nbx + nay*nby + naz*nbz;
    cos_nn = fminf(1.f, fmaxf(-1.f, cos_nn));
    const float abs_cnn = fabsf(cos_nn);

    // Prefiltro unione finestre: [0–30°] U [50–90°]
    // => |cos| ≥ cos30° (~0.8660)  OR  |cos| ≤ cos50° (~0.6428)
    if (!(abs_cnn >= 0.86602540f || abs_cnn <= 0.64278764f)) { poison(); return; }

    // cos(normale–centroide) per A→B e B→A, con |cos|
    const float inv_v = rsqrtf(v2 + 1e-20f);
    float cos_a = (nax*vx + nay*vy + naz*vz) * inv_v;
    float cos_b = (nbx*(-vx) + nby*(-vy) + nbz*(-vz)) * inv_v;
    cos_a = fminf(1.f, fmaxf(-1.f, cos_a));
    cos_b = fminf(1.f, fmaxf(-1.f, cos_b));

    // Prefiltro 0–33° ⇒ |cos| ≥ cos33° (~0.8387) per entrambi
    if (fabsf(cos_a) < 0.83867057f || fabsf(cos_b) < 0.83867057f) { poison(); return; }

    // ---- solo ora sqrt/acos (per scrivere valori "puliti" che l'host confronterà) ----
    const float dist = sqrtf(v2);
    distances[idx]              = dist;
    planesAngles[idx]           = acosf(fabsf(cos_nn)) * 180.0f / (float)M_PI; // [0,90]
    normalCentroidAnglesA[idx]  = acosf(fabsf(cos_a))  * 180.0f / (float)M_PI; // [0,90]
    normalCentroidAnglesB[idx]  = acosf(fabsf(cos_b))  * 180.0f / (float)M_PI; // [0,90]
}


__global__ void calculateMetalBondKernel(float* posA_x, float* posA_y, float* posA_z,
                                         float* posB_x, float* posB_y, float* posB_z,
                                         float* distances, int numA, int numB)
{
    // B -> X, A -> Y
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // indice su B
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // indice su A

    if (i < numA && j < numB) {
        float dx = posA_x[i] - posB_x[j];
        float dy = posA_y[i] - posB_y[j];
        float dz = posA_z[i] - posB_z[j];
        float distance = sqrtf(dx * dx + dy * dy + dz * dz);

        distances[i * numB + j] = (distance <= DISTANCE_METAL) ? distance : -1.0f;
    }
}

// ---------------------------------------------------------------------------------------------- LAUNCHERS ------------------------------------------------------------------------------------------------------------

extern "C" void launchHydrophobicBondKernel(float* d_posA_x, float* d_posA_y, float* d_posA_z,
                                            float* d_posB_x, float* d_posB_y, float* d_posB_z,
                                            float* d_distances, int numA, int numB,
                                            int blockSizeX, int blockSizeY, cudaStream_t stream)
{
    dim3 threadsPerBlock(blockSizeX, blockSizeY);
    // Coalescente: grid.x su B, grid.y su A
    dim3 blocksPerGrid((numB + blockSizeX - 1) / blockSizeX,
                       (numA + blockSizeY - 1) / blockSizeY);

    calculateHydrophobicBondKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_posA_x, d_posA_y, d_posA_z,
        d_posB_x, d_posB_y, d_posB_z,
        d_distances, numA, numB);
}

extern "C" void launchHydrogenBondKernel(float* d_donor_x, float* d_donor_y, float* d_donor_z,
                                         float* d_hydrogen_x, float* d_hydrogen_y, float* d_hydrogen_z,
                                         float* d_acceptor_x, float* d_acceptor_y, float* d_acceptor_z,
                                         float* d_distances,
                                         int numDonors, int numAcceptors,
                                         int blockSizeX, int blockSizeY)
{
    dim3 threadsPerBlock(blockSizeX, blockSizeY);
    // grid.x = B(acceptors), grid.y = A(donors)
    dim3 blocksPerGrid((numAcceptors + blockSizeX - 1) / blockSizeX,
                       (numDonors    + blockSizeY - 1) / blockSizeY);

    calculateHydrogenBondKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_donor_x, d_donor_y, d_donor_z,
        d_hydrogen_x, d_hydrogen_y, d_hydrogen_z,
        d_acceptor_x, d_acceptor_y, d_acceptor_z,
        d_distances,
        numDonors, numAcceptors);
}

extern "C" void launchHalogenBondKernel(float* d_donor_x, float* d_donor_y, float* d_donor_z,
                                        float* d_halogen_x, float* d_halogen_y, float* d_halogen_z,
                                        float* d_acceptor_x, float* d_acceptor_y, float* d_acceptor_z,
                                        float* d_any_x, float* d_any_y, float* d_any_z,
                                        float* d_distances, int numDonors, int numAcceptors,
                                        int blockSizeX, int blockSizeY, cudaStream_t stream)
{
    dim3 threadsPerBlock(blockSizeX, blockSizeY);
    // grid.x = B(acceptors), grid.y = A(donors)
    dim3 blocksPerGrid((numAcceptors + blockSizeX - 1) / blockSizeX,
                       (numDonors    + blockSizeY - 1) / blockSizeY);

    calculateHalogenBondKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_donor_x, d_donor_y, d_donor_z,
        d_halogen_x, d_halogen_y, d_halogen_z,
        d_acceptor_x, d_acceptor_y, d_acceptor_z,
        d_any_x, d_any_y, d_any_z,
        d_distances, numDonors, numAcceptors);
}

extern "C" void launchIonicInteractionsKernel_CationAnion(float* d_cation_x, float* d_cation_y, float* d_cation_z,
                                                          float* d_anion_x,  float* d_anion_y,  float* d_anion_z,
                                                          float* d_distances, int numCations, int numAnions,
                                                          int blockSizeX, int blockSizeY)
{
    dim3 threadsPerBlock(blockSizeX, blockSizeY);
    // grid.x = B(anions), grid.y = A(cations)
    dim3 blocksPerGrid((numAnions + blockSizeX - 1) / blockSizeX,
                       (numCations + blockSizeY - 1) / blockSizeY);

    calculateCationAnionKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_cation_x, d_cation_y, d_cation_z,
        d_anion_x,  d_anion_y,  d_anion_z,
        d_distances, numCations, numAnions);
}

extern "C" void launchIonicInteractionsKernel_CationRing(float* d_cation_x, float* d_cation_y, float* d_cation_z,
                                                         float* d_ring_centroid_x, float* d_ring_centroid_y, float* d_ring_centroid_z,
                                                         float* d_ring_normal_x,   float* d_ring_normal_y,   float* d_ring_normal_z,
                                                         float* d_distances, float* d_angles,
                                                         int numCations, int numRings,
                                                         int blockSizeX, int blockSizeY)
{
    dim3 threadsPerBlock(blockSizeX, blockSizeY);
    // grid.x = B(rings), grid.y = A(cations)
    dim3 blocksPerGrid((numRings   + blockSizeX - 1) / blockSizeX,
                       (numCations + blockSizeY - 1) / blockSizeY);

    calculateCationRingKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_cation_x, d_cation_y, d_cation_z,
        d_ring_centroid_x, d_ring_centroid_y, d_ring_centroid_z,
        d_ring_normal_x,   d_ring_normal_y,   d_ring_normal_z,
        d_distances, d_angles, numCations, numRings);
}

extern "C" void launchPiStackingKernel(float* d_centroidA_x, float* d_centroidA_y, float* d_centroidA_z,
                                       float* d_normalA_x,   float* d_normalA_y,   float* d_normalA_z,
                                       float* d_centroidB_x, float* d_centroidB_y, float* d_centroidB_z,
                                       float* d_normalB_x,   float* d_normalB_y,   float* d_normalB_z,
                                       float* d_distances, float* d_planesAngles,
                                       float* d_normalCentroidAnglesA, float* d_normalCentroidAnglesB,
                                       int numRingsA, int numRingsB,
                                       int blockSizeX, int blockSizeY)
{
    dim3 threadsPerBlock(blockSizeX, blockSizeY);
    // grid.x = B(ringsB), grid.y = A(ringsA)
    dim3 blocksPerGrid((numRingsB + blockSizeX - 1) / blockSizeX,
                       (numRingsA + blockSizeY - 1) / blockSizeY);

    calculatePiStackingKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_centroidA_x, d_centroidA_y, d_centroidA_z,
        d_normalA_x,   d_normalA_y,   d_normalA_z,
        d_centroidB_x, d_centroidB_y, d_centroidB_z,
        d_normalB_x,   d_normalB_y,   d_normalB_z,
        d_distances, d_planesAngles, d_normalCentroidAnglesA, d_normalCentroidAnglesB,
        numRingsA, numRingsB);
}

extern "C" void launchMetalBondKernel(float* d_posA_x, float* d_posA_y, float* d_posA_z,
                                      float* d_posB_x, float* d_posB_y, float* d_posB_z,
                                      float* d_distances, int numA, int numB,
                                      int blockSizeX, int blockSizeY, cudaStream_t stream)
{
    dim3 threadsPerBlock(blockSizeX, blockSizeY);
    // grid.x = B, grid.y = A
    dim3 blocksPerGrid((numB + blockSizeX - 1) / blockSizeX,
                       (numA + blockSizeY - 1) / blockSizeY);

    calculateMetalBondKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_posA_x, d_posA_y, d_posA_z,
        d_posB_x, d_posB_y, d_posB_z,
        d_distances, numA, numB);
}
