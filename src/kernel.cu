/**
 * @file kernel.cu
 * @brief CUDA kernel implementations for molecular interaction detection.
 *
 * This file contains the GPU-accelerated kernels used to compute
 * intermolecular interactions such as:
 * - Hydrophobic contacts
 * - Hydrogen bonds
 * - Halogen bonds
 * - Ionic interactions (cation–anion, cation–ring)
 * - π-stacking
 * - Metal coordination
 *
 * Each kernel operates on device memory arrays containing atomic
 * coordinates and outputs computed geometric properties (distances,
 * angles, planes) for further analysis.
 *
 * The kernels are launched via wrapper functions declared in helpers.cuh
 * and invoked by main.cpp.
 */

#include <cuda_runtime.h>
#include <cmath>
#include "main.hpp"

// ---------------------------------------------------------------------------------------------- KERNELS ------------------------------------------------------------------------------------------------------------

/**
 * @brief CUDA kernel to compute hydrophobic interactions between two molecules.
 *
 * This kernel calculates the pairwise distances between atoms of molecule A and B
 * and marks interactions as valid if they fall within the hydrophobic cutoff.
 *
 * @param posA_x Device array of x-coordinates for atoms in molecule A
 * @param posA_y Device array of y-coordinates for atoms in molecule A
 * @param posA_z Device array of z-coordinates for atoms in molecule A
 * @param posB_x Device array of x-coordinates for atoms in molecule B
 * @param posB_y Device array of y-coordinates for atoms in molecule B
 * @param posB_z Device array of z-coordinates for atoms in molecule B
 * @param distances Device array to store computed atom-atom distances (-1 if beyond cutoff)
 * @param numA Number of atoms in molecule A
 * @param numB Number of atoms in molecule B
 * @return void
 */
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

/**
 * @brief Thresholds and options for hydrogen bond detection.
 *
 * These macros control the geometric constraints used in CUDA kernels
 * to detect hydrogen bonds between donor and acceptor atoms.
 */

/**
 * @brief Minimum cosine of the donor–H–acceptor angle.
 *
 * Default is -0.64278764 (circa 130°).  
 * Compile time override: -DHBOND_MIN_COS=-0.5f sets it to circa 120°.
 */
#ifndef HBOND_MIN_COS
#define HBOND_MIN_COS -0.64278764f
#endif

/**
 * @brief Relaxation option if no explicit hydrogen is present.
 *
 * Default is 0 (disabled).  
 * Compile-time override: -DHBOND_RELAX_IF_NO_H=1 enables fallback
 * distance-only detection when the hydrogen position is missing.
 */
#ifndef HBOND_RELAX_IF_NO_H
#define HBOND_RELAX_IF_NO_H 0
#endif

/**
 * @brief CUDA kernel to compute hydrogen bond interactions between donor and acceptor atoms.
 *
 * This kernel calculates donor–acceptor distances and verifies angular constraints
 * on the D–H...A geometry. Optionally, it supports a fallback distance-only check
 * if explicit hydrogens are not available (controlled by HBOND_RELAX_IF_NO_H).
 *
 * @param donor_x Device array of x-coordinates for donor atoms
 * @param donor_y Device array of y-coordinates for donor atoms
 * @param donor_z Device array of z-coordinates for donor atoms
 * @param hydrogen_x Device array of x-coordinates for donor hydrogens
 * @param hydrogen_y Device array of y-coordinates for donor hydrogens
 * @param hydrogen_z Device array of z-coordinates for donor hydrogens
 * @param acceptor_x Device array of x-coordinates for acceptor atoms
 * @param acceptor_y Device array of y-coordinates for acceptor atoms
 * @param acceptor_z Device array of z-coordinates for acceptor atoms
 * @param distances Device array to store valid donor–acceptor distances (-1 if not valid)
 * @param numDonors Number of donor atoms
 * @param numAcceptors Number of acceptor atoms
 * @return void
 */
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

    // distanza^2 e early-out
    const float dx = donor_x[i] - acceptor_x[j];
    const float dy = donor_y[i] - acceptor_y[j];
    const float dz = donor_z[i] - acceptor_z[j];
    const float d2 = dx*dx + dy*dy + dz*dz;
    const float thr2 = (float)(DISTANCE_HYDROGENBOND * DISTANCE_HYDROGENBOND);
    if (d2 > thr2) { distances[idx] = -1.0f; return; }

    // vettori per l’angolo D–H...H–A
    const float hx = hydrogen_x[i], hy = hydrogen_y[i], hz = hydrogen_z[i];
    float dhx = donor_x[i]    - hx, dhy = donor_y[i]    - hy, dhz = donor_z[i]    - hz;
    float ahx = acceptor_x[j] - hx, ahy = acceptor_y[j] - hy, ahz = acceptor_z[j] - hz;

    // Fallback opzionale se l’H non è disponibile o valido
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

    // Angolo minimo: cosθ <= HBOND_MIN_COS (130° default)
    if (cosang > HBOND_MIN_COS) { distances[idx] = -1.0f; return; }

    distances[idx] = sqrtf(d2);
}

/**
 * @brief CUDA kernel to compute halogen bond interactions between donor and acceptor atoms.
 *
 * This kernel checks donor–halogen–acceptor geometry and validates
 * halogen bonds based on distance and angular constraints, including
 * the secondary angle with the acceptor–any connected atom.
 *
 * @param donor_x Device array of x-coordinates for donor atoms
 * @param donor_y Device array of y-coordinates for donor atoms
 * @param donor_z Device array of z-coordinates for donor atoms
 * @param halogen_x Device array of x-coordinates for halogen atoms
 * @param halogen_y Device array of y-coordinates for halogen atoms
 * @param halogen_z Device array of z-coordinates for halogen atoms
 * @param acceptor_x Device array of x-coordinates for acceptor atoms
 * @param acceptor_y Device array of y-coordinates for acceptor atoms
 * @param acceptor_z Device array of z-coordinates for acceptor atoms
 * @param any_x Device array of x-coordinates for atoms bonded to the acceptor
 * @param any_y Device array of y-coordinates for atoms bonded to the acceptor
 * @param any_z Device array of z-coordinates for atoms bonded to the acceptor
 * @param distances Device array to store valid donor–acceptor distances (-1 if not valid)
 * @param numDonors Number of donor atoms
 * @param numAcceptors Number of acceptor atoms
 * @return void
 */
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

    // distanza^2 e early-out 
    const float dx = donor_x[i] - acceptor_x[j];
    const float dy = donor_y[i] - acceptor_y[j];
    const float dz = donor_z[i] - acceptor_z[j];
    const float d2 = dx*dx + dy*dy + dz*dz;
    const float thr2 = (float)(DISTANCE_HALOGENBOND * DISTANCE_HALOGENBOND);
    if (d2 > thr2) { distances[idx] = -1.0f; return; }

    // Vettori per l’angolo 1: D–Hal...Hal–A
    const float hx = halogen_x[i], hy = halogen_y[i], hz = halogen_z[i];
    const float ax = acceptor_x[j], ay = acceptor_y[j], az = acceptor_z[j];
    const float anyx = any_x[j], anyy = any_y[j], anyz = any_z[j];

    const float dhx = donor_x[i]-hx,  dhy = donor_y[i]-hy,  dhz = donor_z[i]-hz;
    const float ahx = ax - hx,        ahy = ay - hy,        ahz = az - hz;

    float inv_dh = rsqrtf(dhx*dhx + dhy*dhy + dhz*dhz + 1e-20f);
    float inv_ah = rsqrtf(ahx*ahx + ahy*ahy + ahz*ahz + 1e-20f);
    float cos1 = (dhx*ahx + dhy*ahy + dhz*ahz) * (inv_dh * inv_ah);
    cos1 = fminf(1.f, fmaxf(-1.f, cos1));

    // 130-180 => cosθ <= cos(130)
    if (cos1 > -0.64278764f) { distances[idx] = -1.0f; return; }

    // Vettori per l’angolo 2: Hal–A...A–Any
    const float ahhx = ax - hx, ahhy = ay - hy, ahhz = az - hz;
    const float aax  = anyx - ax, aay  = anyy - ay, aaz  = anyz - az;

    float inv_ahh = rsqrtf(ahhx*ahhx + ahhy*ahhy + ahhz*ahhz + 1e-20f);
    float inv_aa  = rsqrtf(aax*aax   + aay*aay   + aaz*aaz   + 1e-20f);
    float cos2 = (ahhx*aax + ahhy*aay + ahhz*aaz) * (inv_ahh * inv_aa);
    cos2 = fminf(1.f, fmaxf(-1.f, cos2));

    // 80–140 => cos(140) <= cosθ <= cos(80)
    if (!(cos2 >= -0.76604444f && cos2 <= 0.17364818f)) { distances[idx] = -1.0f; return; }

    // sqrt/acos solo per i validi
    distances[idx] = sqrtf(d2);
}

/**
 * @brief CUDA kernel to compute ionic interactions between cations and anions.
 *
 * This kernel calculates pairwise distances between cation–anion pairs
 * and marks them as valid interactions if they fall within the ionic cutoff.
 *
 * @param cation_x Device array of x-coordinates for cations
 * @param cation_y Device array of y-coordinates for cations
 * @param cation_z Device array of z-coordinates for cations
 * @param anion_x Device array of x-coordinates for anions
 * @param anion_y Device array of y-coordinates for anions
 * @param anion_z Device array of z-coordinates for anions
 * @param distances Device array to store cation–anion distances (-1 if not valid)
 * @param numCations Number of cations
 * @param numAnions Number of anions
 * @return void
 */
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

/**
 * @brief CUDA kernel to compute ionic interactions between cations and aromatic rings.
 *
 * This kernel calculates cation–centroid distances and evaluates the angle
 * between the cation–centroid vector and the ring plane normal. Interactions
 * are marked valid if both distance and angular constraints are satisfied.
 *
 * @param cation_x Device array of x-coordinates for cations
 * @param cation_y Device array of y-coordinates for cations
 * @param cation_z Device array of z-coordinates for cations
 * @param ring_centroid_x Device array of x-coordinates for ring centroids
 * @param ring_centroid_y Device array of y-coordinates for ring centroids
 * @param ring_centroid_z Device array of z-coordinates for ring centroids
 * @param ring_normal_x Device array of x-components of ring plane normals
 * @param ring_normal_y Device array of y-components of ring plane normals
 * @param ring_normal_z Device array of z-components of ring plane normals
 * @param distances Device array to store valid cation–centroid distances (-1 if not valid)
 * @param angles Device array to store corresponding cation–ring angles
 * @param numCations Number of cations
 * @param numRings Number of aromatic rings
 * @return void
 */
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

/**
 * @brief CUDA kernel to compute π-stacking interactions between aromatic rings.
 *
 * This kernel evaluates centroid–centroid distances, plane–plane angles,
 * and normal–centroid angles to identify parallel (sandwich) and perpendicular
 * (T-shape) π-stacking arrangements between two sets of rings.
 *
 * @param centroidA_x Device array of x-coordinates for centroids of rings in set A
 * @param centroidA_y Device array of y-coordinates for centroids of rings in set A
 * @param centroidA_z Device array of z-coordinates for centroids of rings in set A
 * @param normalA_x Device array of x-components of normals for rings in set A
 * @param normalA_y Device array of y-components of normals for rings in set A
 * @param normalA_z Device array of z-components of normals for rings in set A
 * @param centroidB_x Device array of x-coordinates for centroids of rings in set B
 * @param centroidB_y Device array of y-coordinates for centroids of rings in set B
 * @param centroidB_z Device array of z-coordinates for centroids of rings in set B
 * @param normalB_x Device array of x-components of normals for rings in set B
 * @param normalB_y Device array of y-components of normals for rings in set B
 * @param normalB_z Device array of z-components of normals for rings in set B
 * @param distances Device array to store centroid–centroid distances (-1 if not valid)
 * @param planesAngles Device array to store angles between ring planes (degrees)
 * @param normalCentroidAnglesA Device array to store normal–centroid angles for rings in set A (degrees)
 * @param normalCentroidAnglesB Device array to store normal–centroid angles for rings in set B (degrees)
 * @param numA Number of aromatic rings in set A
 * @param numB Number of aromatic rings in set B
 * @return void
 */
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
    // B -> X (j), A -> Y (i)
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

    //  distanza^2 e early-out con soglia T-shape 
    const float cax = centroidA_x[i], cay = centroidA_y[i], caz = centroidA_z[i];
    const float cbx = centroidB_x[j], cby = centroidB_y[j], cbz = centroidB_z[j];
    const float vx = cbx - cax, vy = cby - cay, vz = cbz - caz;
    const float v2 = vx*vx + vy*vy + vz*vz;
    const float thr2 = (float)(DISTANCE_TSHAPE * DISTANCE_TSHAPE);
    if (v2 > thr2) { poison(); return; }

    //  normali normalizzate 
    float nax = normalA_x[i], nay = normalA_y[i], naz = normalA_z[i];
    float nbx = normalB_x[j], nby = normalB_y[j], nbz = normalB_z[j];
    const float inv_nA = rsqrtf(nax*nax + nay*nay + naz*naz + 1e-20f);
    const float inv_nB = rsqrtf(nbx*nbx + nby*nby + nbz*nbz + 1e-20f);
    nax *= inv_nA; nay *= inv_nA; naz *= inv_nA;
    nbx *= inv_nB; nby *= inv_nB; nbz *= inv_nB;

    // cos(piano-piano), usiamo |cos|
    float cos_nn = nax*nbx + nay*nby + naz*nbz;
    cos_nn = fminf(1.f, fmaxf(-1.f, cos_nn));
    const float abs_cnn = fabsf(cos_nn);

    // Prefiltro unione finestre: [0–30] U [50–90]
    if (!(abs_cnn >= 0.86602540f || abs_cnn <= 0.64278764f)) { poison(); return; }

    // cos(normale-centroide) per A->B e B->A, con |cos|
    const float inv_v = rsqrtf(v2 + 1e-20f);
    float cos_a = (nax*vx + nay*vy + naz*vz) * inv_v;
    float cos_b = (nbx*(-vx) + nby*(-vy) + nbz*(-vz)) * inv_v;
    cos_a = fminf(1.f, fmaxf(-1.f, cos_a));
    cos_b = fminf(1.f, fmaxf(-1.f, cos_b));

    // Prefiltro 0–33 -> |cos| >= cos33 per entrambi
    if (fabsf(cos_a) < 0.83867057f || fabsf(cos_b) < 0.83867057f) { poison(); return; }

    // solo ora sqrt/acos
    const float dist = sqrtf(v2);
    distances[idx]              = dist;
    planesAngles[idx]           = acosf(fabsf(cos_nn)) * 180.0f / (float)M_PI; 
    normalCentroidAnglesA[idx]  = acosf(fabsf(cos_a))  * 180.0f / (float)M_PI; 
    normalCentroidAnglesB[idx]  = acosf(fabsf(cos_b))  * 180.0f / (float)M_PI; 
}

/**
 * @brief CUDA kernel to compute metal coordination interactions between metal and chelating atoms.
 *
 * This kernel calculates pairwise distances between metal-chelator atom pairs
 * and marks them as valid interactions if they fall within the metal coordination cutoff.
 *
 * @param posA_x Device array of x-coordinates for metal atoms
 * @param posA_y Device array of y-coordinates for metal atoms
 * @param posA_z Device array of z-coordinates for metal atoms
 * @param posB_x Device array of x-coordinates for chelating atoms
 * @param posB_y Device array of y-coordinates for chelating atoms
 * @param posB_z Device array of z-coordinates for chelating atoms
 * @param distances Device array to store metal–chelator distances (-1 if not valid)
 * @param numA Number of metal atoms
 * @param numB Number of chelating atoms
 * @return void
 */
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

/**
 * @brief Launches the CUDA kernel to compute hydrophobic interactions.
 *
 * This wrapper configures the execution grid and block dimensions,
 * then invokes the hydrophobic bond kernel on the given CUDA stream.
 *
 * @param d_posA_x Device array of x-coordinates for atoms in molecule A
 * @param d_posA_y Device array of y-coordinates for atoms in molecule A
 * @param d_posA_z Device array of z-coordinates for atoms in molecule A
 * @param d_posB_x Device array of x-coordinates for atoms in molecule B
 * @param d_posB_y Device array of y-coordinates for atoms in molecule B
 * @param d_posB_z Device array of z-coordinates for atoms in molecule B
 * @param d_distances Device array to store computed atom–atom distances (-1 if beyond cutoff)
 * @param numA Number of atoms in molecule A
 * @param numB Number of atoms in molecule B
 * @param blockSizeX CUDA block size in the X dimension
 * @param blockSizeY CUDA block size in the Y dimension
 * @param stream CUDA stream for asynchronous execution
 * @return void
 */
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

/**
 * @brief Launches the CUDA kernel to compute hydrogen bond interactions.
 *
 * This wrapper sets up the grid and block dimensions, then calls the
 * hydrogen bond kernel to evaluate donor–acceptor pairs with distance
 * and angular constraints.
 *
 * @param d_donor_x Device array of x-coordinates for donor atoms
 * @param d_donor_y Device array of y-coordinates for donor atoms
 * @param d_donor_z Device array of z-coordinates for donor atoms
 * @param d_hydrogen_x Device array of x-coordinates for donor hydrogens
 * @param d_hydrogen_y Device array of y-coordinates for donor hydrogens
 * @param d_hydrogen_z Device array of z-coordinates for donor hydrogens
 * @param d_acceptor_x Device array of x-coordinates for acceptor atoms
 * @param d_acceptor_y Device array of y-coordinates for acceptor atoms
 * @param d_acceptor_z Device array of z-coordinates for acceptor atoms
 * @param d_distances Device array to store donor–acceptor distances (-1 if not valid)
 * @param numDonors Number of donor atoms
 * @param numAcceptors Number of acceptor atoms
 * @param blockSizeX CUDA block size in the X dimension
 * @param blockSizeY CUDA block size in the Y dimension
 * @return void
 */
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

/**
 * @brief Launches the CUDA kernel to compute halogen bond interactions.
 *
 * This wrapper sets up the grid and block dimensions, then invokes the
 * halogen bond kernel to evaluate donor–halogen–acceptor geometries with
 * distance and angular constraints (including the acceptor–any connected atom).
 *
 * @param d_donor_x Device array of x-coordinates for donor atoms
 * @param d_donor_y Device array of y-coordinates for donor atoms
 * @param d_donor_z Device array of z-coordinates for donor atoms
 * @param d_halogen_x Device array of x-coordinates for halogen atoms
 * @param d_halogen_y Device array of y-coordinates for halogen atoms
 * @param d_halogen_z Device array of z-coordinates for halogen atoms
 * @param d_acceptor_x Device array of x-coordinates for acceptor atoms
 * @param d_acceptor_y Device array of y-coordinates for acceptor atoms
 * @param d_acceptor_z Device array of z-coordinates for acceptor atoms
 * @param d_any_x Device array of x-coordinates for atoms bonded to acceptors
 * @param d_any_y Device array of y-coordinates for atoms bonded to acceptors
 * @param d_any_z Device array of z-coordinates for atoms bonded to acceptors
 * @param d_distances Device array to store donor–acceptor distances (-1 if not valid)
 * @param numDonors Number of donor atoms
 * @param numAcceptors Number of acceptor atoms
 * @param blockSizeX CUDA block size in the X dimension
 * @param blockSizeY CUDA block size in the Y dimension
 * @param stream CUDA stream for asynchronous execution
 * @return void
 */
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

/**
 * @brief Launches the CUDA kernel to compute ionic interactions between cations and anions.
 *
 * This wrapper configures the execution grid and block dimensions, then
 * invokes the cation–anion kernel to compute pairwise distances and
 * validate ionic interactions within the cutoff threshold.
 *
 * @param d_cation_x Device array of x-coordinates for cations
 * @param d_cation_y Device array of y-coordinates for cations
 * @param d_cation_z Device array of z-coordinates for cations
 * @param d_anion_x Device array of x-coordinates for anions
 * @param d_anion_y Device array of y-coordinates for anions
 * @param d_anion_z Device array of z-coordinates for anions
 * @param d_distances Device array to store cation–anion distances (-1 if not valid)
 * @param numCations Number of cations
 * @param numAnions Number of anions
 * @param blockSizeX CUDA block size in the X dimension
 * @param blockSizeY CUDA block size in the Y dimension
 * @return void
 */
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

/**
 * @brief Launches the CUDA kernel to compute ionic interactions between cations and aromatic rings.
 *
 * This wrapper sets up the grid and block dimensions, then calls the cation–ring
 * kernel to evaluate cation–centroid distances and the angle between the cation–centroid
 * vector and the ring plane normal.
 *
 * @param d_cation_x Device array of x-coordinates for cations
 * @param d_cation_y Device array of y-coordinates for cations
 * @param d_cation_z Device array of z-coordinates for cations
 * @param d_ring_centroid_x Device array of x-coordinates for ring centroids
 * @param d_ring_centroid_y Device array of y-coordinates for ring centroids
 * @param d_ring_centroid_z Device array of z-coordinates for ring centroids
 * @param d_ring_normal_x Device array of x-components of ring plane normals
 * @param d_ring_normal_y Device array of y-components of ring plane normals
 * @param d_ring_normal_z Device array of z-components of ring plane normals
 * @param d_distances Device array to store cation–centroid distances (-1 if not valid)
 * @param d_angles Device array to store corresponding cation–ring angles
 * @param numCations Number of cations
 * @param numRings Number of aromatic rings
 * @param blockSizeX CUDA block size in the X dimension
 * @param blockSizeY CUDA block size in the Y dimension
 * @return void
 */
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

/**
 * @brief Launches the CUDA kernel to compute π-stacking interactions between aromatic rings.
 *
 * This wrapper configures the grid and block dimensions, then calls the π-stacking
 * kernel to evaluate centroid–centroid distances, plane–plane angles, and normal–centroid
 * angles for parallel (sandwich) and perpendicular (T-shape) stacking.
 *
 * @param d_centroidA_x Device array of x-coordinates for centroids of rings in set A
 * @param d_centroidA_y Device array of y-coordinates for centroids of rings in set A
 * @param d_centroidA_z Device array of z-coordinates for centroids of rings in set A
 * @param d_normalA_x Device array of x-components of normals for rings in set A
 * @param d_normalA_y Device array of y-components of normals for rings in set A
 * @param d_normalA_z Device array of z-components of normals for rings in set A
 * @param d_centroidB_x Device array of x-coordinates for centroids of rings in set B
 * @param d_centroidB_y Device array of y-coordinates for centroids of rings in set B
 * @param d_centroidB_z Device array of z-coordinates for centroids of rings in set B
 * @param d_normalB_x Device array of x-components of normals for rings in set B
 * @param d_normalB_y Device array of y-components of normals for rings in set B
 * @param d_normalB_z Device array of z-components of normals for rings in set B
 * @param d_distances Device array to store centroid–centroid distances (-1 if not valid)
 * @param d_planesAngles Device array to store plane–plane angles (degrees)
 * @param d_normalCentroidAnglesA Device array to store normal–centroid angles for rings in set A (degrees)
 * @param d_normalCentroidAnglesB Device array to store normal–centroid angles for rings in set B (degrees)
 * @param numRingsA Number of aromatic rings in set A
 * @param numRingsB Number of aromatic rings in set B
 * @param blockSizeX CUDA block size in the X dimension
 * @param blockSizeY CUDA block size in the Y dimension
 * @return void
 */
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

/**
 * @brief Launches the CUDA kernel to compute metal coordination interactions.
 *
 * This wrapper sets up the grid and block dimensions, then calls the metal bond
 * kernel to evaluate distances between metal atoms and chelating atoms, marking
 * interactions valid if they fall within the metal coordination cutoff.
 *
 * @param d_posA_x Device array of x-coordinates for metal atoms
 * @param d_posA_y Device array of y-coordinates for metal atoms
 * @param d_posA_z Device array of z-coordinates for metal atoms
 * @param d_posB_x Device array of x-coordinates for chelating atoms
 * @param d_posB_y Device array of y-coordinates for chelating atoms
 * @param d_posB_z Device array of z-coordinates for chelating atoms
 * @param d_distances Device array to store metal–chelator distances (-1 if not valid)
 * @param numA Number of metal atoms
 * @param numB Number of chelating atoms
 * @param blockSizeX CUDA block size in the X dimension
 * @param blockSizeY CUDA block size in the Y dimension
 * @param stream CUDA stream for asynchronous execution
 * @return void
 */
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
