/**
 * @file helpers.cuh
 * @brief CUDA helper definitions and kernel launch wrappers for molecular interaction detection.
 *
 * This header centralizes:
 * - Geometric cutoff values and angular constraints for all supported interactions
 *   (hydrophobic, hydrogen bond, halogen bond, ionic, π-stacking, metal coordination).
 * - Default CUDA execution parameters (block sizes, streams).
 * - Declarations of host wrapper functions that configure grid/block dimensions
 *   and launch the corresponding GPU kernels.
 *
 * All wrappers assume a coalesced memory mapping convention: molecule B->X dimension, molecule A->Y dimension.
 *
 * @note Constants can be adjusted at compile time to tune detection thresholds.
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Maximum atom–atom distance for hydrophobic interactions (Å).
 */
#define DISTANCE_HYDROPHOBIC 4.5

/**
 * @brief Hydrogen bond distance and angular constraints.
 */
#define DISTANCE_HYDROGENBOND 3.5
#define MIN_ANGLE_HYDROGENBOND 130
#define MAX_ANGLE_HYDROGENBOND 180

/**
 * @brief Halogen bond distance and angular constraints.
 */
#define DISTANCE_HALOGENBOND 3.5
#define MIN_ANGLE1_HALOGENBOND 130
#define MAX_ANGLE1_HALOGENBOND 180
#define MIN_ANGLE2_HALOGENBOND 80
#define MAX_ANGLE2_HALOGENBOND 140

/**
 * @brief Ionic interaction distance and angular constraints.
 */
#define DISTANCE_IONIC 4.5
#define MIN_ANGLE_IONIC 30
#define MAX_ANGLE_IONIC 150

/**
 * @brief π-stacking (sandwich) distance and angular constraints.
 */
#define DISTANCE_SANDWICH 5.5
#define MIN_PLANES_ANGLE_SANDWICH 0
#define MAX_PLANES_ANGLE_SANDWICH 30
#define MIN_NORMAL_CENTROID_ANGLE_SANDWICH 0
#define MAX_NORMAL_CENTROID_ANGLE_SANDWICH 33

/**
 * @brief π-stacking (T-shape) distance and angular constraints.
 */
#define DISTANCE_TSHAPE 6.5
#define MIN_PLANES_ANGLE_TSHAPE 50
#define MAX_PLANES_ANGLE_TSHAPE 90
#define MIN_NORMAL_CENTROID_ANGLE_TSHAPE 0
#define MAX_NORMAL_CENTROID_ANGLE_TSHAPE 30

/**
 * @brief Maximum metal–chelator distance for metal coordination (Å).
 */
#define DISTANCE_METAL 2.8

/**
 * @brief Default CUDA launch parameters.
 */
#define BLOCKSIZEX 128
#define BLOCKSIZEY 1
#define NUM_STREAMS 1


// Dichiarazione di launchDistanceKernel2D con il parametro stream

/**
 * @brief Launches the CUDA kernel to compute hydrophobic interactions.
 * @param d_posA_x Device array of x-coordinates for atoms in molecule A
 * @param d_posA_y Device array of y-coordinates for atoms in molecule A
 * @param d_posA_z Device array of z-coordinates for atoms in molecule A
 * @param d_posB_x Device array of x-coordinates for atoms in molecule B
 * @param d_posB_y Device array of y-coordinates for atoms in molecule B
 * @param d_posB_z Device array of z-coordinates for atoms in molecule B
 * @param d_distances Device array to store atom–atom distances (-1 if not valid)
 * @param numA Number of atoms in molecule A
 * @param numB Number of atoms in molecule B
 * @param blockSizeX CUDA block size in X dimension
 * @param blockSizeY CUDA block size in Y dimension
 * @param stream CUDA stream for asynchronous execution
 * @return void
 */
extern void launchHydrophobicBondKernel(float* d_posA_x, float* d_posA_y, float* d_posA_z,
                                        float* d_posB_x, float* d_posB_y, float* d_posB_z,
                                        float* d_distances, int numA, int numB,
                                        int blockSizeX, int blockSizeY, cudaStream_t stream);

/**
 * @brief Launches the CUDA kernel to compute hydrogen bond interactions.
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
 * @param blockSizeX CUDA block size in X dimension
 * @param blockSizeY CUDA block size in Y dimension
 * @return void
 */
extern void launchHydrogenBondKernel(float* d_donor_x, float* d_donor_y, float* d_donor_z,
                                     float* d_hydrogen_x, float* d_hydrogen_y, float* d_hydrogen_z,
                                     float* d_acceptor_x, float* d_acceptor_y, float* d_acceptor_z,
                                     float* d_distances,
                                     int numDonors, int numAcceptors,
                                     int blockSizeX, int blockSizeY);

/**
 * @brief Launches the CUDA kernel to compute halogen bond interactions.
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
 * @param blockSizeX CUDA block size in X dimension
 * @param blockSizeY CUDA block size in Y dimension
 * @param stream CUDA stream for asynchronous execution
 * @return void
 */
extern void launchHalogenBondKernel(float* d_donor_x, float* d_donor_y, float* d_donor_z,
                                    float* d_halogen_x, float* d_halogen_y, float* d_halogen_z,
                                    float* d_acceptor_x, float* d_acceptor_y, float* d_acceptor_z,
                                    float* d_any_x, float* d_any_y, float* d_any_z,
                                    float* d_distances,
                                    int numDonors, int numAcceptors,
                                    int blockSizeX, int blockSizeY, cudaStream_t stream);

/**
 * @brief Launches the CUDA kernel to compute ionic interactions (cation–anion).
 * @param d_cation_x Device array of x-coordinates for cations
 * @param d_cation_y Device array of y-coordinates for cations
 * @param d_cation_z Device array of z-coordinates for cations
 * @param d_anion_x Device array of x-coordinates for anions
 * @param d_anion_y Device array of y-coordinates for anions
 * @param d_anion_z Device array of z-coordinates for anions
 * @param d_distances Device array to store cation–anion distances (-1 if not valid)
 * @param numCations Number of cations
 * @param numAnions Number of anions
 * @param blockSizeX CUDA block size in X dimension
 * @param blockSizeY CUDA block size in Y dimension
 * @return void
 */
extern void launchIonicInteractionsKernel_CationAnion(float* d_cation_x, float* d_cation_y, float* d_cation_z,
                                                      float* d_anion_x, float* d_anion_y, float* d_anion_z,
                                                      float* d_distances, int numCations, int numAnions,
                                                      int blockSizeX, int blockSizeY);


/**
 * @brief Launches the CUDA kernel to compute ionic interactions (cation–aromatic ring).
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
 * @param d_angles Device array to store cation–ring angles
 * @param numCations Number of cations
 * @param numRings Number of aromatic rings
 * @param blockSizeX CUDA block size in X dimension
 * @param blockSizeY CUDA block size in Y dimension
 * @return void
 */
extern void launchIonicInteractionsKernel_CationRing(float* d_cation_x, float* d_cation_y, float* d_cation_z,
                                                     float* d_ring_centroid_x, float* d_ring_centroid_y, float* d_ring_centroid_z,
                                                     float* d_ring_normal_x, float* d_ring_normal_y, float* d_ring_normal_z,
                                                     float* d_distances, float* d_angles,
                                                     int numCations, int numRings,
                                                     int blockSizeX, int blockSizeY);

/**
 * @brief Launches the CUDA kernel to compute π-stacking interactions.
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
 * @param d_normalCentroidAnglesA Device array to store normal–centroid angles for set A
 * @param d_normalCentroidAnglesB Device array to store normal–centroid angles for set B
 * @param numRingsA Number of aromatic rings in set A
 * @param numRingsB Number of aromatic rings in set B
 * @param blockSizeX CUDA block size in X dimension
 * @param blockSizeY CUDA block size in Y dimension
 * @return void
 */
extern void launchPiStackingKernel(float* d_centroidA_x, float* d_centroidA_y, float* d_centroidA_z,
                                   float* d_normalA_x,   float* d_normalA_y,   float* d_normalA_z,
                                   float* d_centroidB_x, float* d_centroidB_y, float* d_centroidB_z,
                                   float* d_normalB_x,   float* d_normalB_y,   float* d_normalB_z,
                                   float* d_distances, float* d_planesAngles,
                                   float* d_normalCentroidAnglesA, float* d_normalCentroidAnglesB,
                                   int numRingsA, int numRingsB,
                                   int blockSizeX, int blockSizeY);

/**
 * @brief Launches the CUDA kernel to compute metal coordination interactions.
 * @param d_posA_x Device array of x-coordinates for metal atoms
 * @param d_posA_y Device array of y-coordinates for metal atoms
 * @param d_posA_z Device array of z-coordinates for metal atoms
 * @param d_posB_x Device array of x-coordinates for chelating atoms
 * @param d_posB_y Device array of y-coordinates for chelating atoms
 * @param d_posB_z Device array of z-coordinates for chelating atoms
 * @param d_distances Device array to store metal–chelator distances (-1 if not valid)
 * @param numA Number of metal atoms
 * @param numB Number of chelating atoms
 * @param blockSizeX CUDA block size in X dimension
 * @param blockSizeY CUDA block size in Y dimension
 * @param stream CUDA stream for asynchronous execution
 * @return void
 */
extern void launchMetalBondKernel(float* d_posA_x, float* d_posA_y, float* d_posA_z,
                                  float* d_posB_x, float* d_posB_y, float* d_posB_z,
                                  float* d_distances, int numA, int numB,
                                  int blockSizeX, int blockSizeY, cudaStream_t stream);


#ifdef __cplusplus
}
#endif
