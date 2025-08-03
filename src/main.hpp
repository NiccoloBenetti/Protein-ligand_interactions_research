#ifdef __cplusplus
extern "C" {
#endif

 // HYDROPHOBIC 
#define DISTANCE_HYDROPHOBIC 4.5

// HYDROGEN BOND
#define DISTANCE_HYDROGENBOND 3.5
#define MIN_ANGLE_HYDROGENBOND 130
#define MAX_ANGLE_HYDROGENBOND 180

// HALOGEN BOND
#define DISTANCE_HALOGENBOND 3.5
#define MIN_ANGLE1_HALOGENBOND 130
#define MAX_ANGLE1_HALOGENBOND 180
#define MIN_ANGLE2_HALOGENBOND 80
#define MAX_ANGLE2_HALOGENBOND 140

// IONIC
#define DISTANCE_IONIC 4.5
#define MIN_ANGLE_IONIC 30
#define MAX_ANGLE_IONIC 150

// PI STACKING - SANDWICH
#define DISTANCE_SANDWICH 5.5
#define MIN_PLANES_ANGLE_SANDWICH 0
#define MAX_PLANES_ANGLE_SANDWICH 30
#define MIN_NORMAL_CENTROID_ANGLE_SANDWICH 0
#define MAX_NORMAL_CENTROID_ANGLE_SANDWICH 33

// PI STACKING - T SHAPE
#define DISTANCE_TSHAPE 6.5
#define MIN_PLANES_ANGLE_TSHAPE 50
#define MAX_PLANES_ANGLE_TSHAPE 90
#define MIN_NORMAL_CENTROID_ANGLE_TSHAPE 0
#define MAX_NORMAL_CENTROID_ANGLE_TSHAPE 30

// METAL COORDINATION
#define DISTANCE_METAL 2.8


// PARAMETRI DI CONFIGURAZIONE CUDA (GPU)
#define BLOCKSIZEX 8
#define BLOCKSIZEY 1
#define NUM_STREAMS 1 


// Dichiarazione di launchDistanceKernel2D con il parametro stream
void launchHydrophobicBondKernel(float* d_posA_x, float* d_posA_y, float* d_posA_z,
                            float* d_posB_x, float* d_posB_y, float* d_posB_z,
                            float* d_distances, int numA, int numB, int blockSizeX, int blockSizeY, cudaStream_t stream);

// Dichiarazione di launchHydrogenBondKernel
void launchHydrogenBondKernel(float* d_donor_x, float* d_donor_y, float* d_donor_z,
                              float* d_hydrogen_x, float* d_hydrogen_y, float* d_hydrogen_z,
                              float* d_acceptor_x, float* d_acceptor_y, float* d_acceptor_z,
                              float* d_distances, float* d_angles,
                              int numDonors, int numAcceptors, int blockSizeX, int blockSizeY);

// Dichiarazione di launchHalogenBondKernel con il parametro stream
void launchHalogenBondKernel(float* d_donor_x, float* d_donor_y, float* d_donor_z,
                             float* d_halogen_x, float* d_halogen_y, float* d_halogen_z,
                             float* d_acceptor_x, float* d_acceptor_y, float* d_acceptor_z,
                             float* d_any_x, float* d_any_y, float* d_any_z,
                             float* d_distances, float* d_firstAngles, float* d_secondAngles,
                             int numDonors, int numAcceptors, int blockSizeX, int blockSizeY, cudaStream_t stream);


// Dichiarazione del wrapper per il kernel Cationi-Anioni
void launchIonicInteractionsKernel_CationAnion(float* d_cation_x, float* d_cation_y, float* d_cation_z,
                                               float* d_anion_x, float* d_anion_y, float* d_anion_z,
                                               float* d_distances, int numCations, int numAnions, 
                                               int blockSizeX, int blockSizeY);

// Dichiarazione del wrapper per il kernel Cationi-Anelli Aromatici
void launchIonicInteractionsKernel_CationRing(float* d_cation_x, float* d_cation_y, float* d_cation_z,
                                              float* d_ring_centroid_x, float* d_ring_centroid_y, float* d_ring_centroid_z,
                                              float* d_ring_normal_x, float* d_ring_normal_y, float* d_ring_normal_z,
                                              float* d_distances, float* d_angles, int numCations, int numRings, 
                                              int blockSizeX, int blockSizeY);

void launchMetalBondKernel(float* d_posA_x, float* d_posA_y, float* d_posA_z,
                            float* d_posB_x, float* d_posB_y, float* d_posB_z,
                            float* d_distances, int numA, int numB, int blockSizeX, int blockSizeY, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
