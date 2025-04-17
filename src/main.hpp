#ifdef __cplusplus
extern "C" {
#endif

// Dichiarazione di launchDistanceKernel2D con il parametro stream
void launchDistanceKernel2D(float* d_posA_x, float* d_posA_y, float* d_posA_z,
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
                             int numDonors, int numAcceptors, int blockSizeX, int blockSizeY,
                             float maxDistance, float minAngle1, float maxAngle1,
                             float minAngle2, float maxAngle2, cudaStream_t stream);


// Dichiarazione del wrapper per il kernel Cationi-Anioni
void launchIonicInteractionsKernel_CationAnion(float* d_cation_x, float* d_cation_y, float* d_cation_z,
                                               float* d_anion_x, float* d_anion_y, float* d_anion_z,
                                               float* d_distances, int numCations, int numAnions, 
                                               int blockSizeX, int blockSizeY, float maxDistance);

// Dichiarazione del wrapper per il kernel Cationi-Anelli Aromatici
void launchIonicInteractionsKernel_CationRing(float* d_cation_x, float* d_cation_y, float* d_cation_z,
                                              float* d_ring_centroid_x, float* d_ring_centroid_y, float* d_ring_centroid_z,
                                              float* d_ring_normal_x, float* d_ring_normal_y, float* d_ring_normal_z,
                                              float* d_distances, float* d_angles, int numCations, int numRings, 
                                              int blockSizeX, int blockSizeY, float maxDistance, float minAngle, float maxAngle);

#ifdef __cplusplus
}
#endif
