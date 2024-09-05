#ifdef __cplusplus
extern "C" {
#endif

// Dichiarazione di launchDistanceKernel2D
void launchDistanceKernel2D(float* d_posA_x, float* d_posA_y, float* d_posA_z,
                            float* d_posB_x, float* d_posB_y, float* d_posB_z,
                            float* d_distances, int numA, int numB, int blockSizeX, int blockSizeY);

// Dichiarazione di launchHydrogenBondKernel (devi aggiungere questa parte)
void launchHydrogenBondKernel(float* d_donor_x, float* d_donor_y, float* d_donor_z,
                              float* d_hydrogen_x, float* d_hydrogen_y, float* d_hydrogen_z,
                              float* d_acceptor_x, float* d_acceptor_y, float* d_acceptor_z,
                              float* d_distances, float* d_angles,
                              int numDonors, int numAcceptors, int blockSizeX, int blockSizeY);

#ifdef __cplusplus
}
#endif
