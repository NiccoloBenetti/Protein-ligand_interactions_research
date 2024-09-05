#ifdef __cplusplus
extern "C" {
#endif

void launchDistanceKernel2D(float* d_posA_x, float* d_posA_y, float* d_posA_z,
                            float* d_posB_x, float* d_posB_y, float* d_posB_z,
                            float* d_distances, int numA, int numB, int blockSizeX, int blockSizeY);

#ifdef __cplusplus
}
#endif
