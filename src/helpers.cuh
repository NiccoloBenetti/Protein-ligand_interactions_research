#pragma once

#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>
#include <chrono>

// We will use this throughout for error checking.
void check_last_error ( ) {

    cudaError_t err;
    if ((err = cudaGetLastError()) != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << " : "
                  << __FILE__ << ", line " << __LINE__ << std::endl;
            exit(1);
    }
}


class GPUTimer {

    float time;
    const uint64_t gpu;
    cudaEvent_t ying, yang;

public:

    GPUTimer (uint64_t gpu=0) : gpu(gpu) {
        cudaSetDevice(gpu);
        cudaEventCreate(&ying);
        cudaEventCreate(&yang);
    }

    ~GPUTimer ( ) {
        cudaSetDevice(gpu);
        cudaEventDestroy(ying);
        cudaEventDestroy(yang);
    }

    void start ( ) {
        cudaSetDevice(gpu);
        cudaEventRecord(ying, 0);
    }

    void stop (std::string label) {
        cudaSetDevice(gpu);
        cudaEventRecord(yang, 0);
        cudaEventSynchronize(yang);
        cudaEventElapsedTime(&time, ying, yang);
        std::cout << "GPU TIMING: " << time << " ms (" << label << ")" << std::endl;
    }
};


class CPUTimer {
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;

public:
    // Metodo per avviare il timer
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    // Metodo per fermare il timer e stampare il tempo trascorso con un'etichetta
    void stop(std::string label) {
        end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end_time - start_time;
        std::cout << "CPU TIMING: " << duration.count() << " ms (" << label << ")" << std::endl;
    }
};

