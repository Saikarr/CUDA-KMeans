#include "cuda_runtime.h"
#include <stdio.h>

// Function to perform k-means clustering using thrust
cudaError_t kmeans_thrust(const float* h_data, int* h_labels, int n, int k, int dims, int max_iter, float** points2Drotated, float* h_centroids);

// Function to perform k-means clustering using cuda kernels
cudaError_t kmeans(const float* h_data, int* h_labels, int n, int k, int dims, int max_iter, float* h_centroids);

// Function to perform k-means clustering using CPU
void kMeansCPU(float** points, int* clusterAssignment, int n, int k, int dimensions, int maxIterations, float* clusters);