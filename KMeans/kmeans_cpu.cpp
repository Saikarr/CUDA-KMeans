#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "kernels.cuh"

// Function to compute the Euclidean distance between two points
float euclideanDistance(float* a, float* b, int dimensions) {
    float sum = 0.0;
    for (int i = 0; i < dimensions; ++i) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sum;
}

// Function to initialize centroids
void initializeCentroids(float** points, float** centroids, int n, int k, int dimensions) {
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < dimensions; ++j) {
            centroids[i][j] = points[i][j];
        }
    }
}

// Function to assign clusters to each point
void assignClusters(float** points, int* clusterAssignment, float** centroids, int n, int k, int dimensions, bool* changed) {
    for (int i = 0; i < n; ++i) {
        float minDistance = FLT_MAX;
        int bestCluster = 0;

        for (int j = 0; j < k; ++j) {
            float distance = euclideanDistance(points[i], centroids[j], dimensions);
            if (distance < minDistance) {
                minDistance = distance;
                bestCluster = j;
            }
        }
		if (clusterAssignment[i] != bestCluster) { // Check if the cluster assignment has changed
            *changed = true;
		}
        clusterAssignment[i] = bestCluster;
    }
}

// Function to update centroids
void updateCentroids(float** points, int* clusterAssignment, float** centroids, int n, int k, int dimensions) {

	float** newCentroids = (float**)malloc(k * sizeof(float*)); // Allocate memory for new centroids
	if (newCentroids == NULL) {
		perror("Failed to allocate memory for new centroids");
		exit(EXIT_FAILURE);
	}
    
	for (int i = 0; i < k; ++i) {
		newCentroids[i] = (float*)calloc(dimensions, sizeof(float));
		if (newCentroids[i] == NULL) {
			perror("Failed to allocate memory for new centroids");
			exit(EXIT_FAILURE);
		}
	}

	int* clusterSizes = (int*)calloc(k, sizeof(int)); // Allocate memory for cluster sizes
	if (clusterSizes == NULL) {
		perror("Failed to allocate memory for cluster sizes");
		exit(EXIT_FAILURE);
	}

    // Accumulate sums for each cluster
    for (int i = 0; i < n; ++i) {
        int cluster = clusterAssignment[i];
        for (int j = 0; j < dimensions; ++j) {
            newCentroids[cluster][j] += points[i][j];
        }
        clusterSizes[cluster]++;
    }

    // Compute the mean for each cluster
    for (int i = 0; i < k; ++i) {
        if (clusterSizes[i] > 0) {
            for (int j = 0; j < dimensions; ++j) {
                centroids[i][j] = newCentroids[i][j] / clusterSizes[i];
            }
        }
    }
	// Free memory
	for (int i = 0; i < k; ++i) {
		free(newCentroids[i]);
		newCentroids[i] = NULL;
	}
	free(newCentroids);
	newCentroids = NULL;
	free(clusterSizes);
	clusterSizes = NULL;
}

// K-Means algorithm
void kMeansCPU(float** points, int* clusterAssignment, int n, int k, int dimensions, int maxIterations, float* clusters) {
	// Allocate memory for centroids
    float** centroids = (float**)malloc(k * sizeof(float*));
	if (centroids == NULL) {
		perror("Failed to allocate memory for centroids");
		exit(EXIT_FAILURE);
	}
	for (int i = 0; i < k; ++i) {
		centroids[i] = (float*)calloc(dimensions, sizeof(float));
		if (centroids[i] == NULL) {
			perror("Failed to allocate memory for centroids");
			exit(EXIT_FAILURE);
		}
	}
	// Initialize centroids
    initializeCentroids(points, centroids, n, k, dimensions);

	bool* changed = (bool*)malloc(sizeof(bool)); // Allocate memory for flag
	if (changed == NULL) {
		perror("Failed to allocate memory for changed");
		exit(EXIT_FAILURE);
	}

	// Run the main loop
	for (int iter = 0; iter < maxIterations; ++iter) { 
        *changed = false;
        assignClusters(points, clusterAssignment, centroids, n, k, dimensions, changed);
        updateCentroids(points, clusterAssignment, centroids, n, k, dimensions);
        printf("Iteration %d done\n", iter + 1);
		if (!(*changed)) break; // If no point changed cluster, we can stop
    }

	// Flatten the centroids array
	for (int i = 0; i < k; i++) {
		for (int j = 0; j < dimensions; j++) {
			clusters[i * dimensions + j] = centroids[i][j];
		}
	}
	// Free memory
	for (int i = 0; i < k; ++i) {
		free(centroids[i]);
		centroids[i] = NULL;
	}
	free(centroids);
	free(changed);
}
