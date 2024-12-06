#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <windows.h>

#include "kernels.cuh"
#include "headers.h"

void usage(const char* prog) {
	printf("Usage: %s <data_format> <computation_method> <input_file> <output_file>\n<data_format> - txt or bin\n"
		"<computation_method> - gpu1, gpu2 or cpu\n<input_file>, <output_file> - paths to input file and output file", prog);
}

int main(int argc, char* argv[])
{
	// Check the arguments
	if (argc != 5) usage(argv[0]);
	const char* data_format = argv[1];
	const char* computation_method = argv[2];
	const char* input_file = argv[3];
	const char* output_file = argv[4];

	if (strcmp(data_format, "txt") != 0 && strcmp(data_format, "bin") != 0) {
		usage(argv[0]);
		exit(EXIT_FAILURE);
	}

	if (strcmp(computation_method, "gpu1") != 0 && strcmp(computation_method, "gpu2") != 0 && strcmp(computation_method, "cpu") != 0) {
		usage(argv[0]);
		exit(EXIT_FAILURE);
	}

	// Measure time
	LARGE_INTEGER frequency;  // Ticks per second
	LARGE_INTEGER start, end;
	double elapsed_time;
	// Get the frequency of the performance counter
	QueryPerformanceFrequency(&frequency);

	ParsedData data = { 0 };
	printf("Reading data\n");

	QueryPerformanceCounter(&start);

	// Read the data
	if (strcmp(data_format, "txt") == 0) {
		data = parse_input_file(input_file);
	}
	else if (strcmp(data_format, "bin") == 0) {
		data = parse_binary_file(input_file);
	}

	QueryPerformanceCounter(&end);
	elapsed_time = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;
	printf("reading data time: %f miliseconds\n", elapsed_time*1000);

	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 1;
	}

	int n = data.num_points; // Number of points
	int k = data.clusters;     // Number of clusters
	int dims = data.dimensions;  // Dimensions of data
	int max_iter = 100;
	float* points = data.points;

	// Convert the data to 2D rotated array, needed for the thrust implementation
	float** points2Drotated = (float**)malloc(dims * sizeof(float*));
	if (points2Drotated == NULL) {
		perror("Failed to allocate memory for points");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < dims; i++) {
		points2Drotated[i] = (float*)malloc(n * sizeof(float));
		if (points2Drotated[i] == NULL) {
			perror("Failed to allocate memory for points");
			exit(EXIT_FAILURE);
		}
	}

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < dims; j++) {
			points2Drotated[j][i] = points[i * dims + j];
		}
	}

	// Convert the data to 2D array, needed for the CPU implementation
	float** points2D = (float**)malloc(n * sizeof(float*));
	if (points2D == NULL) {
		perror("Failed to allocate memory for points");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < n; i++) {
		points2D[i] = (float*)malloc(dims * sizeof(float));
		if (points2D[i] == NULL) {
			perror("Failed to allocate memory for points");
			exit(EXIT_FAILURE);
		}
	}

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < dims; j++) {
			points2D[i][j] = points[i * dims + j];
		}
	}

	// Allocate memory for labels and clusters
	int* labels = (int*)malloc(n * sizeof(int));
	if (labels == NULL) {
		perror("Failed to allocate memory for labels");
		exit(EXIT_FAILURE);
	}
	float* clusters = (float*)malloc(k * dims * sizeof(float));
	if (clusters == NULL) {
		perror("Failed to allocate memory for clusters");
		exit(EXIT_FAILURE);
	}

	// Output the parsed data
	printf("Parsed Data:\n");
	printf("Data Format: %s\n", data_format);
	printf("Number of Points: %d\n", data.num_points);
	printf("Number of Dimensions: %d\n", data.dimensions);
	printf("Number of Clusters: %d\n", data.clusters);
	
	// Perform the k-means clustering
	QueryPerformanceCounter(&start);
	printf("Calculation start\n");

	if (strcmp(computation_method, "gpu1") == 0) {
		printf("Computational Method: cuda kernels\n");
		cudaError_t cudaStatus = kmeans(points, labels, n, k, dims, max_iter, clusters);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "kmeans failed!");
			return 1;
		}
	}
	else if (strcmp(computation_method, "gpu2") == 0) {
		printf("Computational Method: thrust library\n");
		cudaError_t cudaStatus = kmeans_thrust(points, labels, n, k, dims, max_iter, points2Drotated, clusters);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "kmeans failed!");
			return 1;
		}
	}
	else if (strcmp(computation_method, "cpu") == 0) {
		printf("Computational Method: cpu\n");
		kMeansCPU(points2D, labels, n, k, dims, max_iter, clusters);
	}

	QueryPerformanceCounter(&end);
	elapsed_time = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;
	printf("calculations time: %f miliseconds\n", elapsed_time * 1000);


	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	// Visualize the clusters
	if (dims == 3) {
		std::vector<float> pointsAndAssignments;
		for (int i = 0; i < n; i++) {
			if (i % 100 == 0) {
				pointsAndAssignments.push_back(points[i * dims]);
				pointsAndAssignments.push_back(points[i * dims + 1]);
				pointsAndAssignments.push_back(points[i * dims + 2]);
				pointsAndAssignments.push_back((float)labels[i]);
			}
		}
		VisualizeClusters(pointsAndAssignments);
	}

	// Save the results
	QueryPerformanceCounter(&start);
	printf("Saving results\n");
	saveClustersToFile(output_file, clusters, labels, k, dims, n);
	QueryPerformanceCounter(&end);
	elapsed_time = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;
	printf("saving data time: %f miliseconds\n", elapsed_time * 1000);

	// Free the memory
	free_data(&data);
	free(labels);
	for (int i = 0; i < n; i++) {
		free(points2D[i]);
	}
	free(points2D);
	for (int i = 0; i < dims; i++) {
		free(points2Drotated[i]);
	}
	free(points2Drotated);
	free(clusters);

	return 0;
}
