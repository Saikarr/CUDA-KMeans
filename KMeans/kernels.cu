#include "kernels.cuh"

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/constant_iterator.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"



#define BLOCK_SIZE 1024

// Kernel to assign points to the nearest cluster
__global__ void assign_clusters(const float* data, const float* centroids, int* labels, int n, int k, int dims, bool* changed) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < n) {
		float min_dist = FLT_MAX;
		int min_idx = -1;

		for (int c = 0; c < k; c++) { // Iterate over all clusters
			float dist = 0.0f;
			for (int d = 0; d < dims; d++) { // Calculate Euclidean distance
				float diff = data[idx * dims + d] - centroids[c * dims + d];
				dist += diff * diff;
			}
			if (dist < min_dist) { // Update the nearest cluster
				min_dist = dist;
				min_idx = c;
			}
		}
		if (labels[idx] != min_idx) { // Check if the cluster assignment has changed
			*changed = true;
		}
		labels[idx] = min_idx;
	}
}

// Kernel to update centroids based on assigned points
__global__ void update_centroids(const float* data, float* centroids, const int* labels, int n, int k, int dims) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < k * dims) {
		int cluster = idx / dims;
		int dim = idx % dims;

		float sum = 0.0f;
		int count = 0;
		
		for (int i = 0; i < n; i++) { // Iterate over all points
			if (labels[i] == cluster) {
				sum += data[i * dims + dim];
				count++;
			}
		}
		float new_centroid = sum / count; // Calculate the new centroid

		centroids[idx] = new_centroid; // Update the centroid
	}
}

// Helper function to initialize centroids
void initialize_centroids(float* centroids, const float* data, int n, int k, int dims) {
	for (int i = 0; i < k; i++) {
		for (int d = 0; d < dims; d++) {
			centroids[i * dims + d] = data[i * dims + d];
		}
	}
}

// K-Means function using Thrust
cudaError_t kmeans_thrust(const float* h_data, int* h_labels, int n, int k, int dims, int max_iter, float** points2Drotated, float* h_centroids) {
	
	cudaError_t cudaStatus;
	float* d_data, * d_centroids;
	int* d_labels;
	size_t data_size = n * dims * sizeof(float);
	size_t centroids_size = k * dims * sizeof(float);
	size_t labels_size = n * sizeof(int);
	bool* d_changed;
	bool* h_changed = (bool*)malloc(sizeof(bool));
	if (h_changed == NULL) {
		fprintf(stderr, "Failed to allocate memory for h_changed\n");
		return cudaErrorMemoryAllocation;
	}

	// Allocate memory on the device
	cudaStatus = cudaMalloc(&d_changed, sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc(&d_data, data_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc(&d_centroids, centroids_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc(&d_labels, labels_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy data to the device
	cudaStatus = cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	
	// Initialize centroids
	initialize_centroids(h_centroids, h_data, n, k, dims);

	cudaStatus = cudaMemcpy(d_centroids, h_centroids, centroids_size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Calculate grid size for assign_clusters
	int grid_size_points = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

	// Run the K-Means algorithm
	for (int iter = 0; iter < 100; iter++) {

		*h_changed = false; // Initialize changed flag
		cudaStatus = cudaMemcpy(d_changed, h_changed, sizeof(bool), cudaMemcpyHostToDevice);

		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		// Assign points to the nearest cluster
		assign_clusters << <grid_size_points, BLOCK_SIZE >> > (d_data, d_centroids, d_labels, n, k, dims, d_changed);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "assign_clusters launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		cudaStatus = cudaDeviceSynchronize(); // Wait for the kernel to finish
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize failed!");
			goto Error;
		}

		// Copy the changed flag back to the host
		cudaStatus = cudaMemcpy(h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		// Update centroids
		for (int i = 0; i < dims; i++) {
			// Copy labels and points to device vectors
			thrust::device_vector<int> dev_labels(n);
			thrust::copy(d_labels, d_labels + n, dev_labels.begin());

			thrust::device_vector<float> dev_points(n);
			thrust::copy(points2Drotated[i], points2Drotated[i] + n, dev_points.begin());

			// Sort labels and points by key
			thrust::sort_by_key(dev_labels.begin(), dev_labels.end(), dev_points.begin());

			thrust::device_vector<float> dev_centroids(k);
			thrust::device_vector<int> dev_cluster_sizes(k);
			thrust::fill(dev_centroids.begin(), dev_centroids.end(), 0);
			thrust::fill(dev_cluster_sizes.begin(), dev_cluster_sizes.end(), 0);
			// Reduce by key to calculate sum of points and cluster sizes
			thrust::reduce_by_key(dev_labels.begin(), dev_labels.end(), dev_points.begin(), thrust::discard_iterator<>(),
				dev_centroids.begin(), thrust::equal_to<int>(), thrust::plus<float>());
			thrust::reduce_by_key(dev_labels.begin(), dev_labels.end(), thrust::constant_iterator<int>(1), thrust::discard_iterator<>(),
				dev_cluster_sizes.begin(), thrust::equal_to<int>(), thrust::plus<int>());

			// Divide sum of points by cluster sizes to get new centroids
			thrust::transform(dev_centroids.begin(), dev_centroids.end(), dev_cluster_sizes.begin(), dev_centroids.begin(), thrust::divides<float>());
			for (int j = 0; j < k; j++) {
				thrust::copy(dev_centroids.begin() + j, dev_centroids.begin() + j + 1, &d_centroids[j * dims + i]);
			}
		}

		printf("Iteration %d done\n", iter + 1);
		// Check if the cluster assignments have changed
		if (!(*h_changed)) {
			printf("Converged\n");
			break;
		}
	}

	// Copy labels and centroids back to the host
	cudaStatus = cudaMemcpy(h_labels, d_labels, labels_size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(h_centroids, d_centroids, centroids_size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	// Free resources
	cudaFree(d_data);
	cudaFree(d_centroids);
	cudaFree(d_labels);
	cudaFree(d_changed);

	free(h_changed);

	return cudaStatus;
}

// K-Means function using CUDA kernels
cudaError_t kmeans(const float* h_data, int* h_labels, int n, int k, int dims, int max_iter, float* h_centroids) {
	cudaError_t cudaStatus;
	float* d_data, * d_centroids;
	int* d_labels;
	size_t data_size = n * dims * sizeof(float);
	size_t centroids_size = k * dims * sizeof(float);
	size_t labels_size = n * sizeof(int);
	bool* d_changed;
	bool* h_changed = (bool*)malloc(sizeof(bool));
	if (h_changed == NULL) {
		fprintf(stderr, "Failed to allocate memory for h_changed\n");
		return cudaErrorMemoryAllocation;
	}

	// Allocate memory on the device
	cudaStatus = cudaMalloc(&d_changed, sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc(&d_data, data_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc(&d_centroids, centroids_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc(&d_labels, labels_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy data to the device
	cudaStatus = cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Initialize centroids
	initialize_centroids(h_centroids, h_data, n, k, dims);

	cudaStatus = cudaMemcpy(d_centroids, h_centroids, centroids_size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Calculate grid size for assign_clusters and update_centroids
	int grid_size_points = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	int grid_size_centroids = (k * dims + BLOCK_SIZE - 1) / BLOCK_SIZE;

	// Run the K-Means algorithm
	for (int iter = 0; iter < 100; iter++) {

		*h_changed = false; // Initialize changed flag
		cudaStatus = cudaMemcpy(d_changed, h_changed, sizeof(bool), cudaMemcpyHostToDevice); // Copy the changed flag to the device
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		// Assign points to the nearest cluster
		assign_clusters << <grid_size_points, BLOCK_SIZE >> > (d_data, d_centroids, d_labels, n, k, dims, d_changed);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "assign_clusters launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		cudaStatus = cudaDeviceSynchronize(); // Wait for the kernel to finish
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize failed!");
			goto Error;
		}

		cudaStatus = cudaMemcpy(h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost); // Copy the changed flag back to the host
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		cudaStatus = cudaMemset(d_centroids, 0, centroids_size); // Reset centroids
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemset failed!");
			goto Error;
		}

		// Update centroids
		update_centroids << <grid_size_centroids, BLOCK_SIZE >> > (d_data, d_centroids, d_labels, n, k, dims);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "update_centroids launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		cudaStatus = cudaDeviceSynchronize(); // Wait for the kernel to finish
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize failed!");
			goto Error;
		}

		printf("Iteration %d done\n", iter + 1);
		// Check if the cluster assignments have changed
		if (!(*h_changed)) {
			printf("Converged\n");
			break;
		}
	}

	// Copy labels and centroids back to the host
	cudaStatus = cudaMemcpy(h_labels, d_labels, labels_size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(h_centroids, d_centroids, centroids_size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	// Free resources
	cudaFree(d_data);
	cudaFree(d_centroids);
	cudaFree(d_labels);
	cudaFree(d_changed);

	free(h_changed);

	return cudaStatus;
}