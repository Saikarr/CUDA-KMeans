#include "headers.h"

// Function to parse the input .txt file
ParsedData parse_input_file(const char* file_path) {
	ParsedData data;
	FILE* file = fopen(file_path, "r");

	if (file == NULL) {
		perror("Failed to open the file");
		exit(EXIT_FAILURE);
	}

	// Read the first line (n, d, k)
	if (fscanf(file, "%d %d %d", &data.num_points, &data.dimensions, &data.clusters) != 3) {
		fprintf(stderr, "Error: Failed to read header line\n");
		fclose(file);
		exit(EXIT_FAILURE);
	}
	// Allocate memory for points (n * d floats)
	data.points = (float*)malloc(data.num_points * data.dimensions * sizeof(float));
	if (data.points == NULL) {
		perror("Failed to allocate memory for points");
		fclose(file);
		exit(EXIT_FAILURE);
	}

	// Read the points
	for (int i = 0; i < data.num_points; ++i) {
		for (int d = 0; d < data.dimensions; ++d) {
			if (fscanf(file, "%f", &data.points[i * data.dimensions + d]) != 1) {
				fprintf(stderr, "Error: Failed to read point data at point %d, dimension %d\n", i, d);
				free(data.points);
				fclose(file);
				exit(EXIT_FAILURE);
			}
		}
	}

	fclose(file);
	return data;
}

// Function to parse the input .bin file
ParsedData parse_binary_file(const char* file_path) {
	ParsedData data;
	FILE* file = fopen(file_path, "rb");

	if (file == NULL) {
		perror("Failed to open the file");
		exit(EXIT_FAILURE);
	}

	// Read the header (n, d, k)
	if (fread(&data.num_points, sizeof(int), 1, file) != 1 ||
		fread(&data.dimensions, sizeof(int), 1, file) != 1 ||
		fread(&data.clusters, sizeof(int), 1, file) != 1) {
		fprintf(stderr, "Error: Failed to read the header\n");
		fclose(file);
		exit(EXIT_FAILURE);
	}

	// Allocate memory for points (n * d floats)
	data.points = (float*)malloc(data.num_points * data.dimensions * sizeof(float));
	if (data.points == NULL) {
		perror("Failed to allocate memory for points");
		fclose(file);
		exit(EXIT_FAILURE);
	}

	// Read the points (n * d floats)
	size_t total_floats = data.num_points * data.dimensions;
	if (fread(data.points, sizeof(float), total_floats, file) != total_floats) {
		fprintf(stderr, "Error: Failed to read points data\n");
		free(data.points);
		fclose(file);
		exit(EXIT_FAILURE);
	}

	fclose(file);

	return data;
}

// Function to free allocated memory in ParsedData
void free_data(ParsedData* data) {

	free(data->points);
	data->points = NULL;
}

// Function to save cluster assignments and points to a file
void saveClustersToFile(const char* filename, float* clusters, int* clusterAssignment, int k, int dimensions, int n) {
	FILE* file = fopen(filename, "w");
	if (file == NULL) {
		perror("Error: Could not open file for writing!\n");
		exit(EXIT_FAILURE);
	}

	// Save clusters (k rows, dimensions columns)
	for (int i = 0; i < k; ++i) {
		for (int j = 0; j < dimensions; ++j) {
			fprintf(file, "%.4f", clusters[i * dimensions + j]); // Print with 4 decimal places
			if (j < dimensions - 1) {
				fprintf(file, "    ");
			}
		}
		fprintf(file, "\n");
	}

	// Save cluster assignments (one per line)
	for (int i = 0; i < n; ++i) {
		fprintf(file, "  %d\n", clusterAssignment[i]);
	}

	fclose(file);
	printf("Cluster assignments and points saved to %s\n", filename);
}