#include <stdio.h>
#include <stdlib.h>
#include <vector>

typedef struct {
	int num_points;  // Number of points (n)
	int dimensions;  // Number of dimensions (d)
	int clusters;    // Number of clusters (k)
	float* points;   // Flattened array of point coordinates
} ParsedData;

// Functions to parse the input file
ParsedData parse_input_file(const char* file_path);
ParsedData parse_binary_file(const char* file_path);

// Function to free allocated memory in ParsedData
void free_data(ParsedData* data);

// Function to visualize clusters
void VisualizeClusters(std::vector<float> pointsAndAssignments);

// Function to write clusters to a file
void saveClustersToFile(const char* filename,
	float* clusters,
	int* clusterAssignment,
	int k, int dimensions, int n);
