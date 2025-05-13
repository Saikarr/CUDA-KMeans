# CUDA KMeans Implementation

This project implements the KMeans clustering algorithm using CUDA with two different GPU approaches (native CUDA kernels and Thrust library) along with a CPU reference implementation.
For 3-dimensional data, the project includes an OpenGL visualization of the clustered points.

## Features

- Three computation methods:
  - `gpu1`: Native CUDA kernel implementation
  - `gpu2`: Thrust library implementation
  - `cpu`: Sequential CPU implementation (reference)
- Support for both text (`txt`) and binary (`bin`) file formats
- Automatic OpenGL visualization for 3D datasets
- Color-coded clusters in visualization
- Efficient GPU memory management
- Configurable through command line arguments

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (version 11.0 or higher recommended)
- OpenGL and GLUT (or FreeGLUT) libraries
- C++ compiler with C++11 support

## Usage

./kmeans <data_format> <computation_method> <input_file> <output_file>
Arguments:
- <data_format>: Input file format - `txt` (text) or `bin` (binary - much faster when there are many points)
- <computation_method>:
  - `gpu1`: CUDA kernel implementation
  - `gpu2`: Thrust library implementation
  - `cpu`: CPU reference implementation
- <input_file>: Path to input data file
- <output_file>: Path to output file for cluster assignments

## Input File Format (text):

<num_points> <num_dimensions> <num_clusters> \
<point1_dim1> <point1_dim2> ... <point1_dimN> \
<point2_dim1> <point2_dim2> ... <point2_dimN> \
... \
<pointM_dim1> <pointM_dim2> ... <pointM_dimN> \
\
Binary format follows the same structure but in binary representation.

## Performance Comparison

The project includes implementations with different computation methods to allow for performance comparison between:
- Native CUDA kernels (gpu1)
- Thrust library (gpu2)
- CPU reference (cpu)

## Visualization
For 3-dimensional datasets, the program will automatically launch an OpenGL visualization window showing points colored by cluster assignment. The view can be rotated by holding LMB and moving your mouse
