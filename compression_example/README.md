# Compression Example

This directory contains example applications demonstrating a prototype implementation of the celerity compression API for point cloud data. The applications process a point cloud in two main steps:

1. **Tiling**: The input point cloud is sorted into tiles.
2. **Shape Factor Calculation**: The shape factors for each point in the cloud are computed.

## Structure

- [`point_cloud_element_wise_compression/`](./point_cloud_element_wise_compression/): Element-wise compression implementation.
- [`point_cloud_global_memory_compression/`](./point_cloud_global_memory_compression/): Global memory compression implementation.
- [`point_cloud_local_memory_compression/`](./point_cloud_local_memory_compression/): Local memory compression implementation.
- [`point_cloud_uncompressed/`](./point_cloud_uncompressed/): Reference implementation without compression.
- Utility headers for binary I/O, quantization, performance measurement, and floating-point precision.

Each compression strategy folder contains:
- `celerity_tiling_shape_factors.cpp`: Main application setting up and starting the Celerity runtime and its kernels.
- `celerity_tiling.hpp`: Kernel for the tiling step.
- `celerity_shape_factors.hpp`: Kernel for computing the shape factors.
