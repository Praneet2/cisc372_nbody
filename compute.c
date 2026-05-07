#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"

#define MATRIX_THREADS_X 16
#define MATRIX_THREADS_Y 16
#define UPDATE_THREADS 256

static vector3 *d_positions = NULL;
static vector3 *d_velocities = NULL;
static vector3 *d_accel_matrix = NULL;
static double *d_masses = NULL;

static void cuda_ok(cudaError_t err, const char *step_name)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "CUDA failed in %s: %s\n", step_name, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

__global__ void fill_accel_matrix_kernel(vector3 *positions,
					 double *masses,
					 vector3 *accel_matrix,
					 int num_entities)
{
	int source = blockIdx.x * blockDim.x + threadIdx.x;
	int target = blockIdx.y * blockDim.y + threadIdx.y;

	if (target >= num_entities || source >= num_entities)
	{
		return;
	}

	int idx = target * num_entities + source;

	if (target == source)
	{
		accel_matrix[idx][0] = 0.0;
		accel_matrix[idx][1] = 0.0;
		accel_matrix[idx][2] = 0.0;
		return;
	}

	double dx = positions[target][0] - positions[source][0];
	double dy = positions[target][1] - positions[source][1];
	double dz = positions[target][2] - positions[source][2];

	double dist_sq = dx * dx + dy * dy + dz * dz;

	if (dist_sq == 0.0)
	{
		accel_matrix[idx][0] = 0.0;
		accel_matrix[idx][1] = 0.0;
		accel_matrix[idx][2] = 0.0;
		return;
	}

	double dist = sqrt(dist_sq);
	double accel_size = -GRAV_CONSTANT * masses[source] / dist_sq;

	accel_matrix[idx][0] = accel_size * dx / dist;
	accel_matrix[idx][1] = accel_size * dy / dist;
	accel_matrix[idx][2] = accel_size * dz / dist;
}

__global__ void integrate_kernel(vector3 *positions,
				 vector3 *velocities,
				 vector3 *accel_matrix,
				 int num_entities)
{
	int entity = blockIdx.x * blockDim.x + threadIdx.x;

	if (entity >= num_entities)
	{
		return;
	}

	double sum_ax = 0.0;
	double sum_ay = 0.0;
	double sum_az = 0.0;

	int row_start = entity * num_entities;
	int other;
	for (other = 0; other < num_entities; other++)
	{
		int cell = row_start + other;
		sum_ax += accel_matrix[cell][0];
		sum_ay += accel_matrix[cell][1];
		sum_az += accel_matrix[cell][2];
	}

	velocities[entity][0] += sum_ax * INTERVAL;
	velocities[entity][1] += sum_ay * INTERVAL;
	velocities[entity][2] += sum_az * INTERVAL;

	positions[entity][0] += velocities[entity][0] * INTERVAL;
	positions[entity][1] += velocities[entity][1] * INTERVAL;
	positions[entity][2] += velocities[entity][2] * INTERVAL;
}

void initDeviceMemory(void)
{
	int n = NUMENTITIES;
	size_t vectors_bytes = sizeof(vector3) * n;
	size_t matrix_bytes = sizeof(vector3) * n * n;
	size_t masses_bytes = sizeof(double) * n;

	cuda_ok(cudaMalloc((void **)&d_positions, vectors_bytes), "cudaMalloc positions");
	cuda_ok(cudaMalloc((void **)&d_velocities, vectors_bytes), "cudaMalloc velocities");
	cuda_ok(cudaMalloc((void **)&d_accel_matrix, matrix_bytes), "cudaMalloc accel matrix");
	cuda_ok(cudaMalloc((void **)&d_masses, masses_bytes), "cudaMalloc masses");

	cuda_ok(cudaMemcpy(d_positions, hPos, vectors_bytes, cudaMemcpyHostToDevice),
		"upload positions");
	cuda_ok(cudaMemcpy(d_velocities, hVel, vectors_bytes, cudaMemcpyHostToDevice),
		"upload velocities");
	cuda_ok(cudaMemcpy(d_masses, mass, masses_bytes, cudaMemcpyHostToDevice),
		"upload masses");
}

void copyHostToDevice(void)
{
	int n = NUMENTITIES;
	size_t vectors_bytes = sizeof(vector3) * n;

	cuda_ok(cudaMemcpy(d_positions, hPos, vectors_bytes, cudaMemcpyHostToDevice),
		"upload positions");
	cuda_ok(cudaMemcpy(d_velocities, hVel, vectors_bytes, cudaMemcpyHostToDevice),
		"upload velocities");
}

void copyDeviceToHost(void)
{
	int n = NUMENTITIES;
	size_t vectors_bytes = sizeof(vector3) * n;

	cuda_ok(cudaMemcpy(hPos, d_positions, vectors_bytes, cudaMemcpyDeviceToHost),
		"download positions");
	cuda_ok(cudaMemcpy(hVel, d_velocities, vectors_bytes, cudaMemcpyDeviceToHost),
		"download velocities");
}

void freeDeviceMemory(void)
{
	cudaFree(d_masses);
	cudaFree(d_accel_matrix);
	cudaFree(d_velocities);
	cudaFree(d_positions);
	d_masses = NULL;
	d_accel_matrix = NULL;
	d_velocities = NULL;
	d_positions = NULL;
}

void compute(void)
{
	int n = NUMENTITIES;

	int blocks_x = (n + MATRIX_THREADS_X - 1) / MATRIX_THREADS_X;
	int blocks_y = (n + MATRIX_THREADS_Y - 1) / MATRIX_THREADS_Y;

	dim3 block_size(MATRIX_THREADS_X, MATRIX_THREADS_Y);
	dim3 grid_size(blocks_x, blocks_y);

	fill_accel_matrix_kernel<<<grid_size, block_size>>>(
	    d_positions,
	    d_masses,
	    d_accel_matrix,
	    n);
	cuda_ok(cudaGetLastError(), "launch fill_accel_matrix_kernel");

	int update_blocks = (n + UPDATE_THREADS - 1) / UPDATE_THREADS;

	integrate_kernel<<<update_blocks, UPDATE_THREADS>>>(
	    d_positions,
	    d_velocities,
	    d_accel_matrix,
	    n);
	cuda_ok(cudaGetLastError(), "launch integrate_kernel");

	cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}
