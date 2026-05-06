#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef USE_CUDA
int checkCuda(cudaError_t code, const char* what)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(code));
		return 0;
	}
	return 1;
}

/* One block per affected body; threads in a block share one tile of sources at a time. */
#define PAIRWISE_TILE 256

__global__ void pairwiseAccelKernel(const vector3* positions, const double* masses, vector3* pairwiseAccels, int entityCount)
{
	int affectedEntity = blockIdx.x;
	int sourceBase = blockIdx.y * PAIRWISE_TILE;
	int sourceEntity = sourceBase + threadIdx.x;

	if (affectedEntity >= entityCount)
		return;

	__shared__ double sAff[3];
	__shared__ double sSrcPos[PAIRWISE_TILE][3];
	__shared__ double sSrcMass[PAIRWISE_TILE];

	if (threadIdx.x == 0)
	{
		sAff[0] = positions[affectedEntity][0];
		sAff[1] = positions[affectedEntity][1];
		sAff[2] = positions[affectedEntity][2];
	}
	int loadIdx = sourceBase + threadIdx.x;
	if (threadIdx.x < PAIRWISE_TILE && loadIdx < entityCount)
	{
		sSrcPos[threadIdx.x][0] = positions[loadIdx][0];
		sSrcPos[threadIdx.x][1] = positions[loadIdx][1];
		sSrcPos[threadIdx.x][2] = positions[loadIdx][2];
		sSrcMass[threadIdx.x] = masses[loadIdx];
	}
	__syncthreads();

	if (sourceEntity >= entityCount)
		return;

	int pairIndex = affectedEntity * entityCount + sourceEntity;
	if (affectedEntity == sourceEntity)
	{
		pairwiseAccels[pairIndex][0] = 0.0;
		pairwiseAccels[pairIndex][1] = 0.0;
		pairwiseAccels[pairIndex][2] = 0.0;
		return;
	}

	int t = threadIdx.x;
	double deltaX = sAff[0] - sSrcPos[t][0];
	double deltaY = sAff[1] - sSrcPos[t][1];
	double deltaZ = sAff[2] - sSrcPos[t][2];
	double distanceSquared = deltaX * deltaX + deltaY * deltaY + deltaZ * deltaZ;
	double distance = sqrt(distanceSquared);
	double accelMagnitude = -1.0 * GRAV_CONSTANT * sSrcMass[t] / distanceSquared;

	pairwiseAccels[pairIndex][0] = accelMagnitude * deltaX / distance;
	pairwiseAccels[pairIndex][1] = accelMagnitude * deltaY / distance;
	pairwiseAccels[pairIndex][2] = accelMagnitude * deltaZ / distance;
}

#define REDUCE_THREADS 256

/* One block per row: parallel sum of columns (coalesced reads along each row). */
__global__ void sumAccelRowsKernel(const vector3* pairwiseAccels, vector3* rowSums, int entityCount)
{
	__shared__ double sx[REDUCE_THREADS];
	__shared__ double sy[REDUCE_THREADS];
	__shared__ double sz[REDUCE_THREADS];

	int row = blockIdx.x;
	unsigned int tid = threadIdx.x;

	if (row >= entityCount)
		return;

	if (tid < (unsigned int)entityCount)
	{
		int idx = row * entityCount + (int)tid;
		sx[tid] = pairwiseAccels[idx][0];
		sy[tid] = pairwiseAccels[idx][1];
		sz[tid] = pairwiseAccels[idx][2];
	}
	else
	{
		sx[tid] = 0.0;
		sy[tid] = 0.0;
		sz[tid] = 0.0;
	}
	__syncthreads();

	for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			sx[tid] += sx[tid + stride];
			sy[tid] += sy[tid + stride];
			sz[tid] += sz[tid + stride];
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		rowSums[row][0] = sx[0];
		rowSums[row][1] = sy[0];
		rowSums[row][2] = sz[0];
	}
}

__global__ void updateStateKernel(vector3* positions, vector3* velocities, const vector3* rowAccelSums, int entityCount)
{
	int entityIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (entityIndex >= entityCount)
		return;

	double accelX = rowAccelSums[entityIndex][0];
	double accelY = rowAccelSums[entityIndex][1];
	double accelZ = rowAccelSums[entityIndex][2];

	velocities[entityIndex][0] += accelX * INTERVAL;
	velocities[entityIndex][1] += accelY * INTERVAL;
	velocities[entityIndex][2] += accelZ * INTERVAL;

	positions[entityIndex][0] += velocities[entityIndex][0] * INTERVAL;
	positions[entityIndex][1] += velocities[entityIndex][1] * INTERVAL;
	positions[entityIndex][2] += velocities[entityIndex][2] * INTERVAL;
}
#endif

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){
#ifdef USE_CUDA
	vector3 *devicePositions = NULL, *deviceVelocities = NULL, *devicePairwiseAccels = NULL;
	vector3 *deviceRowAccelSums = NULL;
	double *deviceMasses = NULL;
	const int entityCount = NUMENTITIES;
	size_t stateBytes = sizeof(vector3) * entityCount;
	size_t pairwiseAccelBytes = sizeof(vector3) * entityCount * entityCount;
	size_t massBytes = sizeof(double) * entityCount;

	if (!checkCuda(cudaMalloc((void**)&devicePositions, stateBytes), "cudaMalloc devicePositions")) exit(EXIT_FAILURE);
	if (!checkCuda(cudaMalloc((void**)&deviceVelocities, stateBytes), "cudaMalloc deviceVelocities")) exit(EXIT_FAILURE);
	if (!checkCuda(cudaMalloc((void**)&deviceMasses, massBytes), "cudaMalloc deviceMasses")) exit(EXIT_FAILURE);
	if (!checkCuda(cudaMalloc((void**)&devicePairwiseAccels, pairwiseAccelBytes), "cudaMalloc devicePairwiseAccels")) exit(EXIT_FAILURE);
	if (!checkCuda(cudaMalloc((void**)&deviceRowAccelSums, stateBytes), "cudaMalloc deviceRowAccelSums")) exit(EXIT_FAILURE);

	if (!checkCuda(cudaMemcpy(devicePositions, hPos, stateBytes, cudaMemcpyHostToDevice), "copy hPos")) exit(EXIT_FAILURE);
	if (!checkCuda(cudaMemcpy(deviceVelocities, hVel, stateBytes, cudaMemcpyHostToDevice), "copy hVel")) exit(EXIT_FAILURE);
	if (!checkCuda(cudaMemcpy(deviceMasses, mass, massBytes, cudaMemcpyHostToDevice), "copy mass")) exit(EXIT_FAILURE);

	dim3 pairwiseThreads(PAIRWISE_TILE);
	dim3 pairwiseBlocks(entityCount, (entityCount + PAIRWISE_TILE - 1) / PAIRWISE_TILE);
	pairwiseAccelKernel<<<pairwiseBlocks, pairwiseThreads>>>(devicePositions, deviceMasses, devicePairwiseAccels, entityCount);
	if (!checkCuda(cudaGetLastError(), "pairwise kernel launch")) exit(EXIT_FAILURE);

	sumAccelRowsKernel<<<entityCount, REDUCE_THREADS>>>(devicePairwiseAccels, deviceRowAccelSums, entityCount);
	if (!checkCuda(cudaGetLastError(), "sum rows kernel launch")) exit(EXIT_FAILURE);

	int threadsPerBlockState = 256;
	int blocksForStateUpdate = (entityCount + threadsPerBlockState - 1) / threadsPerBlockState;
	updateStateKernel<<<blocksForStateUpdate, threadsPerBlockState>>>(devicePositions, deviceVelocities, deviceRowAccelSums, entityCount);
	if (!checkCuda(cudaGetLastError(), "update kernel launch")) exit(EXIT_FAILURE);
	if (!checkCuda(cudaDeviceSynchronize(), "kernel synchronize")) exit(EXIT_FAILURE);

	if (!checkCuda(cudaMemcpy(hPos, devicePositions, stateBytes, cudaMemcpyDeviceToHost), "copy hPos back")) exit(EXIT_FAILURE);
	if (!checkCuda(cudaMemcpy(hVel, deviceVelocities, stateBytes, cudaMemcpyDeviceToHost), "copy hVel back")) exit(EXIT_FAILURE);

	cudaFree(deviceRowAccelSums);
	cudaFree(devicePairwiseAccels);
	cudaFree(deviceMasses);
	cudaFree(deviceVelocities);
	cudaFree(devicePositions);
#else
	//make an acceleration matrix which is NUMENTITIES squared in size;
	int affectedEntity, sourceEntity, axis;
	vector3* pairwiseAccelMatrix=(vector3*)malloc(sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	vector3** pairwiseAccelRows=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);
	for (affectedEntity=0;affectedEntity<NUMENTITIES;affectedEntity++)
		pairwiseAccelRows[affectedEntity]=&pairwiseAccelMatrix[affectedEntity*NUMENTITIES];
	//first compute the pairwise accelerations. Effect is on the first argument.
	for (affectedEntity=0;affectedEntity<NUMENTITIES;affectedEntity++){
		for (sourceEntity=0;sourceEntity<NUMENTITIES;sourceEntity++){
			if (affectedEntity==sourceEntity) {
				FILL_VECTOR(pairwiseAccelRows[affectedEntity][sourceEntity],0,0,0);
			}
			else{
				vector3 distanceVector;
				for (axis=0;axis<3;axis++) distanceVector[axis]=hPos[affectedEntity][axis]-hPos[sourceEntity][axis];
				double distanceSquared=distanceVector[0]*distanceVector[0]+distanceVector[1]*distanceVector[1]+distanceVector[2]*distanceVector[2];
				double distance=sqrt(distanceSquared);
				double accelMagnitude=-1*GRAV_CONSTANT*mass[sourceEntity]/distanceSquared;
				FILL_VECTOR(pairwiseAccelRows[affectedEntity][sourceEntity],accelMagnitude*distanceVector[0]/distance,accelMagnitude*distanceVector[1]/distance,accelMagnitude*distanceVector[2]/distance);
			}
		}
	}
	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	for (affectedEntity=0;affectedEntity<NUMENTITIES;affectedEntity++){
		vector3 totalAccel={0,0,0};
		for (sourceEntity=0;sourceEntity<NUMENTITIES;sourceEntity++){
			for (axis=0;axis<3;axis++)
				totalAccel[axis]+=pairwiseAccelRows[affectedEntity][sourceEntity][axis];
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (axis=0;axis<3;axis++){
			hVel[affectedEntity][axis]+=totalAccel[axis]*INTERVAL;
			hPos[affectedEntity][axis]+=hVel[affectedEntity][axis]*INTERVAL;
		}
	}
	free(pairwiseAccelRows);
	free(pairwiseAccelMatrix);
#endif
}
