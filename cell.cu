// Shared Memory
__shared__ ParticlePositionAndCollisionForce sharedParticles[CELL_KERNEL_BLOCK_WIDTH][CELL_KERNEL_BLOCK_HEIGHT][CELL_KERNEL_BLOCK_DEPTH][MAX_PARTICLES_PER_CELL];
__shared__ unsigned int sharedNumberOfParticlesInCell[CELL_KERNEL_BLOCK_WIDTH][CELL_KERNEL_BLOCK_HEIGHT][CELL_KERNEL_BLOCK_DEPTH];

// Device Functions
__device__ void applyCollisionForce(ParticlePositionAndCollisionForce* particleOnWhichForceIsApplied, ParticlePositionAndCollisionForce* particleThatIsBeingHit)
{
	float3 distBtwnPoints = distanceBetweenPoints((*particleThatIsBeingHit).position, (*particleOnWhichForceIsApplied).position);
	float distanceMagnitude = vectorMagnitude(distBtwnPoints);
	float collisionDistance = PARTICLE_RADIUS + PARTICLE_RADIUS;

	if (distanceMagnitude < 1.05 * collisionDistance)
	{
		// Spring repulsive force
		float force = COLLISION_SPRING_CONSTANT * (1.05 * collisionDistance - distanceMagnitude);
		(*particleOnWhichForceIsApplied).collisionForce.x += (distBtwnPoints.x / distanceMagnitude) * force;
		(*particleOnWhichForceIsApplied).collisionForce.y += (distBtwnPoints.y / distanceMagnitude) * force;
		(*particleOnWhichForceIsApplied).collisionForce.z += (distBtwnPoints.z / distanceMagnitude) * force;

		// Attraction
		(*particleOnWhichForceIsApplied).collisionForce += ATTRACTION_CONSTANT * distBtwnPoints;

		// Unable to do shearing or damping because no velocity in shared memory
	}
}

__device__  void computeForcesOnParticle(int3 cellIdInBlock, ParticlePositionAndCollisionForce* p)
{
	assert(cellIdInBlock.x > 0);
	assert(cellIdInBlock.y > 0);
	assert(cellIdInBlock.z > 0);
	assert(cellIdInBlock.x < CELL_KERNEL_BLOCK_WIDTH - 1);
	assert(cellIdInBlock.y < CELL_KERNEL_BLOCK_HEIGHT - 1);
	assert(cellIdInBlock.z < CELL_KERNEL_BLOCK_DEPTH - 1);
	assert((*p).collisionForce.x == 0);
	assert((*p).collisionForce.y == 0);
	assert((*p).collisionForce.z == 0);

	//Iterate over neighbor cells
	for(unsigned int i = cellIdInBlock.x - 1; i <= cellIdInBlock.x + 1; i++)
	{
		for(unsigned int j = cellIdInBlock.y - 1; j <= cellIdInBlock.y + 1; j++)
		{
			for(unsigned int k = cellIdInBlock.z - 1; k <= cellIdInBlock.z + 1; k++)
			{
				for(unsigned int particleIndex = 0; particleIndex < sharedNumberOfParticlesInCell[i][j][k]; particleIndex++)
				{
					if(sharedParticles[i][j][k][particleIndex].id != NO_PARTICLE)
					{
						if(sharedParticles[i][j][k][particleIndex].id == (*p).id)
						{
							continue;
						}
						else
						{
							applyCollisionForce(p, &sharedParticles[i][j][k][particleIndex]);
						}
					}
				}
			}
		}
	}
}

// Kernel Function
__global__ void cellKernel(Particle* particleArray, Cell* cellArray, float deltaTime)
{
	// Assertions
	assert(blockDim.x == CELL_KERNEL_BLOCK_WIDTH);
	assert(blockDim.y == CELL_KERNEL_BLOCK_HEIGHT);
	assert(blockDim.z == CELL_KERNEL_BLOCK_DEPTH);

	// Get id for current cell within the block
	// Yeah, having 2D grids is annoying
	int3 cellId;
	cellId.x = blockIdx.x * blockDim.x + threadIdx.x;
	cellId.x = (cellId.x / CELL_KERNEL_BLOCK_WIDTH) * (CELL_KERNEL_BLOCK_WIDTH - 2) + (cellId.x % CELL_KERNEL_BLOCK_WIDTH) - 1;
	cellId.y = (blockIdx.y % ((CELL_KERNEL_BLOCK_HEIGHT + 2) * CELLS_IN_Y / CELL_KERNEL_BLOCK_HEIGHT)) * CELL_KERNEL_BLOCK_HEIGHT + threadIdx.y;
	cellId.y = (cellId.y / CELL_KERNEL_BLOCK_HEIGHT) * (CELL_KERNEL_BLOCK_HEIGHT - 2) + (cellId.y % CELL_KERNEL_BLOCK_HEIGHT) - 1;
	cellId.z = (blockIdx.y / ((CELL_KERNEL_BLOCK_HEIGHT + 2) * CELLS_IN_Y / CELL_KERNEL_BLOCK_HEIGHT)) * CELL_KERNEL_BLOCK_DEPTH + threadIdx.z;
	cellId.z = (cellId.z / CELL_KERNEL_BLOCK_DEPTH) * (CELL_KERNEL_BLOCK_DEPTH - 2) + (cellId.z % CELL_KERNEL_BLOCK_DEPTH) - 1;

	int3 cellIdInBlock;
	cellIdInBlock.x = threadIdx.x;
	cellIdInBlock.y = threadIdx.y;
	cellIdInBlock.z = threadIdx.z;

	// Copy particles to shared memory
	unsigned int numberOfParticlesInThisCell = 0;
	if(cellId.x >= 0 && cellId.y >= 0 && cellId.z >= 0 && cellId.x < CELLS_IN_X && cellId.y < CELLS_IN_Y && cellId.z < CELLS_IN_Z)
	{

		Cell thisCell = cellArray[((cellId.x * CELLS_IN_X) + cellId.y) * CELLS_IN_Y + cellId.z];

		numberOfParticlesInThisCell = thisCell.numberOfParticles;

		for(unsigned int i = 0; i < numberOfParticlesInThisCell; i++)
		{
			sharedParticles[cellIdInBlock.x][cellIdInBlock.y][cellIdInBlock.z][i].id = NO_PARTICLE;
			if(cellId.x >= 0 && cellId.y >= 0 && cellId.z >= 0 && cellId.x < CELLS_IN_X && cellId.y < CELLS_IN_Y && cellId.z < CELLS_IN_Z)
			{
				unsigned int particleId = thisCell.particleidxs[i];
				if(particleId != NO_PARTICLE)
				{
					sharedParticles[cellIdInBlock.x][cellIdInBlock.y][cellIdInBlock.z][i].id = particleId;
					sharedParticles[cellIdInBlock.x][cellIdInBlock.y][cellIdInBlock.z][i].position = particleArray[particleId].position;
					sharedParticles[cellIdInBlock.x][cellIdInBlock.y][cellIdInBlock.z][i].collisionForce.x = 0.0;
					sharedParticles[cellIdInBlock.x][cellIdInBlock.y][cellIdInBlock.z][i].collisionForce.y = 0.0;
					sharedParticles[cellIdInBlock.x][cellIdInBlock.y][cellIdInBlock.z][i].collisionForce.z = 0.0;
				}
			}
		}
	}

	sharedNumberOfParticlesInCell[cellIdInBlock.x][cellIdInBlock.y][cellIdInBlock.z] = numberOfParticlesInThisCell;

	syncthreads();

	// Compute forces
	if(cellIdInBlock.x > 0 && cellIdInBlock.y > 0 && cellIdInBlock.z > 0 && cellIdInBlock.x < (CELL_KERNEL_BLOCK_WIDTH - 1) && cellIdInBlock.y < (CELL_KERNEL_BLOCK_HEIGHT - 1) && cellIdInBlock.y < (CELL_KERNEL_BLOCK_DEPTH - 1))
	{
		for(unsigned int i = 0; i < numberOfParticlesInThisCell; i++)
		{
			if(sharedParticles[cellIdInBlock.x][cellIdInBlock.y][cellIdInBlock.z][i].id != NO_PARTICLE)
			{
				computeForcesOnParticle(cellIdInBlock, &sharedParticles[cellIdInBlock.x][cellIdInBlock.y][cellIdInBlock.z][i]);
			}
		}
	}

	// Copy particle collision force to global memory
	for(unsigned int i = 0; i < numberOfParticlesInThisCell; i++)
	{
		if(cellId.x >= 0 && cellId.y >= 0 && cellId.z >= 0 && cellId.x < CELLS_IN_X && cellId.y < CELLS_IN_Y && cellId.z < CELLS_IN_Z)
		{
			unsigned int particleId = sharedParticles[cellIdInBlock.x][cellIdInBlock.y][cellIdInBlock.z][i].id;
			if(particleId != NO_PARTICLE)
			{
				particleArray[particleId].collisionForce = sharedParticles[cellIdInBlock.x][cellIdInBlock.y][cellIdInBlock.z][i].collisionForce;
				particleArray[particleId].position = sharedParticles[cellIdInBlock.x][cellIdInBlock.y][cellIdInBlock.z][i].position;
			}
		}
	}

	cellArray[(((cellId.x*CELLS_IN_X)+cellId.y)*CELLS_IN_Y) + cellId.z].numberOfParticles = 0;
}
