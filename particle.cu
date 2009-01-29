// Device Functions
__device__ __host__ int getCellId(Particle* p)
{
	int cell_x = (int) floor((*p).position.x / CELL_LENGTH);
	int cell_y = (int) floor((*p).position.y / CELL_LENGTH);
	int cell_z = (int) floor((*p).position.z / CELL_LENGTH);

	return ((int)(((int)cell_x*CELLS_IN_X)+cell_y)*CELLS_IN_Y) + cell_z;
}

__device__ void applyCollisionForceWithParticles(Particle* particleOnWhichForceIsApplied, Particle* particleThatIsBeingHit)
{
	float3 distBtwnPoints = distanceBetweenPoints((*particleThatIsBeingHit).position, (*particleOnWhichForceIsApplied).position);
	float distanceMagnitude = vectorMagnitude(distBtwnPoints);
	float collisionDistance = PARTICLE_RADIUS + PARTICLE_RADIUS;

	if (distanceMagnitude < 1.05*collisionDistance)
	{
		// Velocities
		float3 normal = distBtwnPoints / distanceMagnitude;
		float3 relativeVelocity;
		relativeVelocity.x = (*particleOnWhichForceIsApplied).velocity.x - (*particleThatIsBeingHit).velocity.x;
		relativeVelocity.y = (*particleOnWhichForceIsApplied).velocity.y - (*particleThatIsBeingHit).velocity.y;
		relativeVelocity.z = (*particleOnWhichForceIsApplied).velocity.z - (*particleThatIsBeingHit).velocity.z;
		float3 tangentalVelocity = relativeVelocity - (dot(relativeVelocity, normal) * normal);

		// Spring repulsive force
		float force = COLLISION_SPRING_CONSTANT * (1.05*collisionDistance - distanceMagnitude);
		(*particleOnWhichForceIsApplied).collisionForce.x += (distBtwnPoints.x / distanceMagnitude) * force;
		(*particleOnWhichForceIsApplied).collisionForce.y += (distBtwnPoints.y / distanceMagnitude) * force;
		(*particleOnWhichForceIsApplied).collisionForce.z += (distBtwnPoints.z / distanceMagnitude) * force;

		// Attraction
		(*particleOnWhichForceIsApplied).collisionForce += ATTRACTION_CONSTANT * distBtwnPoints;

		// Dashpot (damping) force
		(*particleOnWhichForceIsApplied).collisionForce += COLLISION_DAMPING_CONSTANT * relativeVelocity;

		// Tangential shear force
		(*particleOnWhichForceIsApplied).collisionForce += SHEARING_CONSTANT * tangentalVelocity;
	}
}

__device__ void computeInterparticleForces(int particleId, Particle* particleArray, Cell* cellArray)
{
	//Iterate over neighbor cells
	Cell mCell = cellArray[getCellId(&particleArray[particleId])];
	for(int i=-1; i<=1; i++)
	{
		for(int j=-1; j<=1; j++)
		{
			for(int k=-1; k<=1; k++)
			{
				int nCellx=mCell.coordinates.x+i;
				int nCelly=mCell.coordinates.y+j;
				int nCellz=mCell.coordinates.z+k;

				if(nCellx>=0 && nCellx<CELLS_IN_X && nCelly>=0 && nCelly<CELLS_IN_Y && nCellz>=0 && nCellz<CELLS_IN_Z)
				{
					Cell nCell = cellArray[((nCellx * CELLS_IN_X) + nCelly) * CELLS_IN_Y + nCellz];

					for(unsigned int particleIndex = 0; particleIndex < nCell.numberOfParticles; particleIndex++)
					{
						if(nCell.particleidxs[particleIndex] != NO_PARTICLE)
						{
							if(nCell.particleidxs[particleIndex] == particleId)
							{
								continue;
							}
							else
							{
								applyCollisionForceWithParticles(&particleArray[particleId], &particleArray[nCell.particleidxs[particleIndex]]);
							}
						}
					}
				}
			}

		}
	}
}

__device__ void computeForces(unsigned int id, Particle* particleArray, float deltaTime)
{
	float R_two =  (1 - GLOBAL_DAMPING_CONSTANT * (deltaTime/2));
	float R_one = (1 + GLOBAL_DAMPING_CONSTANT * (deltaTime/2));

	float3 gravitationalForce;
	gravitationalForce.x = 0.0;
	gravitationalForce.y = PARTICLE_MASS * GRAVITY_CONSTANT;
	gravitationalForce.z = 0.0;

	particleArray[id].velocity.x = particleArray[id].velocity.x * (R_two/R_one)   +   (deltaTime/(R_one*PARTICLE_MASS)) * (gravitationalForce.x + particleArray[id].collisionForce.x);
	particleArray[id].velocity.y = particleArray[id].velocity.y * (R_two/R_one)   +   (deltaTime/(R_one*PARTICLE_MASS)) * (gravitationalForce.y + particleArray[id].collisionForce.y);
	particleArray[id].velocity.z = particleArray[id].velocity.z * (R_two/R_one)   +   (deltaTime/(R_one*PARTICLE_MASS)) * (gravitationalForce.z + particleArray[id].collisionForce.z);

	particleArray[id].collisionForce.x = 0;
	particleArray[id].collisionForce.y = 0;
	particleArray[id].collisionForce.z = 0;
}

__device__ void updatePositions(unsigned int id, Particle* particleArray, Cell* cellArray, float deltaTime)
{
	// Empty the old values for the cell array
	//cellArray[getCellId(particleArray[id])].counter=0;

	 // Update particle position
	float x = particleArray[id].position.x + particleArray[id].velocity.x*deltaTime;
	float y = particleArray[id].position.y + particleArray[id].velocity.y*deltaTime;
	float z = particleArray[id].position.z + particleArray[id].velocity.z*deltaTime;

	// Boundary check
	if(x - PARTICLE_RADIUS < 0.0)
	{
		x = 2* PARTICLE_RADIUS - x;
		particleArray[id].velocity.x = -particleArray[id].velocity.x*BOUNDARY_DAMPING_CONSTANT;
	}
	else if(x + PARTICLE_RADIUS > BOUNDARY_X)
	{
		x = BOUNDARY_X - (x + 2*PARTICLE_RADIUS - BOUNDARY_X);
		particleArray[id].velocity.x = -particleArray[id].velocity.x*BOUNDARY_DAMPING_CONSTANT;
	}

	if(y - PARTICLE_RADIUS < 0.0)
	{
		y = 2* PARTICLE_RADIUS-y;
		particleArray[id].velocity.y = -particleArray[id].velocity.y*BOUNDARY_DAMPING_CONSTANT;
	}
	else if(y + PARTICLE_RADIUS > BOUNDARY_Y)
	{
		y = BOUNDARY_Y - (y+2*PARTICLE_RADIUS - BOUNDARY_Y);
		particleArray[id].velocity.y = -particleArray[id].velocity.y*BOUNDARY_DAMPING_CONSTANT;
	}

	if(z- PARTICLE_RADIUS < 0.0)
	{
		z = 2* PARTICLE_RADIUS-z;
		particleArray[id].velocity.z = -particleArray[id].velocity.z*BOUNDARY_DAMPING_CONSTANT;
	}
	else if(z+PARTICLE_RADIUS > BOUNDARY_Z)
	{
		z = BOUNDARY_Z - (z+2*PARTICLE_RADIUS - BOUNDARY_Z);
		particleArray[id].velocity.z = -particleArray[id].velocity.z*BOUNDARY_DAMPING_CONSTANT;
	}

	particleArray[id].position.x = x;
	particleArray[id].position.y = y;
	particleArray[id].position.z = z;

	#ifndef THREAD_PER_CELL_COLLISIONS
		//Update cell information
		cellArray[getCellId(&particleArray[id])].numberOfParticles=0;
		__syncthreads();
	#endif

}

__device__ void updateCells(int id, Particle* particleArray, Cell* cellArray)
{
	int cellidx = getCellId(&particleArray[id]);

	#if defined CUDA_NO_SM_11_ATOMIC_INTRINSICS
		int counter = 0;

		for(int i=0; i< NUMBER_OF_PARTICLES; i++)
		{
			if (cellidx == getCellId(&particleArray[i]))
			{
				if(i==id)
				{
					cellArray[cellidx].particleidxs[counter]=id;

					if(counter >= cellArray[cellidx].numberOfParticles && counter < NUMBER_OF_PARTICLES)
					{
						cellArray[cellidx].numberOfParticles=counter+1;
					}
				}
				counter=counter+1;
				if(counter >= MAX_PARTICLES_PER_CELL)
				{
					break;
				}
			}
		}

	#else
		int counter = atomicAdd(&cellArray[cellidx].numberOfParticles, 1);
		if(counter >= MAX_PARTICLES_PER_CELL)
		{
			cellArray[cellidx].numberOfParticles = MAX_PARTICLES_PER_CELL;
		}
		counter = min(counter, MAX_PARTICLES_PER_CELL-1);
		cellArray[cellidx].particleidxs[counter]=id;
	#endif
}


// Kernel Functions
#ifndef NO_DISPLAY
	#ifdef USE_VBO
		__global__ void particleKernel(float4* spheres, float3* colors, float3* normals, Particle* particleArray, Cell* cellArray, float deltaTime)
	#else
		__global__ void particleKernel(Particle* particleArray, Cell* cellArray, float deltaTime)
	#endif
#else
	__global__ void particleKernel(Particle* particleArray, Cell* cellArray, float deltaTime)
#endif
{
	// Get id for current particle
	unsigned int id = blockIdx.x* blockDim.x + threadIdx.x;

	if(id < NUMBER_OF_PARTICLES)
	{
		#ifndef THREAD_PER_CELL_COLLISIONS
			computeInterparticleForces(id, particleArray, cellArray);
		#endif

		computeForces(id, particleArray, deltaTime);

		updatePositions(id, particleArray, cellArray, deltaTime);

		updateCells(id, particleArray, cellArray);

		#ifndef NO_DISPLAY
			#ifdef USE_VBO
				makeSphere(spheres, colors, normals, id, particleArray[id].position.x, particleArray[id].position.y, particleArray[id].position.z, PARTICLE_RADIUS, particleArray[id].color);
			#endif
		#endif
	}
}

/*
__global__ void colorParticleKernel(float3* colors)
{
	// Get id for current particle
	unsigned int id = blockIdx.x* blockDim.x + threadIdx.x;
	unsigned int offset = id * SPHERE_VERTICES_SIZE;

	if(id < NUMBER_OF_PARTICLES)
	{
		for(unsigned int i = 0; i < SPHERE_VERTICES_SIZE; i++)
		{
			colors[offset + i].x = constantMemColorArray[id].x;
			colors[offset + i].y = constantMemColorArray[id].y;
			colors[offset + i].z = constantMemColorArray[id].z;
		}
	}
}
*/
