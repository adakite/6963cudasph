
__device__ void updatePosition(int id, float4* spheres,float boundary,float cell_size, Particle* particleArray,Cell* cellArray )
{
	 // Update particle position
	float x = particleArray[id].position.x + particleArray[id].velocity.x;
	float y = particleArray[id].position.y + particleArray[id].velocity.y;
	float z = particleArray[id].position.z + particleArray[id].velocity.z;

	// Boundary check
	if(x < 0.0)
	{
		x = -x;
		particleArray[id].velocity.x = -particleArray[id].velocity.x;
	}
	else if(x > boundary)
	{
		x = boundary - (x - boundary);
		particleArray[id].velocity.x = -particleArray[id].velocity.x;
	}

	if(y < 0.0)
	{
		y = -y;
		particleArray[id].velocity.y = -particleArray[id].velocity.y;
	}
	else if(y > boundary)
	{
		y = boundary - (y - boundary);
		particleArray[id].velocity.y = -particleArray[id].velocity.y;
	}

	if(z < 0.0)
	{
		z = -z;
		particleArray[id].velocity.z = -particleArray[id].velocity.z;
	}
	else if(z > boundary)
	{
		z = boundary - (z - boundary);
		particleArray[id].velocity.z = -particleArray[id].velocity.z;
	}

	makeSphere(spheres, id, x , y, z , 0.5f);

	particleArray[id].position.x = x;
	particleArray[id].position.y = y;
	particleArray[id].position.z = z;

	//Update cell information
	int cell_x= (int) floor(x/ cell_size);
	int cell_y= (int) floor(y/ cell_size);
	int cell_z= (int) floor(z/ cell_size);

	int cellidx= (cell_x*boundary+cell_y)*boundary + cell_z;
	particleArray[id].cellidx= cellidx;
	cellArray[cellidx].counter=0;
}



__device__ void computeInteractions(int id, Particle* particleArray,Cell* cellArray)
{


}

__device__ void updateCells (int id,int maxParticlesPerCell, Particle* particleArray, Cell* cellArray)
{
	int cellidx=particleArray[id].cellidx;

	#if defined CUDA_NO_SM_11_ATOMIC_INTRINSICS
		int counter = 0;
	#else
		int counter = atomicAdd(&cellArray[cellidx].counter, 1);
		counter = min(counter, 4-1);
	#endif

	cellArray[cellidx].particleidxs[counter]=id;

}


__global__ void runKernel(float4* spheres, int maxParticlesPerCell, float boundary,float cell_size, Particle* particleArray, Cell* cellArray)
{
	// Get id for current particle
   unsigned int id = blockIdx.x* blockDim.x + threadIdx.x;

   computeInteractions(id,particleArray, cellArray);
   __syncthreads();

   updatePosition(id,spheres, boundary, cell_size, particleArray, cellArray);
	__syncthreads();

   updateCells (id, maxParticlesPerCell, particleArray, cellArray);
   __syncthreads();
}



