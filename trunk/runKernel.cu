#include "cutil_math.h"

__device__ float3 evaluateCollision(Particle one,Particle two, Parameters params)
{

	// calculate relative position
	float3 relPos;
	relPos.x = two.position.x - one.position.x;
	relPos.y = two.position.y - one.position.y;
	relPos.z = two.position.z - one.position.z;
	float dist = length(relPos);

	float collideDist = params.particleRadious + params.particleRadious;

	float3 force = make_float3(0.0f);

	if (dist < collideDist)
	{
		//Relative position normalized
		float3 norm = relPos / dist;

		// relative velocity
		float3 relVel;
		relVel.x = two.velocity.x - one.velocity.x;
		relVel.y = two.velocity.y - one.velocity.y;
		relVel.z = two.velocity.z - one.velocity.z;

		// relative tangential velocity
		float3 tanVel = relVel - (dot(relVel, norm) * norm);
		// spring force
		force = -params.spring*(collideDist - dist) * norm;
		// dashpot (damping) force
		force += params.damping*relVel;
		// tangential shear force
		force += params.shear*tanVel;
		// attraction
		force += params.attraction*relPos;
	}

	return force;
}

__device__ float3 collisionsInCell(int id, Cell nCell, Parameters params, Particle* particleArray)
{
	float3 force=make_float3(0.0f);

	Particle one= particleArray[id];

	//Iterate over particles in this cell

	for (int i=0; i<nCell.counter; i++)
	{
		//Get neighbor particle index
		int neighboridx= nCell.particleidxs[i];

		//If neighbor exist and not collide with itself
		if(neighboridx!= -1 && neighboridx!= id)
		{
			Particle two= particleArray[neighboridx];
			force+= evaluateCollision(one, two, params);
		}
	}

	return force;
}
__device__ void computeInteractions(int id,Parameters params, Particle* particleArray, Cell* cellArray)
{

	// Get cell index of this particle
	int cellidx=particleArray[id].cellidx;

	//Iterate over neighbor cells
	float3 force = make_float3(0.0f);

	for(int x=-1; x<=1; x++)
	{
		for(int y=-1; y<=1; y++)
		{
			for(int z=-1; z<=1; z++)
			{
				Cell mCell= cellArray[cellidx];
				int nCellx=mCell.coordinates.x+x;
				int nCelly=mCell.coordinates.y+y;
				int nCellz=mCell.coordinates.z+z;

				if(nCellx>=0 && nCellx<params.boundary && nCelly>=0 && nCelly<params.boundary && nCellz>=0 && nCellz<params.boundary)
				{
					int neighboridx= (nCellx*params.boundary+nCelly)*params.boundary + nCellz;
					Cell nCell= cellArray[neighboridx];

					if(nCell.counter>0)
					{
						force+=collisionsInCell(id,nCell, params, particleArray);
					}
				}
			}

		}
	}


	__syncthreads();

	//Modify velocity
	particleArray[id].velocity= particleArray[id].velocity +force;

}

__device__ void updatePosition(int id, float4* spheres,Parameters params, Particle* particleArray,Cell* cellArray )
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
	else if(x > params.boundary)
	{
		x = params.boundary - (x - params.boundary);
		particleArray[id].velocity.x = -particleArray[id].velocity.x;
	}

	if(y < 0.0)
	{
		y = -y;
		particleArray[id].velocity.y = -particleArray[id].velocity.y;
	}
	else if(y > params.boundary)
	{
		y = params.boundary - (y - params.boundary);
		particleArray[id].velocity.y = -particleArray[id].velocity.y;
	}

	if(z < 0.0)
	{
		z = -z;
		particleArray[id].velocity.z = -particleArray[id].velocity.z;
	}
	else if(z > params.boundary)
	{
		z = params.boundary - (z - params.boundary);
		particleArray[id].velocity.z = -particleArray[id].velocity.z;
	}

	makeSphere(spheres, id, x , y, z , params.particleRadious);

	particleArray[id].position.x = x;
	particleArray[id].position.y = y;
	particleArray[id].position.z = z;

	//Update cell information
	int cell_x= (int) floor(x/ params.cellSize);
	int cell_y= (int) floor(y/ params.cellSize);
	int cell_z= (int) floor(z/ params.cellSize);

	int cellidx= (cell_x*params.boundary+cell_y)*params.boundary + cell_z;
	particleArray[id].cellidx= cellidx;
	cellArray[cellidx].counter=0;
}




__device__ void updateCells (int id,Parameters params, Particle* particleArray, Cell* cellArray)
{
	int cellidx=particleArray[id].cellidx;

	#if defined CUDA_NO_SM_11_ATOMIC_INTRINSICS
		int counter = 0;
	#else
		int counter = atomicAdd(&cellArray[cellidx].counter, 1);
		counter = min(counter, params.maxParticlesPerCell-1);
	#endif

	cellArray[cellidx].particleidxs[counter]=id;

}


__global__ void runKernel(float4* spheres, Parameters params, Particle* particleArray, Cell* cellArray)
{
	// Get id for current particle
   unsigned int id = blockIdx.x* blockDim.x + threadIdx.x;

   computeInteractions(id,params,particleArray, cellArray);
   __syncthreads();

   updatePosition(id,spheres, params, particleArray, cellArray);
	__syncthreads();

   updateCells (id, params, particleArray, cellArray);
   __syncthreads();
}



