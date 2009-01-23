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

	//Relative position normalized
	float3 norm = relPos / dist;

	// relative velocity
	float3 relVel;
	relVel.x = two.velocity.x - one.velocity.x;
	relVel.y = two.velocity.y - one.velocity.y;
	relVel.z = two.velocity.z - one.velocity.z;

	float3 tanVel = relVel - (dot(relVel, norm) * norm);

	float3 force = make_float3(0.0f);

	if (dist < 1.2* collideDist)
	{

		// relative tangential velocity
		float3 tanVel = relVel - (dot(relVel, norm) * norm);
		// spring force
		force = -params.spring*(collideDist - dist) * norm;
		// dashpot (damping) force
		force += params.collisionDamping*relVel;
		// tangential shear force
		force += params.shear*tanVel;
		// attraction
		force += params.attraction*relPos;
		//printf("COLLISION!!!!!!!");

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

	Cell mCell= cellArray[cellidx];

	//printf("particle id {%d},coor {%d}{%d}{%d} \n", id, mCell.coordinates.x, mCell.coordinates.y, mCell.coordinates.z);
	for(int x=-1; x<=1; x++)
	{
		for(int y=-1; y<=1; y++)
		{
			for(int z=-1; z<=1; z++)
			{
				int nCellx=mCell.coordinates.x+x;
				int nCelly=mCell.coordinates.y+y;
				int nCellz=mCell.coordinates.z+z;

				if(nCellx>=0 && nCellx<params.cellsPerDim && nCelly>=0 && nCelly<params.cellsPerDim && nCellz>=0 && nCellz<params.cellsPerDim)
				{
					int neighboridx= ((int)(((int)nCellx*params.cellsPerDim)+nCelly)*params.cellsPerDim) + nCellz;
					Cell nCell= cellArray[neighboridx];

					if(nCell.counter>0)
					{
						//printf(" Analyzed: particle id {%d}, neighbor coor {%d}{%d}{%d} \n", id, nCellx, nCelly, nCellz);
						force+=collisionsInCell(id,nCell, params, particleArray);
					}
				}
			}

		}
	}
	//printf(" \n");

	__syncthreads();

	//Modify velocity
	float R_two=  (1- params.globalDamping * (deltaTime/2));
	float R_one= (1+ params.globalDamping * (deltaTime/2));

	particleArray[id].velocity= particleArray[id].velocity * (R_two/R_one)  + (deltaTime/(R_one*params.mass)) * (params.gravity + force);
	__syncthreads();
}

__device__ void updatePosition(int id, float4* spheres,Parameters params, Particle* particleArray,Cell* cellArray,float deltaTime )
{

	cellArray[particleArray[id].cellidx].counter=0;


	 // Update particle position
	float x = particleArray[id].position.x + particleArray[id].velocity.x*deltaTime;
	float y = particleArray[id].position.y + particleArray[id].velocity.y*deltaTime;
	float z = particleArray[id].position.z + particleArray[id].velocity.z*deltaTime;

	// Boundary check
	if(x- params.particleRadious < 0.0)
	{
		x = 2* params.particleRadious - x;
		particleArray[id].velocity.x = -particleArray[id].velocity.x*params.boundaryDamping;
	}
	else if(x + params.particleRadious > params.boundary)
	{
		x = params.boundary - (x +2*params.particleRadious - params.boundary);
		particleArray[id].velocity.x = -particleArray[id].velocity.x*params.boundaryDamping;
	}

	if(y- params.particleRadious  < 0.0)
	{
		y = 2* params.particleRadious-y;
		particleArray[id].velocity.y = -particleArray[id].velocity.y*params.boundaryDamping;
	}
	else if(y + params.particleRadious> params.boundary)
	{
		y = params.boundary - (y+2*params.particleRadious - params.boundary);
		particleArray[id].velocity.y = -particleArray[id].velocity.y*params.boundaryDamping;
	}

	if(z- params.particleRadious < 0.0)
	{
		z = 2* params.particleRadious-z;
		particleArray[id].velocity.z = -particleArray[id].velocity.z*params.boundaryDamping;
	}
	else if(z+params.particleRadious > params.boundary)
	{
		z = params.boundary - (z+2*params.particleRadious - params.boundary);
		particleArray[id].velocity.z = -particleArray[id].velocity.z*params.boundaryDamping;
	}

	//makeSphere(spheres, id, x , y, z , params.particleRadious);

	particleArray[id].position.x = x;
	particleArray[id].position.y = y;
	particleArray[id].position.z = z;

	//Update cell information
	int cell_x= (int) floor(x/ params.cellSize);
	int cell_y= (int) floor(y/ params.cellSize);
	int cell_z= (int) floor(z/ params.cellSize);

	int cellidx= ((int)(((int)cell_x*params.cellsPerDim)+cell_y)*params.cellsPerDim) + cell_z;
	particleArray[id].cellidx= cellidx;
	cellArray[cellidx].counter=0;

}




__device__ void updateCells (int id,Parameters params, Particle* particleArray, Cell* cellArray)
{
	int cellidx=particleArray[id].cellidx;

	#if defined CUDA_NO_SM_11_ATOMIC_INTRINSICS
		int counter = 0;

		for(int i=0; i< params.maxParticles; i++)
		{
			if (cellidx== particleArray[i].cellidx)
			{
				if(i==id)
				{
					cellArray[cellidx].particleidxs[counter]=id;

					if(counter >= cellArray[cellidx].counter && counter< params.maxParticlesPerCell)
					{
						cellArray[cellidx].counter=counter+1;
					}
				}
				counter=counter+1;
				if(counter >= params.maxParticlesPerCell)
				{
					break;
				}
			}
		}

	#else
		int counter = atomicAdd(&cellArray[cellidx].counter, 1);
		counter = min(counter, params.maxParticlesPerCell-1);
		cellArray[cellidx].particleidxs[counter]=id;
	#endif



}


__global__ void runKernel(float4* spheres, Parameters params, Particle* particleArray, Cell* cellArray, float deltaTime)
{
	// Get id for current particle
   unsigned int id = blockIdx.x* blockDim.x + threadIdx.x;

   computeInteractions(id,params,particleArray, cellArray);
   __syncthreads();

   updatePosition(id,spheres, params, particleArray, cellArray,deltaTime);
   __syncthreads();

   updateCells (id, params, particleArray, cellArray);
   __syncthreads();

}



