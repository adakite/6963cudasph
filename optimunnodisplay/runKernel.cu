#include "cutil_math.h"

__device__ float3 evaluateCollision(float4 particleOnePosition, float3 particleOneVelocity, float4 particleTwoPosition, float3 particleTwoVelocity, Parameters params)
{

	// calculate relative position
	float3 relPos;
	relPos.x = particleTwoPosition.x - particleOnePosition.x;
	relPos.y = particleTwoPosition.y - particleOnePosition.y;
	relPos.z = particleTwoPosition.z - particleOnePosition.z;

	float dist = length(relPos);
	float collideDist = params.particleRadious + params.particleRadious;

	//Relative position normalized
	float3 norm = relPos / dist;

	// relative velocity
	float3 relVel;
	relVel.x = particleTwoVelocity.x - particleOneVelocity.x;
	relVel.y = particleTwoVelocity.y - particleOneVelocity.y;
	relVel.z = particleTwoVelocity.z - particleOneVelocity.z;

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

__device__ float3 collisionsInCell(int id, Cell nCell, Parameters params, float4* particlePosition, float3* particleVelocity)
{
	float3 force=make_float3(0.0f);

	float4 particleOnePosition=particlePosition[id];
	float3 particleOneVelocity= particleVelocity[id];

	//Iterate over particles in this cell

	for (int i=0; i<nCell.counter; i++)
	{
		//Get neighbor particle index
		int neighboridx= nCell.particleidxs[i];

		//If neighbor exist and not collide with itself
		if(neighboridx!= -1 && neighboridx!= id)
		{
			float4 particleTwoPosition= particlePosition[neighboridx];
			float3 particleTwoVelocity= particleVelocity[neighboridx];

			force+= evaluateCollision(particleOnePosition,particleOneVelocity, particleTwoPosition, particleTwoVelocity, params);
		}
	}

	return force;
}
__device__ void computeInteractions(int id,Parameters params, float4* particlePosition, float3* particleVelocity, Cell* cellArray)
{

	// Get cell index of this particle
	int cellidx=particlePosition[id].w;

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
						force+=collisionsInCell(id,nCell, params, particlePosition, particleVelocity);
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

	particleVelocity[id]= particleVelocity[id] * (R_two/R_one)  + (deltaTime/(R_one*params.mass)) * (params.gravity + force);
	__syncthreads();
}

__device__ void updatePosition(int id,Parameters params, float4* particlePosition, float3* particleVelocity,Cell* cellArray,float deltaTime )
{

	cellArray[(int)particlePosition[id].w].counter=0;


	 // Update particle position
	float x = particlePosition[id].x + particleVelocity[id].x*deltaTime;
	float y = particlePosition[id].y + particleVelocity[id].y*deltaTime;
	float z = particlePosition[id].z + particleVelocity[id].z*deltaTime;

	// Boundary check
	if(x- params.particleRadious < 0.0)
	{
		x = 2* params.particleRadious - x;
		particleVelocity[id].x = -particleVelocity[id].x*params.boundaryDamping;
	}
	else if(x + params.particleRadious > params.boundary)
	{
		x = params.boundary - (x +2*params.particleRadious - params.boundary);
		particleVelocity[id].x = -particleVelocity[id].x*params.boundaryDamping;
	}

	if(y- params.particleRadious  < 0.0)
	{
		y = 2* params.particleRadious-y;
		particleVelocity[id].y = -particleVelocity[id].y*params.boundaryDamping;
	}
	else if(y + params.particleRadious> params.boundary)
	{
		y = params.boundary - (y+2*params.particleRadious - params.boundary);
		particleVelocity[id].y = -particleVelocity[id].y*params.boundaryDamping;
	}

	if(z- params.particleRadious < 0.0)
	{
		z = 2* params.particleRadious-z;
		particleVelocity[id].z = -particleVelocity[id].z*params.boundaryDamping;
	}
	else if(z+params.particleRadious > params.boundary)
	{
		z = params.boundary - (z+2*params.particleRadious - params.boundary);
		particleVelocity[id].z = -particleVelocity[id].z*params.boundaryDamping;
	}


	particlePosition[id].x = x;
	particlePosition[id].y = y;
	particlePosition[id].z = z;

	//Update cell information
	int cell_x= (int) floor(x/ params.cellSize);
	int cell_y= (int) floor(y/ params.cellSize);
	int cell_z= (int) floor(z/ params.cellSize);

	int cellidx= ((int)(((int)cell_x*params.cellsPerDim)+cell_y)*params.cellsPerDim) + cell_z;
	particlePosition[id].w= cellidx;
	cellArray[cellidx].counter=0;

}




__device__ void updateCells (int id,Parameters params, float4* particlePosition, Cell* cellArray)
{


	#if defined CUDA_NO_SM_11_ATOMIC_INTRINSICS


	int cellidx=(int) particlePosition[id].w;
	int counter = 0;

	for(int i=0; i< params.maxParticles; i++)
	{
		if (cellidx== particlePosition[i].w)
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
		int cellidx=particlePosition[id].w;
		int counter = atomicAdd(&cellArray[cellidx].counter, 1);
		counter = min(counter, params.maxParticlesPerCell-1);
		cellArray[cellidx].particleidxs[counter]=id;
	#endif



}


__global__ void runKernel(Parameters params, float4* particlePosition, float3* particleVelocity, float3* particleColor, Cell* cellArray, float deltaTime)
{
	// Get id for current particle
   unsigned int id = blockIdx.x* blockDim.x + threadIdx.x;

   computeInteractions(id,params,particlePosition, particleVelocity, cellArray);
   __syncthreads();

   updatePosition(id,params, particlePosition, particleVelocity, cellArray,deltaTime);
   __syncthreads();

   updateCells (id, params, particlePosition, cellArray);
   __syncthreads();

}

