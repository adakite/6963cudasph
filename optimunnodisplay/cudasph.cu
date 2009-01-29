
#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, GL
#include <GL/glew.h>

#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

// includes
#include <cutil.h>
#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>
#include <particle.h>



////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width = 1024;
const unsigned int window_height = 1024;

const float particle_size = 0.5f;
const float cell_size = 2.0f* particle_size;

const float maxVelocity = 10.0f;
const float minVelocity = -10.0f;
const int boundary= 32.0f;

const unsigned int numberOfParticles = 10240;
const unsigned int numberOfParticlesPerBlock = 32;
const unsigned int numberOfCellsPerDim=((int)floor((boundary)/cell_size));
const unsigned int numberOfCells= numberOfCellsPerDim*numberOfCellsPerDim*numberOfCellsPerDim;
const unsigned int maxParticlesPerCell=4;
const float deltaTime=0.05f;

unsigned int timer;
unsigned int iterations;

const float mass=1.0f;
const float spring=1.0f;
const float globalDamping=1.0f;
const float shear=0.1f;
const float attraction= 0.01f;
const float gravityValue= -10.0f;
const float boundaryDamping=0.7f;
const float collisionDamping=0.01f;


/////////////////////////////////////////////////////////////////////////////////
//Physics variables
float anim = 0.0;
Parameters params;

//Particle position, velocity, color
//3D position + cellid
float4* particlePosition_h= (float4*) malloc (numberOfParticles*sizeof(float4));
float4* particlePosition_d;
float3* particleVelocity_h= (float3*) malloc (numberOfParticles*sizeof(float3));
float3* particleVelocity_d;
float3* particleColor_h= (float3*) malloc (numberOfParticles*sizeof(float3));
float3* particleColor_d;


// cell arrays
Cell* cellArray_h = (Cell*) malloc(numberOfCells*sizeof(Cell));
Cell* cellArray_d;


// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3*boundary;
float translate_x = -boundary/2.0;
float translate_y = -boundary/2.0;


////////////////////////////////////////////////////////////////////////////////
// kernels
#include <runKernel.cu>

void initializeParameters();
void initializeParticles();
void initializeCells();

void copyParticlesFromHostToDevice();
void copyCellsFromHostToDevice();


void copyParticlesFromDeviceToHost();
void copyCellsFromDeviceToHost();

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);



// Cuda functionality
void runCuda();

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv)
{
	runTest( argc, argv);

    CUT_EXIT(argc, argv);
}


void initializeParameters()
{
	params.mass=mass;
	params.maxParticles= numberOfParticles;
	params.maxParticlesPerCell= maxParticlesPerCell;
	params.cellsPerDim=numberOfCellsPerDim;
	params.boundary=boundary;
	params.cellSize=cell_size;
	params.particleRadious=particle_size;
	params.spring=spring;
	params.globalDamping=globalDamping;
	params.shear=shear;
	params.attraction=attraction;
	params.boundaryDamping=boundaryDamping;
	params.gravity=make_float3(0.0f);
	params.gravity.y= gravityValue;
	params.collisionDamping=collisionDamping;

}

void initializeCells()
{
	for(unsigned int i = 0; i < numberOfCellsPerDim; i++)
	{
		for(unsigned int j = 0; j < numberOfCellsPerDim; j++)
		{
			for(unsigned int k = 0; k < numberOfCellsPerDim; k++)
			{
				int cellidx=(i*numberOfCellsPerDim+j)*numberOfCellsPerDim + k;
				cellArray_h[cellidx].coordinates.x=i;
				cellArray_h[cellidx].coordinates.y=j;
				cellArray_h[cellidx].coordinates.z=k;
				cellArray_h[cellidx].counter=0;

				for (int m=0; m<maxParticlesPerCell;m++)
				{
					cellArray_h[cellidx].particleidxs[m]=-1;
				}
			}
		}
	}
}

void initializeParticles()
{
	for(unsigned int i = 0; i < numberOfParticles; i++)
	{
		particlePosition_h[i].x = ((rand() / ((unsigned)RAND_MAX + 1.0)) * (float)boundary);
		particlePosition_h[i].y = ((rand() / ((unsigned)RAND_MAX + 1.0)) * (float)boundary);
		particlePosition_h[i].z = ((rand() / ((unsigned)RAND_MAX + 1.0)) * (float)boundary);

		particleColor_h[i].x = (rand() / ((unsigned)RAND_MAX + 1.0));
		particleColor_h[i].y = (rand() / ((unsigned)RAND_MAX + 1.0));
		particleColor_h[i].z = (rand() / ((unsigned)RAND_MAX + 1.0));

		particleVelocity_h[i].x = ((rand() / ((unsigned)RAND_MAX + 1.0)) * (float)(maxVelocity - minVelocity) + (float)minVelocity);
		particleVelocity_h[i].y = ((rand() / ((unsigned)RAND_MAX + 1.0)) * (float)(maxVelocity - minVelocity) + (float)minVelocity);
		particleVelocity_h[i].z = ((rand() / ((unsigned)RAND_MAX + 1.0)) * (float)(maxVelocity - minVelocity) + (float)minVelocity);

		int cell_x= (int) floor(particlePosition_h[i].x/ cell_size);
		int cell_y= (int) floor(particlePosition_h[i].y/ cell_size);
		int cell_z= (int) floor(particlePosition_h[i].z/ cell_size);

		int cellidx=((int)(((int)cell_x*numberOfCellsPerDim)+cell_y)*numberOfCellsPerDim) + cell_z;
		particlePosition_h[i].w= (float)cellidx;

		if(cellArray_h[cellidx].counter< maxParticlesPerCell)
		{
			cellArray_h[cellidx].particleidxs[cellArray_h[cellidx].counter]=i;
			cellArray_h[cellidx].counter=cellArray_h[cellidx].counter+1;
		}
	}

}

void copyParticlesFromHostToDevice()
{
	int sizePosition = numberOfParticles*sizeof(float4);

	cudaMalloc((void**)&particlePosition_d, sizePosition);
	cudaMemcpy(particlePosition_d, particlePosition_h, sizePosition, cudaMemcpyHostToDevice);

	int sizeVelocity = numberOfParticles*sizeof(float3);
	cudaMalloc((void**)&particleVelocity_d, sizeVelocity);
	cudaMemcpy(particleVelocity_d, particleVelocity_h, sizeVelocity, cudaMemcpyHostToDevice);

	int sizeColor = numberOfParticles*sizeof(float3);
	cudaMalloc((void**)&particleColor_d, sizeColor);
	cudaMemcpy(particleColor_d, particleColor_h, sizeColor, cudaMemcpyHostToDevice);
}

void copyParticlesFromDeviceToHost()
{
	int sizePosition = numberOfParticles*sizeof(float4);
	cudaMemcpy(particlePosition_h, particlePosition_d, sizePosition,cudaMemcpyDeviceToHost);

	int sizeVelocity = numberOfParticles*sizeof(float3);
	cudaMemcpy(particleVelocity_h, particleVelocity_d, sizeVelocity,cudaMemcpyDeviceToHost);

	int sizeColor = numberOfParticles*sizeof(float3);
	cudaMemcpy(particleColor_h, particleColor_d, sizeColor,cudaMemcpyDeviceToHost);


}

void copyCellsFromHostToDevice()
{
	int size = numberOfCells*sizeof(Cell);

	cudaMalloc((void**)&cellArray_d, size);

	cudaMemcpy(cellArray_d, cellArray_h, size, cudaMemcpyHostToDevice);
}


void copyCellsFromDeviceToHost()
{
	int size = numberOfCells*sizeof(Cell);
	cudaMemcpy(cellArray_h, cellArray_d, size,cudaMemcpyDeviceToHost);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest( int argc, char** argv)
{
    CUT_DEVICE_INIT(argc, argv);

    initializeParameters();
    initializeCells();
    initializeParticles();

    // run the cuda part
    copyParticlesFromHostToDevice();
    copyCellsFromHostToDevice();

    cutCreateTimer(&timer);
    for (int i=0; i<300; i++)
	{
		runCuda();
	}

}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda()
{

    // execute the kernel
    dim3 block(numberOfParticlesPerBlock, 1, 1);
    dim3 grid(numberOfParticles / block.x, 1, 1);

    cutStartTimer(timer);
    runKernel<<< grid, block>>>(params, particlePosition_d, particleVelocity_d, particleColor_d,cellArray_d, deltaTime);
    copyParticlesFromDeviceToHost();
    cutStopTimer(timer);
	float milliseconds = cutGetTimerValue(timer);
	iterations=iterations+1;
	printf("%d particles, iterations %d , total time %0f ms\n", numberOfParticles,iterations, milliseconds/iterations);

}


