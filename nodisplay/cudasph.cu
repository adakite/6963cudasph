

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

#include <sphere.c>
#include <particle.h>


const float particle_size = 0.2f;
const float cell_size = 2.0f* particle_size;

const float maxVelocity = 1.0f;
const float minVelocity = -1.0f;
const int boundary= 32.0f;
unsigned int timer;
unsigned int iterations;

const unsigned int numberOfParticles = 10240;
const unsigned int numberOfParticlesPerBlock = 128;
const unsigned int numberOfCellsPerDim=((int)floor((boundary)/cell_size));
const unsigned int numberOfCells= numberOfCellsPerDim*numberOfCellsPerDim*numberOfCellsPerDim;
const unsigned int maxParticlesPerCell=4;
const float deltaTime=0.1f;

const float mass=1.5f;
const float spring=0.2f;
const float damping=0.2f;
const float shear=0.1f;
const float attraction= 0.01f;
const float gravity= -0.2f;
const float boundaryDamping=0.3f;

Parameters params;


// particle arrays
Particle* particleArray_h = (Particle*) malloc(numberOfParticles*sizeof(Particle));
Particle* particleArray_d;

// cell arrays
Cell* cellArray_h = (Cell*) malloc(numberOfCells*sizeof(Cell));
Cell* cellArray_d;


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



// rendering callbacks
void display();
void mouse(int button, int state, int x, int y);
void keyboard( unsigned char key, int x, int y);
void motion(int x, int y);

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
	params.damping=damping;
	params.shear=shear;
	params.attraction=attraction;
	params.gravity=gravity;
	params.boundaryDamping=boundaryDamping;

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
		particleArray_h[i].position.x = ((rand() / ((unsigned)RAND_MAX + 1.0)) * (float)boundary);
		particleArray_h[i].position.y = ((rand() / ((unsigned)RAND_MAX + 1.0)) * (float)boundary);
		particleArray_h[i].position.z = ((rand() / ((unsigned)RAND_MAX + 1.0)) * (float)boundary);

		particleArray_h[i].color.x = (rand() / ((unsigned)RAND_MAX + 1.0));
		particleArray_h[i].color.y = (rand() / ((unsigned)RAND_MAX + 1.0));
		particleArray_h[i].color.z = (rand() / ((unsigned)RAND_MAX + 1.0));

		particleArray_h[i].velocity.x = ((rand() / ((unsigned)RAND_MAX + 1.0)) * (float)(maxVelocity - minVelocity) + (float)minVelocity);
		particleArray_h[i].velocity.y = ((rand() / ((unsigned)RAND_MAX + 1.0)) * (float)(maxVelocity - minVelocity) + (float)minVelocity);
		particleArray_h[i].velocity.z = ((rand() / ((unsigned)RAND_MAX + 1.0)) * (float)(maxVelocity - minVelocity) + (float)minVelocity);

		int cell_x= (int) floor(particleArray_h[i].position.x/ cell_size);
		int cell_y= (int) floor(particleArray_h[i].position.y/ cell_size);
		int cell_z= (int) floor(particleArray_h[i].position.z/ cell_size);

		int cellidx=((int)(((int)cell_x*numberOfCellsPerDim)+cell_y)*numberOfCellsPerDim) + cell_z;
		particleArray_h[i].cellidx= cellidx;

		if(cellArray_h[cellidx].counter< maxParticlesPerCell)
		{
			cellArray_h[cellidx].particleidxs[cellArray_h[cellidx].counter]=i;
			cellArray_h[cellidx].counter=cellArray_h[cellidx].counter+1;
		}
	}

}

void copyParticlesFromHostToDevice()
{
	int size = numberOfParticles*sizeof(Particle);

	cudaMalloc((void**)&particleArray_d, size);

	cudaMemcpy(particleArray_d, particleArray_h, size, cudaMemcpyHostToDevice);
}

void copyParticlesFromDeviceToHost()
{
	int size = numberOfParticles*sizeof(Particle);
	cudaMemcpy(particleArray_h, particleArray_d, size,cudaMemcpyDeviceToHost);
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

	runKernel<<< grid, block>>>(params, particleArray_d,cellArray_d, deltaTime);


    copyParticlesFromDeviceToHost();
    cutStopTimer(timer);
	float milliseconds = cutGetTimerValue(timer);
	iterations=iterations+1;
	printf("%d Particles, iterations %d , total time %f ms\n", numberOfParticles,iterations, milliseconds/iterations);

}



