
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

const unsigned int numberOfParticles = 1024;
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

CUTBoolean initGL();
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

    // Create GL context
    glutInit( &argc, argv);
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize( window_width, window_height);
    glutCreateWindow( "Cuda Smoothed particle hydrodynamics Simulation");

    // initialize GL
    if( CUTFalse == initGL()) {
        return;
    }

    // register callbacks
    glutDisplayFunc( display);
    glutKeyboardFunc( keyboard);
    glutMouseFunc( mouse);
    glutMotionFunc( motion);


    initializeParameters();
    initializeCells();
    initializeParticles();

    // run the cuda part
    copyParticlesFromHostToDevice();
    copyCellsFromHostToDevice();

    cutCreateTimer(&timer);
    runCuda();
    // start rendering mainloop
    glutMainLoop();
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

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
CUTBoolean initGL()
{
    // initialize necessary OpenGL extensions
    glewInit();
    if (! glewIsSupported( "GL_VERSION_2_0 "
        "GL_ARB_pixel_buffer_object"
		)) {
        fprintf( stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush( stderr);
        return CUTFalse;
    }


    // default initialization
    glClearColor( 1.0, 1.0, 1.0, 1.0);
    glDisable( GL_DEPTH_TEST);

    // viewport
    glViewport( 0, 0, window_width, window_height);

    // projection
    glMatrixMode( GL_PROJECTION);
    glLoadIdentity();
    //gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);
    gluPerspective( /* field of view in degree */ 40.0,
      /* aspect ratio */ 1.0,
        /* Z near */ 0.5, /* Z far */ 150.0);


    //CUT_CHECK_ERROR_GL();

    return CUTTrue;
}
////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{

    runCuda();

    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_LIGHT0);

    GLfloat lightpos[] = {-boundary/2, -boundary/2, -boundary/2, 1.0};
    glLightfv(GL_LIGHT0, GL_POSITION, lightpos);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(translate_x, translate_y, translate_z);

    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    //Draw the walls

    //glDepthMask(GL_FALSE);

	glPushMatrix();
	// Draw the Back
	glColor3f(0.7f, 0.7f, 0.7f);
	glBegin(GL_QUADS);
			glVertex3f(0.0f, 0.0f, 0.0f);
			glVertex3f(0.0f, boundary, 0.0f);
			glVertex3f( boundary, boundary, 0.0f);
			glVertex3f( boundary, 0.0f, 0.0f);
	glEnd();
	// Draw the left
	glColor3f(0.7f, 0.7f, 0.7f);
	glBegin(GL_QUADS);
			glVertex3f(0.0f, boundary, boundary);
			glVertex3f(0.0f, 0.0f, boundary);
			glVertex3f(0.0f, 0.0f, 0.0f);
			glVertex3f(0.0f, boundary, 0.0f);
	glEnd();
	// Draw the right
	glColor3f(0.7f, 0.7f, 0.7f);
	glBegin(GL_QUADS);
			glVertex3f(boundary, boundary, boundary);
			glVertex3f(boundary, 0.0f, boundary);
			glVertex3f(boundary, 0.0f, 0.0f);
			glVertex3f(boundary, boundary, 0.0f);
	glEnd();
	// Draw the top
	glColor3f(0.7f, 0.7f, 0.7f);
	glBegin(GL_QUADS);
			glVertex3f(boundary, boundary, boundary);
			glVertex3f(0.0f, boundary, boundary);
			glVertex3f(0.0f, boundary, 0.0f);
			glVertex3f(boundary, boundary, 0.0f);
	glEnd();
	// Draw the bottom
	glColor3f(0.7f, 0.7f, 0.7f);
	glBegin(GL_QUADS);
			glVertex3f(boundary, 0.0f, boundary);
			glVertex3f(0.0f, 0.0f, boundary);
			glVertex3f(0.0f, 0.0f, 0.0f);
			glVertex3f(boundary, 0.0f, 0.0f);
	glEnd();
	glPopMatrix();

	//Draw the wired cube
	glPushMatrix();
	glColor3f(1.0, 1.0, 1.0);
	glTranslatef(boundary/2,boundary/2,boundary/2);
	glutWireCube(boundary);
	glPopMatrix();

    // Draw the particles

	for(int i=0; i<numberOfParticles; i++)
	{
		glPushMatrix();
		glColor3f(particleColor_h[i].x,particleColor_h[i].y,particleColor_h[i].z);
		glTranslatef(particlePosition_h[i].x,particlePosition_h[i].y,particlePosition_h[i].z );
		glutSolidSphere(particle_size,20,20);
		glPopMatrix();

	}

    glutSwapBuffers();
    glutPostRedisplay();

    anim += 0.01;
}
////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard( unsigned char key, int /*x*/, int /*y*/)
{
	 switch( key)
	 {
	    case( 27) :
	        exit( 0);
	    case 'j':
	    	translate_z +=  1.0f;
			break;
		case 'J':
			translate_z -=  1.0f;
			break;
		case 'k':
			translate_x +=  1.0f;
			break;
		case 'K':
			translate_x -=  1.0f;
			break;
		case 'l':
			translate_y +=  1.0f;
			break;
		case 'L':
			translate_y -=  1.0f;
			break;

	 }
	     glutPostRedisplay(); // Redraw the scene


}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN) {
        mouse_buttons |= 1<<button;
    } else if (state == GLUT_UP) {
        mouse_buttons = 0;
    }
    mouse_old_x = x;
    mouse_old_y = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = x - mouse_old_x;
    dy = y - mouse_old_y;

    if (mouse_buttons & 1) {
        rotate_x += dy * 0.2;
        rotate_y += dx * 0.2;
    } else if (mouse_buttons & 4) {
        translate_z += dy * 0.01;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

