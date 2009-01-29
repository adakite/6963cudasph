
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
#include <sphere.c>



////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width = 1024;
const unsigned int window_height = 1024;

const float particle_size = 0.2f;
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
const float spring=2.0f;
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

/////////////////////////////////////////////////////////////////////////////////
// vbo variables
GLuint vbo;
GLuint cbo;
GLuint nbo;



// particle arrays
Particle* particleArray_h = (Particle*) malloc(numberOfParticles*sizeof(Particle));
Particle* particleArray_d;

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

// GL functionality
CUTBoolean initGL();
void createVBO(GLuint* vbo, GLuint* cbo, GLuint* nbo);
void deleteVBO(GLuint* vbo, GLuint* cbo, GLuint* nbo);



// rendering callbacks
void display();
void mouse(int button, int state, int x, int y);
void keyboard( unsigned char key, int x, int y);
void motion(int x, int y);

// Cuda functionality
void runCuda(GLuint vbo, GLuint cbo, GLuint nbo);

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

    // create VBO
	createVBO(&vbo, &cbo,&nbo);

    // run the cuda part
    copyParticlesFromHostToDevice();
    copyCellsFromHostToDevice();

    cutCreateTimer(&timer);
    runCuda(vbo, cbo,nbo);
    // start rendering mainloop
    glutMainLoop();
}


////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(GLuint vbo, GLuint cbo, GLuint nbo)
{
	// map OpenGL buffer object for writing from CUDA
	float4 *dptr;
	cudaGLMapBufferObject( (void**)&dptr, vbo);
	float3 *cptr;
	cudaGLMapBufferObject( (void**)&cptr, cbo);
	float3 *nptr;
	cudaGLMapBufferObject( (void**)&nptr, nbo);

    // execute the kernel
    dim3 block(numberOfParticlesPerBlock, 1, 1);
    dim3 grid(numberOfParticles / block.x, 1, 1);

    cutStartTimer(timer);
    runKernel<<< grid, block>>>(dptr,cptr,nptr, params, particleArray_d,cellArray_d, deltaTime);
    //copyParticlesFromDeviceToHost();
    cutStopTimer(timer);
	float milliseconds = cutGetTimerValue(timer);
	iterations=iterations+1;
	printf("%d particles, iterations %d , total time %0f ms\n", numberOfParticles,iterations, milliseconds/iterations);
    // unmap buffer object
	cudaGLUnmapBufferObject(vbo);
	cudaGLUnmapBufferObject(cbo);
	cudaGLUnmapBufferObject(nbo);
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
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint* vbo, GLuint* cbo,GLuint* nbo)
{
    // create buffer object
    glGenBuffers( 1, vbo);
    glBindBuffer( GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    unsigned int size = numberOfParticles * SPHERE_VERTICES_SIZE * 4 * sizeof(float);
    glBufferData( GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer( GL_ARRAY_BUFFER, 0);

    // register buffer object with CUDA
    cudaGLRegisterBufferObject(*vbo);

    // create buffer object
	glGenBuffers( 1, cbo);
	glBindBuffer( GL_ARRAY_BUFFER, *cbo);

	// initialize buffer object
	unsigned int sizec = numberOfParticles * SPHERE_VERTICES_SIZE * 3 * sizeof(float);
	glBufferData( GL_ARRAY_BUFFER, sizec, 0, GL_DYNAMIC_DRAW);

	glBindBuffer( GL_ARRAY_BUFFER, 0);

	// register buffer object with CUDA
	cudaGLRegisterBufferObject(*cbo);

	// create buffer object
	glGenBuffers( 1, nbo);
	glBindBuffer( GL_ARRAY_BUFFER, *nbo);

	// initialize buffer object
	unsigned int sizen = numberOfParticles * SPHERE_VERTICES_SIZE * 3 * sizeof(float);
	glBufferData( GL_ARRAY_BUFFER, sizen, 0, GL_DYNAMIC_DRAW);

	glBindBuffer( GL_ARRAY_BUFFER, 0);

	// register buffer object with CUDA
	cudaGLRegisterBufferObject(*nbo);
    CUT_CHECK_ERROR_GL();
}


////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO( GLuint* vbo, GLuint* cbo,GLuint* nbo)
{
    glBindBuffer( 1, *vbo);
    glDeleteBuffers( 1, vbo);

    cudaGLUnregisterBufferObject(*vbo);

    *vbo = 0;

    glBindBuffer( 1, *cbo);
	glDeleteBuffers( 1, cbo);

	cudaGLUnregisterBufferObject(*cbo);

	*cbo = 0;

	glBindBuffer( 1, *nbo);
	glDeleteBuffers( 1, nbo);

	cudaGLUnregisterBufferObject(*nbo);

	*nbo = 0;
}

void display()
{

    runCuda(vbo,cbo,nbo);

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

	glShadeModel(GL_SMOOTH);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, cbo);
	glColorPointer(3,GL_FLOAT,0,0);

	glBindBuffer(GL_ARRAY_BUFFER,nbo);
	glNormalPointer(GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glEnableClientState (GL_NORMAL_ARRAY);

	glDrawArrays(GL_TRIANGLES, 0, numberOfParticles*60);
	glDisableClientState(GL_VERTEX_ARRAY);


    // Draw the particles

/*
	for(int i=0; i<numberOfParticles; i++)
	{
		glPushMatrix();
		glColor3f(particleArray_h[i].color.x,particleArray_h[i].color.y,particleArray_h[i].color.z);
		glTranslatef(particleArray_h[i].position.x,particleArray_h[i].position.y,particleArray_h[i].position.z );
		glutSolidSphere(particle_size,20,20);
		glPopMatrix();

	}*/



	//glDisable ( GL_LIGHTING ) ;

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

