/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/*
    This example demonstrates how to use the Cuda OpenGL bindings to
    dynamically modify a vertex buffer using a Cuda kernel.

    The steps are:
    1. Create an empty vertex buffer object (VBO)
    2. Register the VBO with Cuda
    3. Map the VBO for writing from Cuda
    4. Run Cuda kernel to modify the vertex positions
    5. Unmap the VBO
    6. Render the results using OpenGL

    Host code
*/


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

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width = 1024;
const unsigned int window_height = 1024;

const float particle_size = 0.5f;
const float cell_size = 2.0f* particle_size;

const float maxVelocity = 0.1;
const float minVelocity = -0.1;
const float boundary= 32.0;

const unsigned int numberOfParticles = 10240;
const unsigned int numberOfParticlesPerBlock = 512;
const unsigned int numberOfCells= ((int)floor((boundary)/cell_size))*((int)floor((boundary)/cell_size))*((int)floor((boundary)/cell_size));


float anim = 0.0;

// vbo variables
GLuint vbo;

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
float translate_z = -84.0;
float translate_x = -16.0;
float translate_y = -16.0;

////////////////////////////////////////////////////////////////////////////////
// kernels
#include <updatePosition_kernel.cu>

void initializeParticles();
void initializeCells();
void initializeNeighbors();


void copyParticlesFromHostToDevice();
void copyCellsFromHostToDevice();

void copyParticlesFromDeviceToHost();
void copyCellsFromDeviceToHost();

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

// GL functionality
CUTBoolean initGL();
void createVBO(GLuint* vbo);
void deleteVBO(GLuint* vbo);


// rendering callbacks
void display();
void mouse(int button, int state, int x, int y);
void keyboard( unsigned char key, int x, int y);
void motion(int x, int y);

// Cuda functionality
void runCuda(GLuint vbo);



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv)
{
	runTest( argc, argv);

    CUT_EXIT(argc, argv);
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

		particleArray_h[i].cellidx= (cell_x*boundary+cell_y)*boundary + cell_z;
		particleArray_h[i].next= -1;
	}

}

void initializeCells()
{
	for(unsigned int i = 0; i < boundary; i++)
	{
		for(unsigned int j = 0; j < boundary; j++)
		{
			for(unsigned int k = 0; k < boundary; k++)
			{
				int cellidx=(i*boundary+j)*boundary + k;
				cellArray_h[cellidx].coordinates.x=i;
				cellArray_h[cellidx].coordinates.y=j;
				cellArray_h[cellidx].coordinates.z=k;

				int minp=numberOfParticles;
				for(unsigned int p = 0; p < numberOfParticles; p++)
				{
					if(particleArray_h[p].cellidx== cellidx && p<minp)
					{
						minp=p;
					}
				}

				if (minp==numberOfParticles)
				{
					minp=-1;
				}
				cellArray_h[cellidx].head=minp;
			}
		}
	}
}

void initializeNeighbors()
{
	for(unsigned int p = 0; p < numberOfParticles; p++)
	{
		int minq=numberOfParticles;
		for(unsigned int q = 0; q < numberOfParticles; q++)
		{
			if(particleArray_h[p].cellidx==particleArray_h[q].cellidx && q<minq && q>p)
			{
				minq=q;
			}
		}
		if (minq==numberOfParticles)
		{
			minq=-1;
		}
		particleArray_h[p].next=minq;
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

    initializeParticles();
    initializeCells();
    initializeNeighbors();

    // create VBO
	createVBO(&vbo);

    // run the cuda part
    copyParticlesFromHostToDevice();
    copyCellsFromHostToDevice();

    runCuda(vbo);

    // start rendering mainloop
    glutMainLoop();
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(GLuint vbo)
{
	// map OpenGL buffer object for writing from CUDA
	float4 *dptr;
	cudaGLMapBufferObject( (void**)&dptr, vbo);

    // execute the kernel
    dim3 block(1, 1, 1);
    dim3 grid(numberOfParticles / block.x, 1, 1);
    //particleInteraction<<< grid, block>>>(dptr, mesh_width, mesh_height, anim);
    updatePosition<<< grid, block>>>(dptr, boundary, particleArray_d,cellArray_d);

    //copyParticlesFromDeviceToHost();
    // unmap buffer object
	cudaGLUnmapBufferObject(vbo);
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
void createVBO(GLuint* vbo)
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

    //CUT_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO( GLuint* vbo)
{
    glBindBuffer( 1, *vbo);
    glDeleteBuffers( 1, vbo);

    cudaGLUnregisterBufferObject(*vbo);

    *vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{

    runCuda(vbo);

    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_LIGHT0);

    GLfloat lightpos[] = {16.0, 16.0, 16.0, 1.0};
    glLightfv(GL_LIGHT0, GL_POSITION, lightpos);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(translate_x, translate_y, translate_z);

    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    //Draw the walls

    glDepthMask(GL_FALSE);

	glPushMatrix();
	// Draw the Back
	glColor3f(0.7f, 0.7f, 0.7f);
	glBegin(GL_QUADS);
			glVertex3f(0.0f, 0.0f, 0.0f);
			glVertex3f(0.0f, 32.0f, 0.0f);
			glVertex3f( 32.0f, 32.0f, 0.0f);
			glVertex3f( 32.0f, 0.0f, 0.0f);
	glEnd();
	// Draw the left
	glColor3f(0.7f, 0.7f, 0.7f);
	glBegin(GL_QUADS);
			glVertex3f(0.0f, 32.0f, 32.0f);
			glVertex3f(0.0f, 0.0f, 32.0f);
			glVertex3f(0.0f, 0.0f, 0.0f);
			glVertex3f(0.0f, 32.0f, 0.0f);
	glEnd();
	// Draw the right
	glColor3f(0.7f, 0.7f, 0.7f);
	glBegin(GL_QUADS);
			glVertex3f(32.0f, 32.0f, 32.0f);
			glVertex3f(32.0f, 0.0f, 32.0f);
			glVertex3f(32.0f, 0.0f, 0.0f);
			glVertex3f(32.0f, 32.0f, 0.0f);
	glEnd();
	// Draw the top
	glColor3f(0.7f, 0.7f, 0.7f);
	glBegin(GL_QUADS);
			glVertex3f(32.0f, 32.0f, 32.0f);
			glVertex3f(0.0f, 32.0f, 32.0f);
			glVertex3f(0.0f, 32.0f, 0.0f);
			glVertex3f(32.0f, 32.0f, 0.0f);
	glEnd();
	// Draw the bottom
	glColor3f(0.7f, 0.7f, 0.7f);
	glBegin(GL_QUADS);
			glVertex3f(32.0f, 0.0f, 32.0f);
			glVertex3f(0.0f, 0.0f, 32.0f);
			glVertex3f(0.0f, 0.0f, 0.0f);
			glVertex3f(32.0f, 0.0f, 0.0f);
	glEnd();
	glPopMatrix();

	glDepthMask(GL_TRUE);


    // render from the vbo
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glColor3f(1.0, 0.0, 0.0);
	glDrawArrays(GL_SPHERE, 0, numberOfParticles);
	glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();
	glutPostRedisplay();

	/*

    // Draw the particles
	for(int i=0; i<numberOfParticles; i++)
	{
		glPushMatrix();
		glColor3f(particleArray_h[i].color.x,particleArray_h[i].color.y,particleArray_h[i].color.z);
		glTranslatef(particleArray_h[i].position.x,particleArray_h[i].position.y,particleArray_h[i].position.z );

		glutSolidSphere(particle_size,20,20);
		glPopMatrix();

	}

	//glDisable ( GL_LIGHTING ) ;

    glutSwapBuffers();
    glutPostRedisplay();
    */

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
	 }

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

