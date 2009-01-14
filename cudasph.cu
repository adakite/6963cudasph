/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
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

#include <particle.h>

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width = 1024;
const unsigned int window_height = 1024;

const float particle_size = 0.2f;

const float maxVelocity = 0.1;
const float minVelocity = -0.1;

const unsigned int numberOfParticles = 100;

Vector3D boundary;

Particle* particleArray_h = (Particle*) malloc(numberOfParticles*sizeof(Particle));
Particle* particleArray_d;

float anim = 0.0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -24.0;

////////////////////////////////////////////////////////////////////////////////
// kernels
#include <particleInteraction_kernel.cu>
#include <updatePosition_kernel.cu>

void initializeParticles();
void copyParticlesFromHostToDevice();


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

// GL functionality
CUTBoolean initGL();

// rendering callbacks
void display();
void mouse(int button, int state, int x, int y);
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


void initializeParticles()
{
	boundary.x = 16;
	boundary.y = 16;
	boundary.z = 16;

	for(unsigned int i = 0; i < numberOfParticles; i++)
	{
		particleArray_h[i].position.x = (rand() / ((unsigned)RAND_MAX + 1.0)) * (float)boundary.x;
		particleArray_h[i].position.y = (rand() / ((unsigned)RAND_MAX + 1.0)) * (float)boundary.y;
		particleArray_h[i].position.z = (rand() / ((unsigned)RAND_MAX + 1.0)) * (float)boundary.z;

		particleArray_h[i].velocity.x = ((rand() / ((unsigned)RAND_MAX + 1.0)) * (float)(maxVelocity - minVelocity) + (float)minVelocity);
		particleArray_h[i].velocity.y = ((rand() / ((unsigned)RAND_MAX + 1.0)) * (float)(maxVelocity - minVelocity) + (float)minVelocity);
		particleArray_h[i].velocity.z = ((rand() / ((unsigned)RAND_MAX + 1.0)) * (float)(maxVelocity - minVelocity) + (float)minVelocity);

		printf("Position {%f,%f,%f}\n",particleArray_h[i].position.x,particleArray_h[i].position.y,particleArray_h[i].position.z );
		printf("Velocity {%f,%f,%f}\n",particleArray_h[i].velocity.x,particleArray_h[i].velocity.y,particleArray_h[i].velocity.z );
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
    glutCreateWindow( "Cuda GL interop");

    // initialize GL
    if( CUTFalse == initGL()) {
        return;
    }

    // register callbacks
    glutDisplayFunc( display);
    glutMouseFunc( mouse);
    glutMotionFunc( motion);

    // run the cuda part
    initializeParticles();

    runCuda();

    // start rendering mainloop
    glutMainLoop();
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda()
{

	copyParticlesFromHostToDevice();
    // execute the kernel
    dim3 block(1, 1, 1);
    dim3 grid(numberOfParticles / block.x, 1, 1);
    //particleInteraction<<< grid, block>>>(dptr, mesh_width, mesh_height, anim);
    updatePosition<<< grid, block>>>(boundary, particleArray_d);

    copyParticlesFromDeviceToHost();

    for(unsigned int i = 0; i < numberOfParticles; i++)
    {
		printf("Updated Position {%f,%f,%f}\n",particleArray_h[i].position.x,particleArray_h[i].position.y,particleArray_h[i].position.z );
		printf("Updated Velocity {%f,%f,%f}\n",particleArray_h[i].velocity.x,particleArray_h[i].velocity.y,particleArray_h[i].velocity.z );
    }

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
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

    CUT_CHECK_ERROR_GL();

    return CUTTrue;
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    // run CUDA kernel to generate vertex positions
    runCuda();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    glEnable(GL_DEPTH_TEST);

    // Draw the particles
	for(int i=0; i<numberOfParticles; i++)
	{
		printf("Draw {%f,%f,%f}\n",particleArray_h[i].position.x,particleArray_h[i].position.y,particleArray_h[i].position.z );
		glPushMatrix();

		glColor3f(0.0f,0.0f,0.0f);
		glTranslatef(particleArray_h[i].position.x,particleArray_h[i].position.y,particleArray_h[i].position.z );
		glutSolidSphere(particle_size,20,20);

		glPopMatrix();
	}

	// Draw the ground
	glColor3f(0.9f, 0.9f, 0.9f);
	glBegin(GL_QUADS);
			glVertex3f(-16.0f, 0.0f, -16.0f);
			glVertex3f(-16.0f, 0.0f,  16.0f);
			glVertex3f( 16.0f, 0.0f,  16.0f);
			glVertex3f( 16.0f, 0.0f, -16.0f);
	glEnd();



    glutSwapBuffers();
    glutPostRedisplay();

    anim += 0.01;
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

