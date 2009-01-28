////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
//
//				Particle Simulation with CUDA
//				Sergio Herrero and Kyle Peter Fritz
//				sherrero@mit.edu, kfritz@mit.edu
//				January 2009
//
//
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// Constants and macros
////////////////////////////////////////////////////////////////////////////////

// Macro for using Vertex Buffer Objects to display particles
#define USE_VBO

// Macro for designation whether OpenGL should run in a separate thread
//#define SEPARATE_GL

// Macro for using thread-per-cell approach for calculating collisions
//#define THREAD_PER_CELL_COLLISIONS

// Macro for not displaying graphics
//#define NO_DISPLAY

// Constants, Display
#define WINDOW_WIDTH 1024
#define WINDOW_HEIGHT 1024

// Constants, Simulation
const float deltaTime=0.005f;



////////////////////////////////////////////////////////////////////////////////
// Includes and environment settings
////////////////////////////////////////////////////////////////////////////////

// Check for Windows
#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// Includes, System
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

// Includes, OpenGL
#include <GL/glew.h>
// Check for Apple
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

// Includes, CUDA
#include <cuda.h>
#include <cutil.h>
#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>
#include <cutil_math.h>

// Includes, Particle Simulation
#include <parameters.c>
#include <assert.cu>
#include <vectors.cu>
#include <sphere.c>
#include <particle.h>
#include <cell.h>
#include <particle.cu>
#include <cell.cu>



////////////////////////////////////////////////////////////////////////////////
// Global Variables
////////////////////////////////////////////////////////////////////////////////

// Variables, Simulation
// Particle arrays
Particle* particleArray_h = (Particle*) malloc(NUMBER_OF_PARTICLES*sizeof(Particle));
Particle* particleArray_d;

// Cell arrays
Cell* cellArray_h = (Cell*) malloc(CELLS_IN_X*CELLS_IN_Y*CELLS_IN_Z*sizeof(Cell));
Cell* cellArray_d;

// Color array
float3* colorArray_h = (float3*) malloc(NUMBER_OF_PARTICLES * sizeof(float3));

// Quit variable
bool continueRunning = true;

// Syncronization variables
bool requestMemCopy = true;
bool readyToDraw = false;

// Timers and counters
unsigned int timer;
unsigned int iterations;
unsigned int drawTimer;

// Variables, Display
// Vertex buffer object
GLuint vBuffer;
// Mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_x = -BOUNDARY_X/2.0;
float translate_y = -BOUNDARY_Y/2.0;
float translate_z = -3*BOUNDARY_Z;



////////////////////////////////////////////////////////////////////////////////
// Function Stubs
////////////////////////////////////////////////////////////////////////////////

// Initialization, Simulation
void initializeCells();
void initializeParticles();

// Initialization, Display
CUTBoolean initGL();
void createVBO(GLuint* vbo);
void deleteVBO(GLuint* vbo);

// Memory copy functions for CUDA
void copyParticlesFromHostToDevice();
void copyCellsFromHostToDevice();
void copyParticlesFromDeviceToHost();
void copyCellsFromDeviceToHost();

// Display callbacks
#ifdef SEPARATE_GL
	void *openGLLoop(void *threadId);
#endif
void display();
void mouse(int button, int state, int x, int y);
void keyboard( unsigned char key, int x, int y);
void motion(int x, int y);

// Run simulation
void runCuda();

// Begin simulation
void runSimulation( int argc, char** argv);



////////////////////////////////////////////////////////////////////////////////
// Functions
////////////////////////////////////////////////////////////////////////////////

// Initialize Cells
// Populates host cell array with cells containing NO_PARTICLEs.
void initializeCells()
{
	for(unsigned int i = 0; i < CELLS_IN_X; i++)
	{
		for(unsigned int j = 0; j < CELLS_IN_Y; j++)
		{
			for(unsigned int k = 0; k < CELLS_IN_Z; k++)
			{
				int cellidx=(i*CELLS_IN_Y+j)*CELLS_IN_Z + k;
				cellArray_h[cellidx].coordinates.x=i;
				cellArray_h[cellidx].coordinates.y=j;
				cellArray_h[cellidx].coordinates.z=k;
				cellArray_h[cellidx].numberOfParticles=0;

				for (int m=0; m<MAX_PARTICLES_PER_CELL;m++)
				{
					cellArray_h[cellidx].particleidxs[m] = NO_PARTICLE;
				}
			}
		}
	}
}

// Initialize Particles
// Populates host particle arrays with randomly positioned particles.
// Particles have random velocities and random colors.
void initializeParticles()
{
	for(unsigned int i = 0; i < NUMBER_OF_PARTICLES; i++)
	{
		particleArray_h[i].position.x = ((rand() / ((unsigned)RAND_MAX + 1.0)) * (float)BOUNDARY_X);
		particleArray_h[i].position.y = ((rand() / ((unsigned)RAND_MAX + 1.0)) * (float)BOUNDARY_Y);
		particleArray_h[i].position.z = ((rand() / ((unsigned)RAND_MAX + 1.0)) * (float)BOUNDARY_Z);

		colorArray_h[i].x = (rand() / ((unsigned)RAND_MAX + 1.0));
		colorArray_h[i].y = (rand() / ((unsigned)RAND_MAX + 1.0));
		colorArray_h[i].z = (rand() / ((unsigned)RAND_MAX + 1.0));

		particleArray_h[i].velocity.x = ((rand() / ((unsigned)RAND_MAX + 1.0)) * (float)(MAX_VELOCITY - MIN_VELOCITY) + (float)MIN_VELOCITY);
		particleArray_h[i].velocity.y = ((rand() / ((unsigned)RAND_MAX + 1.0)) * (float)(MAX_VELOCITY - MIN_VELOCITY) + (float)MIN_VELOCITY);
		particleArray_h[i].velocity.z = ((rand() / ((unsigned)RAND_MAX + 1.0)) * (float)(MAX_VELOCITY - MIN_VELOCITY) + (float)MIN_VELOCITY);

		particleArray_h[i].collisionForce.x = 0.0;
		particleArray_h[i].collisionForce.y = 0.0;
		particleArray_h[i].collisionForce.z = 0.0;
	}

	// Set two particles to travel at each other.
	particleArray_h[0].position.x = 16.0;
	particleArray_h[0].position.y = 1.0;
	particleArray_h[0].position.z = 16.0;

	particleArray_h[1].position.x = 16.0;
	particleArray_h[1].position.y = 31.0;
	particleArray_h[1].position.z = 16.0;

	particleArray_h[0].velocity.x = 0;
	particleArray_h[0].velocity.y = 3.0;
	particleArray_h[0].velocity.z = 0;

	particleArray_h[1].velocity.x = 0;
	particleArray_h[1].velocity.y = -3.0;
	particleArray_h[1].velocity.z = 0;
}

// Init GL
// Initializes OpenGL and prepares the window.
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
    glViewport( 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

    // projection
    glMatrixMode( GL_PROJECTION);
    glLoadIdentity();
    //gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);
    gluPerspective( /* field of view in degree */ 40.0,
      /* aspect ratio */ 1.0,
        /* Z near */ 0.5, /* Z far */ 150.0);

    CUT_CHECK_ERROR_GL();

    return CUTTrue;
}

// Create VBO
// Creates a Vertex Buffer Object and sets the given pointer.
// Vertex Buffer Object is registered with CUDA.
void createVBO(GLuint* vbo)
{
    // create buffer object
    glGenBuffers( 1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    //unsigned int size = NUMBER_OF_PARTICLES * SPHERE_VERTICES_SIZE * 4 * sizeof(float);
    unsigned int size = NUMBER_OF_PARTICLES * sizeof(float4);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register buffer object with CUDA
    cudaGLRegisterBufferObject(*vbo);

    CUT_CHECK_ERROR_GL();
}

// Delete VBO
// Deletes a Vertex Buffer Object and resets the given pointer.
// Vertex Buffer Object is unregistered with CUDA.
void deleteVBO( GLuint* vbo)
{
    glBindBuffer( 1, *vbo);
    glDeleteBuffers( 1, vbo);

    cudaGLUnregisterBufferObject(*vbo);

    *vbo = 0;
}

// Copy Particles From Host To Device
// Copies the contents of the host particle arrays to the device particle arrays.
// Allocates space on device.
// Sets pointers for device particle arrays.
void copyParticlesFromHostToDevice()
{
	int size = NUMBER_OF_PARTICLES*sizeof(Particle);
	cudaMalloc((void**)&particleArray_d, size);
	cudaMemcpy(particleArray_d, particleArray_h, size, cudaMemcpyHostToDevice);
}

// Copy Particles From Device To Host
// Copies the contents of the device particle arrays to the host particle arrays.
void copyParticlesFromDeviceToHost()
{
	int size = NUMBER_OF_PARTICLES*sizeof(Particle);
	cudaMemcpy(particleArray_h, particleArray_d, size,cudaMemcpyDeviceToHost);
}

// Copy Cells From Host To Device
// Copies the contents of the host cell array to the device cell array.
// Allocates space on device.
// Sets pointer for device cell array.
void copyCellsFromHostToDevice()
{
	int size = CELLS_IN_X*CELLS_IN_Y*CELLS_IN_Z*sizeof(Cell);
	cudaMalloc((void**)&cellArray_d, size);
	cudaMemcpy(cellArray_d, cellArray_h, size, cudaMemcpyHostToDevice);
}

// Copy Cells From Device To Host
// Copies the contents of the device cell array to the host cell array.
void copyCellsFromDeviceToHost()
{
	int size = CELLS_IN_X*CELLS_IN_Y*CELLS_IN_Z*sizeof(Cell);
	cudaMemcpy(cellArray_h, cellArray_d, size,cudaMemcpyDeviceToHost);
}

#ifdef SEPARATE_GL
	// OpenGL Loop
	// Starts the rendering mainloop.
	void *openGLLoop(void *threadId)
	{
		// Start rendering mainloop
		glutMainLoop();
	}
#endif

// Display
// Display callback used by OpenGL to repaint.
void display()
{

	#ifndef SEPARATE_GL
		runCuda();
	#endif

	// Start CUDA timer
	cutCreateTimer(&drawTimer);
	cutStartTimer(drawTimer);

	#ifndef USE_VBO
		while(!readyToDraw)
		{
			//sleep(0.001);
		}
		readyToDraw = false;
	#endif


    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_LIGHT0);

    GLfloat lightpos[] = {-BOUNDARY_X/2, -BOUNDARY_Y/2, -BOUNDARY_Z/2, 1.0};
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
			glVertex3f(0.0f, BOUNDARY_Y, 0.0f);
			glVertex3f( BOUNDARY_X, BOUNDARY_Y, 0.0f);
			glVertex3f( BOUNDARY_X, 0.0f, 0.0f);
	glEnd();
	// Draw the left
	glColor3f(0.7f, 0.7f, 0.7f);
	glBegin(GL_QUADS);
			glVertex3f(0.0f, BOUNDARY_Y, BOUNDARY_Z);
			glVertex3f(0.0f, 0.0f, BOUNDARY_Z);
			glVertex3f(0.0f, 0.0f, 0.0f);
			glVertex3f(0.0f, BOUNDARY_Y, 0.0f);
	glEnd();
	// Draw the right
	glColor3f(0.7f, 0.7f, 0.7f);
	glBegin(GL_QUADS);
			glVertex3f(BOUNDARY_X, BOUNDARY_Y, BOUNDARY_Z);
			glVertex3f(BOUNDARY_X, 0.0f, BOUNDARY_Z);
			glVertex3f(BOUNDARY_X, 0.0f, 0.0f);
			glVertex3f(BOUNDARY_X, BOUNDARY_Y, 0.0f);
	glEnd();
	// Draw the top
	glColor3f(0.7f, 0.7f, 0.7f);
	glBegin(GL_QUADS);
			glVertex3f(BOUNDARY_X, BOUNDARY_Y, BOUNDARY_Z);
			glVertex3f(0.0f, BOUNDARY_Y, BOUNDARY_Z);
			glVertex3f(0.0f, BOUNDARY_Y, 0.0f);
			glVertex3f(BOUNDARY_X, BOUNDARY_Y, 0.0f);
	glEnd();
	// Draw the bottom
	glColor3f(0.7f, 0.7f, 0.7f);
	glBegin(GL_QUADS);
			glVertex3f(BOUNDARY_X, 0.0f, BOUNDARY_Z);
			glVertex3f(0.0f, 0.0f, BOUNDARY_Z);
			glVertex3f(0.0f, 0.0f, 0.0f);
			glVertex3f(BOUNDARY_X, 0.0f, 0.0f);
	glEnd();
	glPopMatrix();

	//glDepthMask(GL_TRUE);


	//Draw the wired cube
	glPushMatrix();
	glColor3f(1.0, 1.0, 1.0);
	glTranslatef(BOUNDARY_X/2,BOUNDARY_Y/2,BOUNDARY_Z/2);
	glutWireCube(BOUNDARY_X);
	glPopMatrix();

	#ifdef USE_VBO
		//Render from the vbo
		glBindBuffer(GL_ARRAY_BUFFER, vBuffer);
		glVertexPointer(4, GL_FLOAT, 0, 0);

		glEnableClientState(GL_VERTEX_ARRAY);
		glColor3f(1.0, 0.0, 0.0);
		glDrawArrays(GL_POINTS, 0, NUMBER_OF_PARTICLES);
		glDisableClientState(GL_VERTEX_ARRAY);
	#else
		// Draw the particles
		for(int i=0; i<NUMBER_OF_PARTICLES; i++)
		{
			glPushMatrix();
			glColor3f(colorArray_h[i].x,colorArray_h[i].y,colorArray_h[i].z);
			glTranslatef(particleArray_h[i].position.x,particleArray_h[i].position.y,particleArray_h[i].position.z );
			glutSolidSphere(PARTICLE_RADIUS,20,20);
			glPopMatrix();

		}
	#endif

	//glDisable ( GL_LIGHTING ) ;

    glutSwapBuffers();
    glutPostRedisplay();



	// Stop CUDA timer
	cutStopTimer(drawTimer);
	float milliseconds = cutGetTimerValue(drawTimer);
	cutDeleteTimer(drawTimer);

	#ifdef SEPARATE_GL
		printf("draw time %0f ms\n", milliseconds);
	#else
		printf(", draw time %0f ms\n", milliseconds);
	#endif

	#ifndef USE_VBO
		requestMemCopy = true;
	#endif
}

// Keyboard
// Keyboard events handler.
void keyboard( unsigned char key, int /*x*/, int /*y*/)
{
	 switch( key)
	 {
	    case( 27) :
	    	continueRunning = false;
	        exit(0);
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

// Mouse
// Mouse events handler.
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

// Motion
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

// Run CUDA
// Execute one time step in the simulation.
void runCuda()
{
	#ifndef NO_DISPLAY
		#ifdef USE_VBO
			// Map OpenGL buffer object for writing from CUDA
			float4 *dptr;
			cudaGLMapBufferObject( (void**)&dptr, vBuffer);
		#endif
	#endif

    // Prepare kernel dimensions
	#ifdef THREAD_PER_CELL_COLLISIONS
		dim3 cellBlock(CELL_KERNEL_BLOCK_WIDTH, CELL_KERNEL_BLOCK_HEIGHT, CELL_KERNEL_BLOCK_DEPTH);
		dim3 cellGrid(((CELL_KERNEL_BLOCK_WIDTH + 2) * CELLS_IN_X / CELL_KERNEL_BLOCK_WIDTH), ((CELL_KERNEL_BLOCK_HEIGHT + 2) * CELLS_IN_Y / CELL_KERNEL_BLOCK_HEIGHT) * ((CELL_KERNEL_BLOCK_DEPTH + 2) * CELLS_IN_Z / CELL_KERNEL_BLOCK_DEPTH), 1);
	#endif
	dim3 particleBlock(PARTICLES_PER_BLOCK, 1, 1);
	dim3 particleGrid(ceil(NUMBER_OF_PARTICLES / (double)particleBlock.x), 1, 1);

	// Start CUDA timer
	cutCreateTimer(&timer);
    cutStartTimer(timer);

    // Execute kernels
	#ifdef THREAD_PER_CELL_COLLISIONS
		cellKernel<<< cellGrid, cellBlock>>>(particleArray_d, cellArray_d, deltaTime);
	#endif
	#ifndef NO_DISPLAY
		#ifdef USE_VBO
			particleKernel<<< particleGrid, particleBlock>>>(dptr, particleArray_d, cellArray_d, deltaTime);
		#else
			particleKernel<<< particleGrid, particleBlock>>>(particleArray_d, cellArray_d, deltaTime);
		#endif
	#else
		particleKernel<<< particleGrid, particleBlock>>>(particleArray_d, cellArray_d, deltaTime);
	#endif

	#ifndef NO_DISPLAY
		#ifndef USE_VBO
			if(requestMemCopy)
			{
				requestMemCopy = false;
				copyParticlesFromDeviceToHost();
				readyToDraw = true;
			}
		#endif
	#else
		copyParticlesFromDeviceToHost();
		printf("Particle 0    x:%0f y:%0f z:%0f \n", particleArray_h[0].position.x, particleArray_h[0].position.y, particleArray_h[0].position.z);
	#endif

	// Stop CUDA timer
    cutStopTimer(timer);
	float milliseconds = cutGetTimerValue(timer);
	cutDeleteTimer(timer);

	// Increment iteration counter
	iterations++;

	// Output results
	#ifndef NO_DISPLAY
		#ifdef SEPARATE_GL
			printf("step %d, %d particles, time %0f ms\n", iterations, NUMBER_OF_PARTICLES, milliseconds);
		#else
			printf("step %d, %d particles, time %0f ms", iterations, NUMBER_OF_PARTICLES, milliseconds);
		#endif
	#else
		printf("step %d, %d particles, time %0f ms\n", iterations, NUMBER_OF_PARTICLES, milliseconds);
	#endif

	#ifndef NO_DISPLAY
		#ifdef USE_VBO
			// Unmap VBO from CUDA
			cudaGLUnmapBufferObject(vBuffer);
		#endif
	#endif
}

// Begin the simulation
void runSimulation( int argc, char** argv)
{
	// Create CUDA context
	CUT_DEVICE_INIT(argc, argv);

	#ifndef NO_DISPLAY
		// Create OpenGL context
		glutInit( &argc, argv);
		glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE);
		glutInitWindowSize( WINDOW_WIDTH, WINDOW_HEIGHT);
		glutCreateWindow( "Particle Simulation with CUDA");

		// Initialize OpenGL
		if( CUTFalse == initGL()) {
			return;
		}

		// Register callbacks
		glutDisplayFunc(display);
		glutKeyboardFunc(keyboard);
		glutMouseFunc(mouse);
		glutMotionFunc(motion);
	#endif

    // Initialize simulation
    initializeCells();
    initializeParticles();

    // Copy to device
    copyParticlesFromHostToDevice();
    copyCellsFromHostToDevice();

	#ifndef NO_DISPLAY
		#ifdef USE_VBO
			// Create Vertex Buffer Object
			createVBO(&vBuffer);
		#endif
	#endif

    // Initialize counter
    iterations = 0;

    // Run the simulation once
    runCuda();

	#ifndef NO_DISPLAY
		#ifdef SEPARATE_GL
			// Start OpenGL in separate thread
			pthread_t glThread;
			int t = 1;
			pthread_create(&glThread, NULL, openGLLoop,(void *)t);

			// Begin the simulation
			while(continueRunning)
			{
				runCuda();
			}
		#else
			// Start rendering mainloop
			glutMainLoop();
		#endif
	#else
			while(true)
			{
				runCuda();
			}
	#endif

}

// Program main
int main( int argc, char** argv)
{
	runSimulation( argc, argv);
    CUT_EXIT(argc, argv);
}
