
__global__ void updatePosition(Vector3D boundary, Particle* particleArray)
{
	// Get id for current particle
    unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;

    // Update particle position
    particleArray[id].position.x = particleArray[id].position.x + particleArray[id].velocity.x;
    particleArray[id].position.y = particleArray[id].position.y + particleArray[id].velocity.y;
    particleArray[id].position.z = particleArray[id].position.z + particleArray[id].velocity.z;

    // Boundary check
    if(particleArray[id].position.x < -boundary.x)
    {
    	particleArray[id].position.x = - boundary.x - particleArray[id].position.x- boundary.x;
    	particleArray[id].velocity.x = - particleArray[id].velocity.x;
    }
    else if(particleArray[id].position.x > boundary.x)
    {
    	particleArray[id].position.x = boundary.x - (particleArray[id].position.x - boundary.x);
    	particleArray[id].velocity.x = - particleArray[id].velocity.x;
    }

    if(particleArray[id].position.y < -boundary.y)
    {
    	particleArray[id].position.y = - boundary.y - particleArray[id].position.y - boundary.y;
    	particleArray[id].velocity.y = - particleArray[id].velocity.y;
    }
    else if(particleArray[id].position.y > boundary.y)
    {
    	particleArray[id].position.y = boundary.y - (particleArray[id].position.y - boundary.y);
    	particleArray[id].velocity.y = - particleArray[id].velocity.y;
    }

    if(particleArray[id].position.z < -boundary.z)
    {
    	particleArray[id].position.z = - boundary.z - particleArray[id].position.z - boundary.z;
    	particleArray[id].velocity.z = - particleArray[id].velocity.z;
    }
    else if(particleArray[id].position.z > boundary.z)
    {
    	particleArray[id].position.z = boundary.z - (particleArray[id].position.z - boundary.z);
    	particleArray[id].velocity.z = - particleArray[id].velocity.z;
    }

	__syncthreads();
}
