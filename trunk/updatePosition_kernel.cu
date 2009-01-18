
__global__ void updatePosition(float4* spheres, float boundary, Particle* particleArray, Cell* cellArray)
{
	// Get id for current particle
    unsigned int id = blockIdx.x* blockDim.x + threadIdx.x;

    // Update particle position
    float x = particleArray[id].position.x + particleArray[id].velocity.x;
    float y = particleArray[id].position.y + particleArray[id].velocity.y;
    float z = particleArray[id].position.z + particleArray[id].velocity.z;

    // Boundary check
    if(x < 0.0)
    {
    	x = -x;
    	particleArray[id].velocity.x = -particleArray[id].velocity.x;
    }
    else if(x > boundary)
    {
    	x = boundary - (x - boundary);
    	particleArray[id].velocity.x = -particleArray[id].velocity.x;
    }

    if(y < 0.0)
    {
    	y = -y;
    	particleArray[id].velocity.y = -particleArray[id].velocity.y;
    }
    else if(y > boundary)
    {
    	y = boundary - (y - boundary);
    	particleArray[id].velocity.y = -particleArray[id].velocity.y;
    }

    if(z < 0.0)
    {
    	z = -z;
    	particleArray[id].velocity.z = -particleArray[id].velocity.z;
    }
    else if(z > boundary)
    {
    	z = boundary - (z - boundary);
    	particleArray[id].velocity.z = -particleArray[id].velocity.z;
    }

    //makeSphere(spheres, id, x - 16, y - 16, z - 16, 0.5f);

    makeSphere(spheres, id, x , y, z , 0.5f);

    particleArray[id].position.x = x;
    particleArray[id].position.y = y;
    particleArray[id].position.z = z;

	__syncthreads();
}



