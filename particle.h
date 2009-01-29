// Type Definitions
// Particle struct definition
typedef struct
{
	float3 position;
	float3 velocity;
	#ifdef USE_VBO
		float3 color;
	#endif
	float3 collisionForce;
	unsigned int id;
} Particle;

// Particle position+force struct definition
typedef struct
{
	float3 position;
	float3 collisionForce;
	unsigned int id;
} ParticlePositionAndCollisionForce;
