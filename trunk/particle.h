// Type Definitions
// Particle struct definition
typedef struct
{
	float3 position;
	float3 velocity;
	float3 color;
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
