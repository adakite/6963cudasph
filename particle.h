
//Particle Struct definition
typedef struct
{
	float3 position;
	float3 velocity;
	float3 color;
	int cellidx;

} Particle;

//Cell Struct definition
typedef struct
{
	int3 coordinates;
	int counter;
	int particleidxs[4];
} Cell;

//Parameters struct definition
typedef struct
{
	int maxParticlesPerCell;
	float boundary;
	float cellSize;
	float particleRadious;
	float spring;
	float damping;
	float shear;
	float attraction;
} Parameters;



