
typedef struct
{
	float3 position;
	float3 velocity;
	float3 color;
	int cellidx;

} Particle;

typedef struct
{
	int3 coordinates;
	int counter;
	int particleidxs[4];
} Cell;



