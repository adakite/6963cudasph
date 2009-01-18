
typedef struct
{
	float3 position;
	float3 velocity;
	float3 color;
	int cellidx;
	int next;
} Particle;

typedef struct
{
	int3 coordinates;
	int head;
} Cell;



