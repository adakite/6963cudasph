#define CUBE_VERTICES_SIZE 24

void makeCube(float4* cubeVertices, float width, float height, float depth)
{
	// First square
	cubeVertices[0] = make_float4(0, 0, 0, 1.0f);
	cubeVertices[1] = make_float4(width, 0, 0, 1.0f);
	cubeVertices[2] = make_float4(width, height, 0, 1.0f);
	cubeVertices[3] = make_float4(0, height, 0, 1.0f);

	// Second square
	cubeVertices[4] = make_float4(0, 0, 0, 1.0f);
	cubeVertices[5] = make_float4(0, 0, depth, 1.0f);
	cubeVertices[6] = make_float4(0, height, depth, 1.0f);
	cubeVertices[7] = make_float4(0, height, 0, 1.0f);
}
