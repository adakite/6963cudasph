#define PI (3.141592653589793)

#define SPHERE_VERTICES_SIZE 12
#define GL_SPHERE GL_TRIANGLES

__device__ void makeSphere(float4* sphereVertices, unsigned int offsetIntoArray, float x, float y, float z, float r)
{
	unsigned int offset = offsetIntoArray * SPHERE_VERTICES_SIZE;

	// First triangle
	sphereVertices[offset + 0] = make_float4(x - r * cos(PI / 6) * cos(PI / 6), y - r * sin(PI / 6) * cos(PI / 6), z - r * sin(PI / 6) * sin(PI / 6), 1.0f);
	sphereVertices[offset + 1] = make_float4(x + r * cos(PI / 6) * cos(PI / 6), y - r * sin(PI / 6) * cos(PI / 6), z - r * sin(PI / 6) * sin(PI / 6), 1.0f);
	sphereVertices[offset + 2] = make_float4(x, y + r * sin(PI / 6) * cos(PI / 6), z, 1.0f);

	// Second triangle
	sphereVertices[offset + 3] = make_float4(x - r * cos(PI / 6) * cos(PI / 6), y - r * sin(PI / 6) * cos(PI / 6), z - r * sin(PI / 6) * sin(PI / 6), 1.0f);
	sphereVertices[offset + 4] = make_float4(x, y - r * sin(PI / 6) * cos(PI / 6), z + r * sin(PI / 6) * sin(PI / 6), 1.0f);
	sphereVertices[offset + 5] = make_float4(x, y + r * sin(PI / 6) * cos(PI / 6), z, 1.0f);

	// Third triangle
	sphereVertices[offset + 6] = make_float4(x + r * cos(PI / 6) * cos(PI / 6), y - r * sin(PI / 6) * cos(PI / 6), z - r * sin(PI / 6) * sin(PI / 6), 1.0f);
	sphereVertices[offset + 7] = make_float4(x, y - r * sin(PI / 6) * cos(PI / 6), z + r * sin(PI / 6) * sin(PI / 6), 1.0f);
	sphereVertices[offset + 8] = make_float4(x, y + r * sin(PI / 6) * cos(PI / 6), z, 1.0f);

	// Fourth triangle
	sphereVertices[offset + 9] = make_float4(x - r * cos(PI / 6) * cos(PI / 6), y - r * sin(PI / 6) * cos(PI / 6), z - r * sin(PI / 6) * sin(PI / 6), 1.0f);
	sphereVertices[offset + 10] = make_float4(x + r * cos(PI / 6) * cos(PI / 6), y - r * sin(PI / 6) * cos(PI / 6), z - r * sin(PI / 6) * sin(PI / 6), 1.0f);
	sphereVertices[offset + 11] = make_float4(x, y - r * sin(PI / 6) * cos(PI / 6), z + r * sin(PI / 6) * sin(PI / 6), 1.0f);
}
