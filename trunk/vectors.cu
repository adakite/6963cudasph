// Device functions
__device__ __host__ float3 distanceBetweenPoints(float3 a, float3 b)
{
	// calculate distance between points
	float3 distance;
	distance.x = b.x - a.x;
	distance.y = b.y - a.y;
	distance.z = b.z - a.z;
	return distance;
}

__device__ __host__ float vectorMagnitude(float3 v)
{
	// calculate magnitude of vector
	return sqrt(v.x * v.x + v.y*v.y + v.z*v.z);
}
