__device__ float3 GetNormal(float4 a, float4 b, float4 c)
{
	float3 x;
	x.x= b.x-a.x;
	x.y= b.y-a.y;
	x.z= b.z-a.z;

	float3 y;
	y.x= c.x-a.x;
	y.y= c.y-a.y;
	y.z= c.z-a.z;

	float3 n= make_float3(-(y.y*x.z- y.z*x.y), -(y.z*x.x- y.x*x.z), -(y.x*x.y - y.y*x.x));
	return n;
}
