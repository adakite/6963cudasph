#define PI (3.141592653589793)
#define X .525731112119133606
#define Z .850650808352039932

#define SPHERE_VERTICES_SIZE 60
#define GL_SPHERE GL_TRIANGLES

__device__ void makeSphere(float4* sphereVertices, unsigned int offsetIntoArray, float x, float y, float z, float r)
{
	unsigned int offset = offsetIntoArray * SPHERE_VERTICES_SIZE;
/*
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
*/

	float g= ((1+sqrtf(5))/2)*r;
	float p= 1.0f *r;

	//Blue
	float4 vertex_zero = make_float4(0.0f+x,p+y,g+z ,1.0f);
	float4 vertex_one = make_float4 (0.0f+x,p+y,-g+z ,1.0f);
	float4 vertex_two= make_float4(0.0f+x,-p+y,g+z ,1.0f);
	float4 vertex_three= make_float4(0.0f+x,-p+y,-g+z ,1.0f);

	//Light Green
	float4 vertex_four= make_float4(p+x, g+y, 0.0f+z ,1.0f);
	float4 vertex_five= make_float4(p+x, -g+y, 0.0f+z ,1.0f);
	float4 vertex_six= make_float4(-p+x, g+y, 0.0f+z ,1.0f);
	float4 vertex_seven= make_float4(-p+x, -g+y, 0.0f+z ,1.0f);

	//Dark Green
	float4 vertex_eight= make_float4(g+x,0.0f+y,p+z,1.0f);
	float4 vertex_nine= make_float4(g+x,0.0f+y,-p+z,1.0f);
	float4 vertex_ten= make_float4(-g+x,0.0f+y,p+z,1.0f);
	float4 vertex_eleven= make_float4(-g+x,0.0f+y,-p+z,1.0f);




	//Triangle_1
	sphereVertices[offset + 0]= vertex_zero;
	sphereVertices[offset + 1]= vertex_two;
	sphereVertices[offset + 2]= vertex_ten;


	//Triangle_2
	sphereVertices[offset + 3]= vertex_zero;
	sphereVertices[offset + 4]= vertex_two;
	sphereVertices[offset + 5]= vertex_eight;

	//Triangle_3
	sphereVertices[offset + 6]= vertex_zero;
	sphereVertices[offset + 7]= vertex_four;
	sphereVertices[offset + 8]= vertex_eight;

	//Triangle_4
	sphereVertices[offset + 9]= vertex_zero;
	sphereVertices[offset + 10]= vertex_six;
	sphereVertices[offset + 11]= vertex_four;

	//Triangle_5
	sphereVertices[offset + 12]= vertex_zero;
	sphereVertices[offset + 13]= vertex_six;
	sphereVertices[offset + 14]= vertex_ten;

	//Triangle_6
	sphereVertices[offset + 15]= vertex_ten;
	sphereVertices[offset + 16]= vertex_two;
	sphereVertices[offset + 17]= vertex_seven;

	//Triangle_7
	sphereVertices[offset + 18]= vertex_two;
	sphereVertices[offset + 19]= vertex_five;
	sphereVertices[offset + 20]= vertex_seven;

	//Triangle_8
	sphereVertices[offset + 21]= vertex_two;
	sphereVertices[offset + 22]= vertex_five;
	sphereVertices[offset + 23]= vertex_eight;

	//Triangle_9
	sphereVertices[offset + 24]= vertex_eight;
	sphereVertices[offset + 25]= vertex_nine;
	sphereVertices[offset + 26]= vertex_five;

	//Triangle_10
	sphereVertices[offset + 27]= vertex_eight;
	sphereVertices[offset + 28]= vertex_nine;
	sphereVertices[offset + 29]= vertex_four;

	//Triangle_11
	sphereVertices[offset + 30]= vertex_six;
	sphereVertices[offset + 31]= vertex_eleven;
	sphereVertices[offset + 32]= vertex_ten;

	//Triangle_12
	sphereVertices[offset + 33]= vertex_seven;
	sphereVertices[offset + 34]= vertex_ten;
	sphereVertices[offset + 35]= vertex_eleven;

	//Triangle_13
	sphereVertices[offset + 36]= vertex_four;
	sphereVertices[offset + 37]= vertex_nine;
	sphereVertices[offset + 38]= vertex_one;

	//Triangle_14
	sphereVertices[offset + 39]= vertex_six;
	sphereVertices[offset + 40]= vertex_four;
	sphereVertices[offset + 41]= vertex_one;

	//Triangle_15
	sphereVertices[offset + 42]= vertex_six;
	sphereVertices[offset + 43]= vertex_one;
	sphereVertices[offset + 44]= vertex_eleven;

	//Triangle_16
	sphereVertices[offset + 45]= vertex_nine;
	sphereVertices[offset + 46]= vertex_three;
	sphereVertices[offset + 47]= vertex_five;

	//Triangle_17
	sphereVertices[offset + 48]= vertex_three;
	sphereVertices[offset + 49]= vertex_seven;
	sphereVertices[offset + 50]= vertex_five;

	//Triangle_18
	sphereVertices[offset + 51]= vertex_three;
	sphereVertices[offset + 52]= vertex_eleven;
	sphereVertices[offset + 53]= vertex_seven;

	//Triangle_19
	sphereVertices[offset + 54]= vertex_eleven;
	sphereVertices[offset + 55]= vertex_one;
	sphereVertices[offset + 56]= vertex_three;

	//Triangle_20
	sphereVertices[offset + 57]= vertex_nine;
	sphereVertices[offset + 58]= vertex_one;
	sphereVertices[offset + 59]= vertex_three;


}
