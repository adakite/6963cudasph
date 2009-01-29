#define SPHERE_VERTICES_SIZE 60
#define GL_SPHERE GL_TRIANGLES

#define PHI ((1.0+sqrtf(5.0))/2.0)

__device__ void makeSphere(float4* sphereVertices, float3* colors, float3* normals, unsigned int offsetIntoArray, float x, float y, float z, float r, float3 c)
{
	unsigned int offset = offsetIntoArray * SPHERE_VERTICES_SIZE;

	float g = r * sinf(atanf(PHI));
	float p = r * cosf(atanf(PHI));

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
	colors[offset + 0]= c;
	colors[offset + 1]= c;
	colors[offset + 2]= c;
	float3 normalone= GetNormal(vertex_zero, vertex_ten, vertex_two);
	normals[offset + 0]= normalone;
	normals[offset + 1]= normalone;
	normals[offset + 2]= normalone;


	//Triangle_2
	sphereVertices[offset + 3]= vertex_zero;
	sphereVertices[offset + 4]= vertex_two;
	sphereVertices[offset + 5]= vertex_eight;
	colors[offset + 3]= c;
	colors[offset + 4]= c;
	colors[offset + 5]= c;
	float3 normaltwo= GetNormal(vertex_eight, vertex_zero, vertex_two);
	normals[offset + 3]= normaltwo;
	normals[offset + 4]= normaltwo;
	normals[offset + 5]= normaltwo;

	//Triangle_3
	sphereVertices[offset + 6]= vertex_zero;
	sphereVertices[offset + 7]= vertex_four;
	sphereVertices[offset + 8]= vertex_eight;
	colors[offset + 6]= c;
	colors[offset + 7]= c;
	colors[offset + 8]= c;
	float3 normalthree= GetNormal(vertex_four, vertex_zero, vertex_eight);
	normals[offset + 6]= normalthree;
	normals[offset + 7]= normalthree;
	normals[offset + 8]= normalthree;

	//Triangle_4
	sphereVertices[offset + 9]= vertex_zero;
	sphereVertices[offset + 10]= vertex_six;
	sphereVertices[offset + 11]= vertex_four;
	colors[offset + 9]= c;
	colors[offset + 10]= c;
	colors[offset + 11]= c;
	float3 normalfour= GetNormal(vertex_four, vertex_six, vertex_zero);
	normals[offset + 9]= normalfour;
	normals[offset + 10]= normalfour;
	normals[offset + 11]= normalfour;

	//Triangle_5
	sphereVertices[offset + 12]= vertex_zero;
	sphereVertices[offset + 13]= vertex_six;
	sphereVertices[offset + 14]= vertex_ten;
	colors[offset + 12]= c;
	colors[offset + 13]= c;
	colors[offset + 14]= c;
	float3 normalfive= GetNormal(vertex_zero, vertex_six, vertex_ten);
	normals[offset + 12]= normalfive;
	normals[offset + 13]= normalfive;
	normals[offset + 14]= normalfive;


	//Triangle_6
	sphereVertices[offset + 15]= vertex_ten;
	sphereVertices[offset + 16]= vertex_two;
	sphereVertices[offset + 17]= vertex_seven;
	colors[offset + 15]= c;
	colors[offset + 16]= c;
	colors[offset + 17]= c;
	float3 normalsix= GetNormal(vertex_ten, vertex_seven, vertex_two);
	normals[offset + 15]= normalsix;
	normals[offset + 16]= normalsix;
	normals[offset + 17]= normalsix;

	//Triangle_7
	sphereVertices[offset + 18]= vertex_two;
	sphereVertices[offset + 19]= vertex_five;
	sphereVertices[offset + 20]= vertex_seven;
	colors[offset + 18]= c;
	colors[offset + 19]= c;
	colors[offset + 20]= c;
	float3 normalseven= GetNormal(vertex_two, vertex_five, vertex_seven);
	normals[offset + 18]= normalseven;
	normals[offset + 19]= normalseven;
	normals[offset + 20]= normalseven;

	//Triangle_8
	sphereVertices[offset + 21]= vertex_two;
	sphereVertices[offset + 22]= vertex_five;
	sphereVertices[offset + 23]= vertex_eight;
	colors[offset + 21]= c;
	colors[offset + 22]= c;
	colors[offset + 23]= c;
	float3 normaleight= GetNormal(vertex_two, vertex_five, vertex_eight);
	normals[offset + 21]= normaleight;
	normals[offset + 22]= normaleight;
	normals[offset + 23]= normaleight;

	//Triangle_9
	sphereVertices[offset + 24]= vertex_eight;
	sphereVertices[offset + 25]= vertex_nine;
	sphereVertices[offset + 26]= vertex_five;
	colors[offset + 24]= c;
	colors[offset + 25]= c;
	colors[offset + 26]= c;
	float3 normalnine= GetNormal(vertex_eight, vertex_five, vertex_nine);
	normals[offset + 27]= normalnine;
	normals[offset + 28]= normalnine;
	normals[offset + 29]= normalnine;

	//Triangle_10
	sphereVertices[offset + 27]= vertex_eight;
	sphereVertices[offset + 28]= vertex_nine;
	sphereVertices[offset + 29]= vertex_four;
	colors[offset + 27]= c;
	colors[offset + 28]= c;
	colors[offset + 29]= c;
	float3 normalten= GetNormal(vertex_eight, vertex_nine, vertex_four);
	normals[offset + 27]= normalten;
	normals[offset + 28]= normalten;
	normals[offset + 29]= normalten;

	//Triangle_11
	sphereVertices[offset + 30]= vertex_six;
	sphereVertices[offset + 31]= vertex_eleven;
	sphereVertices[offset + 32]= vertex_ten;
	colors[offset + 30]= c;
	colors[offset + 31]= c;
	colors[offset + 32]= c;
	float3 normaleleven= GetNormal(vertex_six, vertex_eleven, vertex_ten);
	normals[offset + 30]= normaleleven;
	normals[offset + 31]= normaleleven;
	normals[offset + 32]= normaleleven;

	//Triangle_12
	sphereVertices[offset + 33]= vertex_seven;
	sphereVertices[offset + 34]= vertex_ten;
	sphereVertices[offset + 35]= vertex_eleven;
	colors[offset + 33]= c;
	colors[offset + 34]= c;
	colors[offset + 35]= c;
	float3 normaltwelve= GetNormal(vertex_ten, vertex_eleven, vertex_seven);
	normals[offset + 33]= normaltwelve;
	normals[offset + 34]= normaltwelve;
	normals[offset + 35]= normaltwelve;

	//Triangle_13
	sphereVertices[offset + 36]= vertex_four;
	sphereVertices[offset + 37]= vertex_nine;
	sphereVertices[offset + 38]= vertex_one;
	colors[offset + 36]= c;
	colors[offset + 37]= c;
	colors[offset + 38]= c;
	float3 normalthirteen= GetNormal(vertex_four, vertex_nine, vertex_one);
	normals[offset + 36]= normalthirteen;
	normals[offset + 37]= normalthirteen;
	normals[offset + 38]= normalthirteen;

	//Triangle_14
	sphereVertices[offset + 39]= vertex_six;
	sphereVertices[offset + 40]= vertex_four;
	sphereVertices[offset + 41]= vertex_one;
	colors[offset + 39]= c;
	colors[offset + 40]= c;
	colors[offset + 41]= c;
	float3 normalfourteen= GetNormal(vertex_four, vertex_one, vertex_six);
	normals[offset + 30]= normalfourteen;
	normals[offset + 40]= normalfourteen;
	normals[offset + 41]= normalfourteen;


	//Triangle_15
	sphereVertices[offset + 42]= vertex_six;
	sphereVertices[offset + 43]= vertex_one;
	sphereVertices[offset + 44]= vertex_eleven;
	colors[offset + 42]= c;
	colors[offset + 43]= c;
	colors[offset + 44]= c;
	float3 normalfifteen= GetNormal(vertex_six, vertex_one, vertex_eleven);
	normals[offset + 42]= normalfifteen;
	normals[offset + 43]= normalfifteen;
	normals[offset + 44]= normalfifteen;

	//Triangle_16
	sphereVertices[offset + 45]= vertex_nine;
	sphereVertices[offset + 46]= vertex_three;
	sphereVertices[offset + 47]= vertex_five;
	colors[offset + 45]= c;
	colors[offset + 46]= c;
	colors[offset + 47]= c;
	float3 normalsixteen= GetNormal(vertex_five, vertex_three, vertex_nine);
	normals[offset + 45]= normalsixteen;
	normals[offset + 46]= normalsixteen;
	normals[offset + 47]= normalsixteen;

	//Triangle_17
	sphereVertices[offset + 48]= vertex_three;
	sphereVertices[offset + 49]= vertex_seven;
	sphereVertices[offset + 50]= vertex_five;
	colors[offset + 48]= c;
	colors[offset + 49]= c;
	colors[offset + 50]= c;
	float3 normalseventeen= GetNormal(vertex_seven, vertex_three, vertex_five);
	normals[offset + 48]= normalseventeen;
	normals[offset + 49]= normalseventeen;
	normals[offset + 50]= normalseventeen;

	//Triangle_18
	sphereVertices[offset + 51]= vertex_three;
	sphereVertices[offset + 52]= vertex_eleven;
	sphereVertices[offset + 53]= vertex_seven;
	colors[offset + 51]= c;
	colors[offset + 52]= c;
	colors[offset + 53]= c;
	float3 normaleighteen= GetNormal(vertex_seven, vertex_eleven, vertex_three);
	normals[offset + 51]= normaleighteen;
	normals[offset + 52]= normaleighteen;
	normals[offset + 53]= normaleighteen;

	//Triangle_19
	sphereVertices[offset + 54]= vertex_eleven;
	sphereVertices[offset + 55]= vertex_one;
	sphereVertices[offset + 56]= vertex_three;
	colors[offset + 54]= c;
	colors[offset + 55]= c;
	colors[offset + 56]= c;
	float3 normalnineteen= GetNormal(vertex_eleven, vertex_one, vertex_three);
	normals[offset + 54]= normalnineteen;
	normals[offset + 55]= normalnineteen;
	normals[offset + 56]= normalnineteen;

	//Triangle_20
	sphereVertices[offset + 57]= vertex_nine;
	sphereVertices[offset + 58]= vertex_one;
	sphereVertices[offset + 59]= vertex_three;
	colors[offset + 57]= c;
	colors[offset + 58]= c;
	colors[offset + 59]= c;
	float3 normaltwenty= GetNormal(vertex_nine, vertex_three, vertex_one);
	normals[offset + 57]= normaltwenty;
	normals[offset + 58]= normaltwenty;
	normals[offset + 59]= normaltwenty;

	/*
	unsigned int offset = offsetIntoArray;
	sphereVertices[offset] = make_float4(x, y, z, 1.0f);
	*/

	/*unsigned int offset = offsetIntoArray * SPHERE_VERTICES_SIZE;

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
}
