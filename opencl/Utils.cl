#include "defs.cl"

typedef struct
{
   float3 r1; //Row 1
   float3 r2; //Row 2
   float3 r3; //Row 3
} Matrix3f;

typedef struct
{
   float4 r1; //Row 1
   float4 r2; //Row 2
   float4 r3; //Row 3
   float4 r4; //Row 4
} Matrix4f;

float4 multipyMat4Vec4(const Matrix4f mat, const float4 vec)
{
    float r1 = dot(mat.r1, vec);
    float r2 = dot(mat.r2, vec);
    float r3 = dot(mat.r3, vec);
    float r4 = dot(mat.r4, vec);
    return (float4){r1, r2, r3, r4};
}

float3 multipyMat3Vec3(const Matrix3f mat, const float3 vec)
{
    float r1 = dot(mat.r1, vec);
    float r2 = dot(mat.r2, vec);
    float r3 = dot(mat.r3, vec);
    return (float3){r1, r2, r3};
}

float3 truncateVec4ToVec3(float4 vec) {
    return (float3){vec.x, vec.y, vec.z};
}

float4 padDimVec3ToVec4(float3 vec) {
    return (float4){vec.x, vec.y, vec.z, 1.f};
}

float3 transformMatrix4ByVec3(const Matrix4f mat, const float3 vec) {
    return truncateVec4ToVec3(multipyMat4Vec4(mat, padDimVec3ToVec4(vec)));
}

float2 project(const float3 vert, const Matrix3f intrinsic) 
{
    // Project to pixel space
    float3 tmp = multipyMat3Vec3(intrinsic, vert);
    return (float2){tmp.x / tmp.z, tmp.y / tmp.z};
}

float norm(const float3 vec) {
    float x = vec.x;
    float y = vec.y;
    float z = vec.z;
    float norm = sqrt(x * x + y *y + z *z);
    return norm;
}

float3 normVec3(const float3 vec) {
    float x = vec.x;
    float y = vec.y;
    float z = vec.z;

    float norm = sqrt(x * x + y *y + z *z);
    return vec / norm;
}

Matrix3f getIntrinsic() {
    Matrix3f result;
    result.r1.x = FX;
    result.r1.y = 0.0;
    result.r1.z = CX;

    result.r2.x = 0.0;
    result.r2.y = FY;
    result.r2.z = CY;

    result.r3.x = 0.0;
    result.r3.y = 0.0;
    result.r3.z = 1.0;
    return result;
}