#include "Defs.cl"
#include "Utils.cl"

// OpenCL Kernel Function
__kernel void VertexMapKernel(
  __global float* inDepthMap, // 0
  Matrix3f inIntrinsicInv,    // 1
  __global float* outVertexMap// 2
  )
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);

    uint idx = x + FRAME_WIDTH * y;
    // Out of boundary
    if (idx > FRAME_WIDTH * FRAME_HEIGHT) return;

    float3 outVertex;

    // remove invalid depth values and set nan instead
    if (isnan(inDepthMap[idx]) || inDepthMap[idx] <= NEAR_DIST || inDepthMap[idx] > FAR_DIST) {
      inDepthMap[idx] = NAN;
      outVertex = (float3){NAN, NAN, NAN};
    } else {
      // back-project to 3d ppint 
      float3 backProjectedVertex = multipyMat3Vec3(inIntrinsicInv, (float3){x, y, 1.0f});
      // recover depth
      outVertex = backProjectedVertex * inDepthMap[idx];
    }

    outVertexMap[idx] = outVertex.x;
    outVertexMap[idx + FRAME_WIDTH * FRAME_HEIGHT] = outVertex.y;
    outVertexMap[idx + 2 * FRAME_WIDTH * FRAME_HEIGHT] = outVertex.z;
}

// OpenCL Kernel Function
__kernel void NormalMapKernel(
  __global float* inVertexDeviceBuffer, // 0
  __global float* outNormalDeviceBuffer // 1
  )
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);

    // Get Neighbor Idx
    uint left, top, right, bottom;
    if (x == 0) {
      left = 0;
    } else {
      left = x -1;
    }

    if (y ==0) {
      top = 0;
    } else {
      top = y - 1;
    }

    if (x == FRAME_WIDTH - 1) {
      right = FRAME_WIDTH - 1;
    } else {
      right = x + 1;
    }

    if (y == FRAME_HEIGHT - 1) {
      bottom = FRAME_HEIGHT - 1;
    } else {
      bottom = y + 1;
    }

    uint idx = x + FRAME_WIDTH * y;

    if (idx > FRAME_WIDTH * FRAME_HEIGHT) return;

    //      -----
    //      |top|
    // ----------------
    // |left| x |right|
    // ----------------
    //      |bot|
    //      -----

    // Left Vertex
    float3 _left = {
        inVertexDeviceBuffer[left + FRAME_WIDTH * y], 
        inVertexDeviceBuffer[left + FRAME_WIDTH * y + FRAME_WIDTH * FRAME_HEIGHT], 
        inVertexDeviceBuffer[left + FRAME_WIDTH * y + 2 * FRAME_WIDTH * FRAME_HEIGHT]
    };

    // Right Vertex
    float3 _right = {
        inVertexDeviceBuffer[right + FRAME_WIDTH * y], 
        inVertexDeviceBuffer[right + FRAME_WIDTH * y + FRAME_WIDTH * FRAME_HEIGHT], 
        inVertexDeviceBuffer[right + FRAME_WIDTH * y + 2 * FRAME_WIDTH * FRAME_HEIGHT]
    };

    // Top Vertex
    float3 _top = {
        inVertexDeviceBuffer[x + FRAME_WIDTH * top], 
        inVertexDeviceBuffer[x + FRAME_WIDTH * top + FRAME_WIDTH * FRAME_HEIGHT], 
        inVertexDeviceBuffer[x + FRAME_WIDTH * top + 2 * FRAME_WIDTH * FRAME_HEIGHT]
    };

    // Bottom Vertex
    float3 _bottom = {
        inVertexDeviceBuffer[x + FRAME_WIDTH * bottom], 
        inVertexDeviceBuffer[x + FRAME_WIDTH * bottom + FRAME_WIDTH * FRAME_HEIGHT], 
        inVertexDeviceBuffer[x + FRAME_WIDTH * bottom + 2 * FRAME_WIDTH * FRAME_HEIGHT]
    };

    // Calculate Central Differentials for horizontal direction
    float3 dx = (_right - _left) / 2.0f;

    // Calculate Central Differentials for vertical direction
    float3 dy = (_bottom - _top) / 2.0f;

    // Get all Signs of all directions
    float3 dxSign = {sign(dx.x), sign(dx.y), sign(dx.z)};
    float3 dySign = {sign(dy.x), sign(dy.y), sign(dy.z)};

    // Ensure all normals are positive 
    dx = dx * dxSign;
    dy = dy * dySign;

    // Cross product to calculate point norm
    float3 normPos = normVec3(cross(dx, dy));    

    // Write the result to OutArray
    outNormalDeviceBuffer[x + FRAME_WIDTH * y] = normPos.x;
    outNormalDeviceBuffer[x + FRAME_WIDTH * y + FRAME_HEIGHT * FRAME_HEIGHT] = normPos.y;
    outNormalDeviceBuffer[x + FRAME_WIDTH * y + 2 * FRAME_HEIGHT * FRAME_HEIGHT] = normPos.z;
}
