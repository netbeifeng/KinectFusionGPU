#include "Defs.cl"
#include "Utils.cl"

// OpenCL Kernel Function
__kernel void TSDFIntegrationKernel(
    Matrix4f transformationInv,         // 0
    __global const float* inDepthMap,   // 1
    __global const float* inColourMap,  // 2
    __global float* outTSDFValues,      // 3
    __global float* outTSDFColours,     // 4
    __global float* outTSDFWeights      // 5
    )
{
    // Work on 3D grid
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint z = get_global_id(2);

    // Out of boundary
    if (x >= VOLUME_GRID_RESOLUTION 
     || y >= VOLUME_GRID_RESOLUTION 
     || z >= VOLUME_GRID_RESOLUTION) 
        return;

    // Calculate idx of the current voxel
    uint idx = x + VOLUME_GRID_RESOLUTION * (y + VOLUME_GRID_RESOLUTION * z);


    // Get the world coordinate of voxel 
    float3 voxelWorld = (float3){x, y, z} * VOXEL_UNIT_LENGTH - VOLUME_LENGTH / 2.0f;
    
    // Get the camera coordinate of voxel
    float4 voxelCam = multipyMat4Vec4(transformationInv, padDimVec3ToVec4(voxelWorld));
    float3 voxelCamTmp = (float3){voxelCam.x, voxelCam.y, voxelCam.z};
    // Closer than DENOISE_DIST just consider as noise
    if (voxelCam.z < DENOISE_DIST) return;
    // Project the voxel Cam point back to image pixel
    float3 voxelPixTmp = multipyMat3Vec3(getIntrinsic(), voxelCamTmp);
    
    float2 voxelPix = (float2){voxelPixTmp.x / voxelPixTmp.z, voxelPixTmp.y/ voxelPixTmp.z};

    // Pixel pos
    int pixelX = ceil(voxelPix.x);
    int pixelY = ceil(voxelPix.y);

    // Pixel index
    int pixelIdx = pixelX + pixelY * FRAME_WIDTH;

    // Filter out invalid index
    if (pixelX < 0 || pixelY < 0) return;
    if (pixelX >= FRAME_WIDTH - 1 || pixelY >= FRAME_HEIGHT - 1) return;

    // If there is NAN no need to integrate in into TSDF
    if (isnan(inDepthMap[pixelIdx])) return;

    // DepthMap -> Actual Depth Value Observed in Camera
    // Voxel Tmp -> Depth should have in TSDF voxel
    // If sdf positive => voxel behind observed point 
    // If sdf negative => voxel ahead of observed point 
    float sdf = voxelCamTmp.z - inDepthMap[pixelIdx];

    // If reach truncation range
    if (sdf <= TRUNC_DIST) {
        // weight
        float w;
        // calculate tsdf value
        float tsdf;
        if (sdf > 0) {
            // Weight of update
      	    w = 1.0f - sdf / TRUNC_DIST;
            tsdf = min(1.0f, sdf / TRUNC_DIST);
        } else {
      	    w = 1.0f;
            // if sdf still far away stay to -1
            // else update
            tsdf = max(-1.0f, sdf / TRUNC_DIST);
        }

        // Integration processing
        // V = (V * W_o + V_n * W_n) / (W_o + W_n)
        outTSDFValues[idx] = (outTSDFValues[idx] * outTSDFWeights[idx] + tsdf * w) / (outTSDFWeights[idx] + w);
        
        // C = (C * W_o + C_n * W_n) / (W_o + W_n);
        outTSDFColours[idx] = (outTSDFColours[idx] * outTSDFWeights[idx] + inColourMap[pixelIdx] * w) / (outTSDFWeights[idx] + w);
        
        outTSDFColours[idx + VOLUME_GRID_RESOLUTION * VOLUME_GRID_RESOLUTION * VOLUME_GRID_RESOLUTION] \
            = (outTSDFColours[idx + VOLUME_GRID_RESOLUTION * VOLUME_GRID_RESOLUTION * VOLUME_GRID_RESOLUTION] \
             * outTSDFWeights[idx] + inColourMap[pixelIdx + FRAME_WIDTH * FRAME_HEIGHT] * w) / (outTSDFWeights[idx] + w);
        
        outTSDFColours[idx + 2 * VOLUME_GRID_RESOLUTION * VOLUME_GRID_RESOLUTION * VOLUME_GRID_RESOLUTION] \
            = (outTSDFColours[idx + 2 * VOLUME_GRID_RESOLUTION * VOLUME_GRID_RESOLUTION * VOLUME_GRID_RESOLUTION] \
             * outTSDFWeights[idx] + inColourMap[pixelIdx + 2 * FRAME_WIDTH * FRAME_HEIGHT] * w) / (outTSDFWeights[idx] + w);

        // W = W_o + W_n
        outTSDFWeights[idx] = outTSDFWeights[idx] + w;
    }
}
