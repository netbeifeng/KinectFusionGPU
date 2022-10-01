#include "Defs.cl"
#include "Utils.cl"

float halfVolume = VOLUME_LENGTH / 2.0;

uint getIdxX(uint x, uint y, uint z) {
    // Get 1D Idx
    // Get 3D Idx - X
    return x + VOLUME_GRID_RESOLUTION * (y + VOLUME_GRID_RESOLUTION * z);
}

uint getIdxY(uint x, uint y, uint z) {
    // Get 3D Idx - Y
    return x + VOLUME_GRID_RESOLUTION * (y + VOLUME_GRID_RESOLUTION * z) + VOLUME_GRID_RESOLUTION * VOLUME_GRID_RESOLUTION * VOLUME_GRID_RESOLUTION;
}

uint getIdxZ(uint x, uint y, uint z) {
    // Get 3D Idx - Z
    return x + VOLUME_GRID_RESOLUTION * (y + VOLUME_GRID_RESOLUTION * z) + 2 * VOLUME_GRID_RESOLUTION * VOLUME_GRID_RESOLUTION * VOLUME_GRID_RESOLUTION;
}

float3 getTSDFValueByIndex(__global const float* values, float3 idx, bool isValue) {
    // Out of boundary
    if (idx.x >= VOLUME_GRID_RESOLUTION - 1 || idx.y >= VOLUME_GRID_RESOLUTION - 1 || idx.z >= VOLUME_GRID_RESOLUTION - 1) {
        if (isValue)
            return NAN; // For depth NAN
        else 
            return (float3) {0.0f, 0.0f, 0.0f}; // For Colorus Black
    }

    // Out of boundary
    if (idx.x <= 0 || idx.y <= 0 || idx.z  <= 0) {
        if (isValue)
            return NAN;
        else 
            return (float3) {0.0f, 0.0f, 0.0f};
    }

    float3 res = (float3){0.0, 0.0, 0.0};

    // Floor the idx to lowest pos
    uint xLeft = floor(idx.x);
    uint yLeft = floor(idx.y);
    uint zLeft = floor(idx.z);

    // Get the pos inside an voxel
    float interX = idx.x - xLeft;
    float interY = idx.y - yLeft;
    float interZ = idx.z - zLeft;

    // Ceil the idx to get bound
    uint xRight = ceil(idx.x);
    uint yRight = ceil(idx.y);
    uint zRight = ceil(idx.z);

    if (isValue) {
        // bottom z, left x, front y
        float vlll = values[getIdxX(xLeft, yLeft, zLeft)];
        float _vlll = vlll * (1 - interX) * (1 - interY) * (1 - interZ);
        // top z, left x, front y
        float vllr = values[getIdxX(xLeft, yLeft, zRight)];
        float _vllr = vllr * (1 - interX) * (1 - interY) * interZ;
        // bottom z, left x, back y
        float vlrl = values[getIdxX(xLeft, yRight, zLeft)];
        float _vlrl = vlrl * (1 - interX) * interY * (1 - interZ);
        // top z, left x, back y 
        float vlrr = values[getIdxX(xLeft, yRight, zRight)];
        float _vlrr = vlrr * (1 - interX) * interY * interZ;

        // bottom z, right x, front y
        float vrll = values[getIdxX(xRight, yLeft, zLeft)];
        float _vrll = vrll *  interX * (1 - interY) * (1 - interZ);
        // top z, right x, front y
        float vrlr = values[getIdxX(xRight, yLeft, zRight)];
        float _vrlr = vrlr * interX * (1 - interY) * interZ;
        // bottom z, right x, back y
        float vrrl = values[getIdxX(xRight, yRight, zLeft)];
        float _vrrl = vrrl * interX * interY * (1 - interZ);
        // top z, right x, back y
        float vrrr = values[getIdxX(xRight, yRight, zRight)];
        float _vrrr = vrrr * interX * interY * interZ;
        // Trilinear interpolation
        res.x = _vlll + _vllr + _vlrl + _vlrr \
                + _vrll + _vrlr + _vrrl + _vrrr;
    } else {
        // bottom z, left x, front y
        float vlll_x = values[getIdxX(xLeft, yLeft, zLeft)];
        float vlll_y = values[getIdxY(xLeft, yLeft, zLeft)];
        float vlll_z = values[getIdxZ(xLeft, yLeft, zLeft)];
        float3 vlll = (float3){vlll_x, vlll_y, vlll_z};
        float3 _vlll = vlll * (1 - interX) * (1 - interY) * (1 - interZ);

        // top z, left x, front y
        float vllr_x = values[getIdxX(xLeft, yLeft, zRight)];
        float vllr_y = values[getIdxY(xLeft, yLeft, zRight)];
        float vllr_z = values[getIdxZ(xLeft, yLeft, zRight)];
        float3 vllr = (float3){vllr_x, vllr_y, vllr_z};
        float3 _vllr = vllr * (1 - interX) * (1 - interY) * interZ;

        // bottom z, left x, back y
        float vlrl_x = values[getIdxX(xLeft, yRight, zLeft)];
        float vlrl_y = values[getIdxY(xLeft, yRight, zLeft)];
        float vlrl_z = values[getIdxZ(xLeft, yRight, zLeft)];
        float3 vlrl = (float3){vlrl_x, vlrl_y, vlrl_z};
        float3 _vlrl = vlrl * (1 - interX) * interY * (1 - interZ);

        // top z, left x, back y 
        float vlrr_x = values[getIdxX(xLeft, yRight, zRight)];
        float vlrr_y = values[getIdxY(xLeft, yRight, zRight)];
        float vlrr_z = values[getIdxZ(xLeft, yRight, zRight)];
        float3 vlrr = (float3){vlrr_x, vlrr_y, vlrr_z};
        float3 _vlrr = vlrr * (1 - interX) * interY * interZ;

        // bottom z, right x, front y
        float vrll_x = values[getIdxX(xRight, yLeft, zLeft)];
        float vrll_y = values[getIdxY(xRight, yLeft, zLeft)];
        float vrll_z = values[getIdxZ(xRight, yLeft, zLeft)];
        float3 vrll = (float3){vrll_x, vrll_y, vrll_z};
        float3 _vrll = vrll *  interX * (1 - interY) * (1 - interZ);

        // top z, right x, front y
        float vrlr_x = values[getIdxX(xRight, yLeft, zRight)];
        float vrlr_y = values[getIdxY(xRight, yLeft, zRight)];
        float vrlr_z = values[getIdxZ(xRight, yLeft, zRight)];
        float3 vrlr = (float3){vrlr_x, vrlr_y, vrlr_z};
        float3 _vrlr = vrlr * interX * (1 - interY) * interZ;

        // bottom z, right x, back y
        float vrrl_x = values[getIdxX(xRight, yRight, zLeft)];
        float vrrl_y = values[getIdxY(xRight, yRight, zLeft)];
        float vrrl_z = values[getIdxZ(xRight, yRight, zLeft)];
        float3 vrrl = (float3){vrrl_x, vrrl_y, vrrl_z};
        float3 _vrrl = vrrl * interX * interY * (1 - interZ);

        // top z, right x, back y
        float vrrr_x = values[getIdxX(xRight, yRight, zRight)];
        float vrrr_y = values[getIdxY(xRight, yRight, zRight)];
        float vrrr_z = values[getIdxZ(xRight, yRight, zRight)];
        float3 vrrr = (float3){vrrr_x, vrrr_y, vrrr_z};
        float3 _vrrr = vrrr * interX * interY * interZ;

        // Trilinear interpolation
        res = _vlll + _vllr + _vlrl + _vlrr \
                + _vrll + _vrlr + _vrrl + _vrrr;
    }

    return res;
}

// OpenCL Kernel Function
__kernel void RayCasterKernel(
    Matrix3f inCameraIntrinsicInv,          // 0
    Matrix4f inTransformation,              // 1
    Matrix4f inTransformationInv,           // 2
    __global const float* inTSDFValuesMap,  // 3
    __global const float* inTSDFColoursMap, // 4
    __global float* outDepthMap,            // 5
    __global float* outColourMap            // 6
    )
{
    // Work on which pixel
    uint x = get_global_id(0);
    uint y = get_global_id(1);


    // Check boundary
    uint idx = x + FRAME_WIDTH * y;
    if (idx > FRAME_WIDTH * FRAME_HEIGHT) return;

    // First set to init value
    outDepthMap[idx] = NAN;
    outColourMap[idx] = 0.f;
    outColourMap[idx + FRAME_WIDTH * FRAME_HEIGHT] = 0.f;
    outColourMap[idx + 2 * FRAME_WIDTH * FRAME_HEIGHT] = 0.f;

    // We want to cast a ray from current camera voxel to pixel voxel

    // Roadmap
    // 1. Convert Camera pos from Camera to World
    // 2. Convert from world to voxel grid index
    float4 cameraCam = (float4){0.f, 0.f, 0.f, 1.f};
    float4 cameraGlobalTmp = multipyMat4Vec4(inTransformation, cameraCam);
    float3 cameraGlobal = (float3){cameraGlobalTmp.x, cameraGlobalTmp.y, cameraGlobalTmp.z};
    
    // TSDF volume grid of camera located voxel
    float3 startVoxel = (cameraGlobal + halfVolume) / VOXEL_UNIT_LENGTH;
    
    
    // Get 3D point from 2D point
    // Roadmap
    // 1. Backproject 2D pixel to 3D camera coordinate point
    // 2. Change to World coordinate by transformation
    // 3. World coordinate to voxel grid index
    float3 targetCam = multipyMat3Vec3(inCameraIntrinsicInv, (float3){(float)(x), (float)(y), 1.f});
    float4 targetWorldTmp = multipyMat4Vec4(inTransformation, padDimVec3ToVec4(targetCam));
    float3 targetWorld = (float3){targetWorldTmp.x, targetWorldTmp.y, targetWorldTmp.z};
    
    // TSDF volume grid of target grid index
    float3 targetVoxel = (targetWorld + halfVolume) / VOXEL_UNIT_LENGTH;

    // draw a line between two voxels and normalize it as a unit forwarad length
    float3 forwardVec = normVec3(targetVoxel - startVoxel);

    // init with last TSDF value with 0, beacause last should be out of boundary
    float prevTSDFValue = 0.f;
    // current TSDF is the TSDF value in camera voxel
    float currTSDFValue = getTSDFValueByIndex(inTSDFValuesMap, startVoxel, true).x;

    // step counter
    int currentStep = 0;
    // forwarded Voxel
    float3 currentVoxel = startVoxel + currentStep * forwardVec;

    // check the status wherther the voxel idx still inside the TSDF volume
    bool stillInside = currentVoxel.x <= VOLUME_GRID_RESOLUTION \
                    && currentVoxel.y <= VOLUME_GRID_RESOLUTION \
                    && currentVoxel.z <= VOLUME_GRID_RESOLUTION;
    
    // If still in side the global TSDF volume
    while(stillInside) {
        // Update the forward voxel
        currentVoxel = startVoxel + currentStep * forwardVec;

        // recalculate status
        stillInside = currentVoxel.x <= VOLUME_GRID_RESOLUTION \
                && currentVoxel.y <= VOLUME_GRID_RESOLUTION \
                && currentVoxel.z <= VOLUME_GRID_RESOLUTION;

        // Update prev and curr
        prevTSDFValue = currTSDFValue;
        currTSDFValue = getTSDFValueByIndex(inTSDFValuesMap, currentVoxel, true).x;

        // If last one is negative, current one positive
        // a zero-crossing should happen
        if (prevTSDFValue < 0 && currTSDFValue > 0) { 
            // Now we get the zero-crossing voxel 
            // Convert it from voxel grid back to world coordinate
            float3 globalVoxel = (currentVoxel * VOXEL_UNIT_LENGTH) - VOLUME_LENGTH / 2;

            // Use inverse of transformation to get it into camera coord
            float4 camVoxel = multipyMat4Vec4(inTransformationInv, padDimVec3ToVec4(globalVoxel));
            camVoxel.w = 0.f;

            float3 camVoxel3;
            camVoxel3.x = camVoxel.x;
            camVoxel3.y = camVoxel.y;
            camVoxel3.z = camVoxel.z;
            // Know the coordinate in cam coordinate
            float normCamVoxel = norm(camVoxel3);

            // Image plane
            // The pixel should corelate to that voxel
            float3 backProjectedPixel = multipyMat3Vec3(inCameraIntrinsicInv, (float3){x, y, 1.0f});
            float normBackProjectedPixel = norm(backProjectedPixel);
            // Write to DepthMap
            outDepthMap[idx] = normCamVoxel / normBackProjectedPixel;

            // Get color by passing grid index
            float3 color = getTSDFValueByIndex(inTSDFColoursMap, currentVoxel, false);
            // Write to ColourMap
            outColourMap[idx] = color.x;
            outColourMap[idx + FRAME_WIDTH * FRAME_HEIGHT] = color.y;
            outColourMap[idx + 2 * FRAME_WIDTH * FRAME_HEIGHT] = color.z;

            // Found break searching zero-crossing
            break;
        }
        currentStep++;
    }
}