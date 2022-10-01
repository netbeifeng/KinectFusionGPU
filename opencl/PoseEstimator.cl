#include "Utils.cl"
#include "Defs.cl"

// OpenCL Kernel Function
// Don't use, still developing
__kernel void DataAssociationKernel(
    Matrix4f transformationInv,       // 0
    __global float* inLastVertexMap,  // 1
    __global float* inLastNormalMap,  // 2
    __global float* inCurrVertexMap,  // 3
    __global float* inCurrNormalMap,  // 4
    __global float* outCorrsp         // 5
    )
{
    // Work on 2D Image
    uint x = get_global_id(0);
	uint y = get_global_id(1);

    // return if thread is out of bounds
    if (x >= FRAME_WIDTH || y >= FRAME_HEIGHT) return;

    float3 nCurr = {
        inCurrNormalMap[x + FRAME_WIDTH * y],
        inCurrNormalMap[x + FRAME_WIDTH * y + FRAME_WIDTH * FRAME_HEIGHT],
        inCurrNormalMap[x + FRAME_WIDTH * y + 2 * FRAME_WIDTH * FRAME_HEIGHT]
    };
    
    if (!isnan(nCurr.z)) {
        float3 vPrev = {
            inLastVertexMap[x + FRAME_WIDTH * y],
            inLastVertexMap[x + FRAME_WIDTH * y + FRAME_WIDTH * FRAME_HEIGHT],
            inLastVertexMap[x + FRAME_WIDTH * y + 2 * FRAME_WIDTH * FRAME_HEIGHT]
        };

        // world
        float2 vPrev_Projected = project(vPrev, getIntrinsic());
        int px = ceil(vPrev_Projected.x);
        int py = ceil(vPrev_Projected.y);
        printf("p %d, %d \n", x, y);

        if(px >= 0 && py >= 0 && px < FRAME_WIDTH && py < FRAME_HEIGHT) {
            float3 vCurr = {
                inCurrVertexMap[x + FRAME_WIDTH * y],
                inCurrVertexMap[x + FRAME_WIDTH * y + FRAME_WIDTH * FRAME_HEIGHT],
                inCurrVertexMap[x + FRAME_WIDTH * y + 2 * FRAME_WIDTH * FRAME_HEIGHT]
            };

            if (!isnan3(vCurr)) {
                float3 diff = vCurr - vPrev;

                // printf("vPrev_ps %f, %f, %f\n", vPrev_ps.x, vPrev_ps.y, vPrev_ps.z);
                // printf("vCurr %f, %f, %f\n", vCurr.x, vCurr.y, vCurr.z);

                // Clacluate dist, if too far shouldn't associated
                float normDiff = norm(diff);

                float4 nPrev = {
                    inLastNormalMap[x + FRAME_WIDTH * y],
                    inLastNormalMap[x + FRAME_WIDTH * y + FRAME_WIDTH * FRAME_HEIGHT],
                    inLastNormalMap[x + FRAME_WIDTH * y + 2 * FRAME_WIDTH * FRAME_HEIGHT],
                    0.0f
                };

                float4 nPrev_ps = multipyMat4Vec4(transformationInv, nPrev);
                float3 nPrev_tr = {nPrev_ps.x, nPrev_ps.y, nPrev_ps.z};

                // If normal diverges too far, same
                float angleDiff = fabs((float)dot(normalize(nPrev_tr), nCurr));
                // printf("This norm diff %f,  angle diff %f \n", normDiff, angleDiff);
                if(normDiff < DIST_THRES && angleDiff > ANGLE_THRES)	// in mm
				{
					outCorrsp[x + FRAME_WIDTH * y] = px;
					outCorrsp[x + FRAME_WIDTH * y + FRAME_WIDTH * FRAME_HEIGHT] = py;
                    printf("p %d, %d \n", px, py);
				}
            }
        }
    }
    outCorrsp[x + FRAME_WIDTH * y] = NAN;
    outCorrsp[x + FRAME_WIDTH * y + FRAME_WIDTH * FRAME_HEIGHT] = NAN;
}