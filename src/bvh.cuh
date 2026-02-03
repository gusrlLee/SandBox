#ifndef BVH_H
#define BVH_H

#include "structs.cuh"

void updateNodeBounds(uint32_t nodeIdx, BVH& bvh, const Scene& scene)
{
    BVHNode& node = bvh.nodes[nodeIdx]; 
    node.aabb = AABB(); 

    for (int i = 0; i < node.primCount; i++)
    {
        uint32_t leafTriIdx = bvh.triIndices[node.firstPrimIdx + i];
        const Triangle& leafTri = scene.triangles[leafTriIdx];
        
        node.aabb.grow(leafTri.v0);
        node.aabb.grow(leafTri.v1);
        node.aabb.grow(leafTri.v2);
    }
}

float evaluateSAH(const BVHNode& node, int axis, float pos, BVH& bvh, const Scene& scene)
{
    AABB leftBox, rightBox;
    int leftCount = 0, rightCount = 0;

    for (int i = 0; i < node.primCount; i++)
    {
        uint32_t triIdx = bvh.triIndices[node.firstPrimIdx + i];
        const Triangle& triangle = scene.triangles[triIdx];
        float3 c = triangle.centroid();
        float centroidVal = (axis == 0) ? c.x : ((axis == 1) ? c.y : c.z);

        if (centroidVal < pos) {
            leftCount++;
            leftBox.grow(triangle.v0); leftBox.grow(triangle.v1); leftBox.grow(triangle.v2);
        } else {
            rightCount++;
            rightBox.grow(triangle.v0); rightBox.grow(triangle.v1); rightBox.grow(triangle.v2);
        }
    }
    float cost = leftCount * leftBox.area() + rightCount * rightBox.area();
    return cost > 0 ? cost : FLT_MAX;
}

void Subdivide(uint32_t nodeIdx, BVH& bvh, const Scene& scene, int depth = 0)
{
    BVHNode& node = bvh.nodes[nodeIdx];

    int bestAxis = -1;
    float bestPos = 0;
    float bestCost = FLT_MAX;

    for (int axis = 0; axis < 3; axis++) 
    {
        for (int i = 0; i < node.primCount; i++)
        {
            uint32_t triIdx = bvh.triIndices[node.firstPrimIdx + i];
            const Triangle& triangle = scene.triangles[triIdx];
            float3 c = triangle.centroid();
            float candidatePos = (axis == 0) ? c.x : ((axis == 1) ? c.y : c.z);
            
            float cost = evaluateSAH(node, axis, candidatePos, bvh, scene);
            if (cost < bestCost) {
                bestPos = candidatePos; bestAxis = axis; bestCost = cost;
            }
        }
    }

    float parentArea = node.aabb.area();
    float parentCost = node.primCount * parentArea;
    if (bestCost >= parentCost) return; // 분할 이득 없으면 리프 노드 유지

    // 2. 파티셔닝 (QuickSort Partition)
    int i = node.firstPrimIdx;
    int j = i + node.primCount - 1;
    
    while (i <= j)
    {
        uint32_t triIdx = bvh.triIndices[i];
        const Triangle& triangle = scene.triangles[triIdx];
        float3 c = triangle.centroid();
        float centroidVal = (bestAxis == 0) ? c.x : ((bestAxis == 1) ? c.y : c.z);

        if (centroidVal < bestPos) i++;
        else std::swap(bvh.triIndices[i], bvh.triIndices[j--]);
    }

    int leftCount = i - node.firstPrimIdx;
    if (leftCount == 0 || leftCount == node.primCount) return;

    int leftChildIdx = bvh.nodeUsed++;
    int rightChildIdx = bvh.nodeUsed++;

    bvh.nodes[leftChildIdx].firstPrimIdx = node.firstPrimIdx;
    bvh.nodes[leftChildIdx].primCount = leftCount;
    
    bvh.nodes[rightChildIdx].firstPrimIdx = i;
    bvh.nodes[rightChildIdx].primCount = node.primCount - leftCount;

    node.leftChild = leftChildIdx;
    node.primCount = 0; // 내부 노드화

    updateNodeBounds(leftChildIdx, bvh, scene);
    updateNodeBounds(rightChildIdx, bvh, scene);

    Subdivide(leftChildIdx, bvh, scene, depth + 1);
    Subdivide(rightChildIdx, bvh, scene, depth + 1);
}

void BuildBVH(Scene& scene, BVH& bvh)
{
    int numTriangles = scene.triangles.size();

    bvh.allocate(numTriangles);

    for(int i = 0; i < numTriangles; i++) {
        bvh.triIndices[i] = i;
    }

    BVHNode& root = bvh.nodes[0];
    root.leftChild = 0;
    root.firstPrimIdx = 0;
    root.primCount = numTriangles;

    bvh.nodeUsed = 1; 

    updateNodeBounds(0, bvh, scene);
    
    std::cout << "[Info] Start Subdividing..." << std::endl;
    Subdivide(0, bvh, scene);
    std::cout << "[BVH] Built successfully. Nodes used: " << bvh.nodeUsed << std::endl;
}

#endif