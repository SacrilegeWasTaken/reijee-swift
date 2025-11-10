#ifndef COMMON_METAL
#define COMMON_METAL

#include <metal_stdlib>
using namespace metal;

struct Vertex {
    float3 position;
    float4 color;
};

struct Uniforms {
    float4x4 projectionMatrix;
    float4x4 viewMatrix;
};

#endif
