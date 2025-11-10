#include <metal_stdlib>
#include "common.metal"
using namespace metal;

struct Vertex {
    float3 position [[attribute(0)]];
    float4 color [[attribute(1)]];
};

struct VertexOut {
    float4 position [[position]];
    float4 color;
};

vertex VertexOut vertex_main(uint vertexID [[vertex_id]],
                              constant Vertex* vertices [[buffer(0)]],
                              constant Uniforms& uniforms [[buffer(1)]]) {
    VertexOut out;
    float4 worldPosition = float4(vertices[vertexID].position, 1.0);
    out.position = uniforms.projectionMatrix * uniforms.viewMatrix * worldPosition;
    out.color = vertices[vertexID].color;
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    return in.color;
}
