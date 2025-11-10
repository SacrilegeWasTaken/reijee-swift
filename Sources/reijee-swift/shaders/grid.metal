#include <metal_stdlib>
#include "common.metal"
using namespace metal;

struct GridVertexOut {
    float4 position [[position]];
    float3 worldPos;
};

vertex GridVertexOut grid_vertex(uint vertexID [[vertex_id]],
                                  constant Vertex* vertices [[buffer(0)]],
                                  constant Uniforms& uniforms [[buffer(1)]]) {
    GridVertexOut out;
    float4 worldPosition = float4(vertices[vertexID].position, 1.0);
    out.worldPos = worldPosition.xyz;
    out.position = uniforms.projectionMatrix * uniforms.viewMatrix * worldPosition;
    return out;
}

fragment float4 grid_fragment(GridVertexOut in [[stage_in]]) {
    float gridSize = 1.0;
    float2 coord = in.worldPos.xz / gridSize;
    float2 grid = abs(fract(coord - 0.5) - 0.5) / fwidth(coord);
    float line = min(grid.x, grid.y);
    float4 color = float4(0.2, 0.2, 0.2, 1.0 - min(line, 1.0));
    
    if (color.a < 0.01) discard_fragment();
    return color;
}
