#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

vertex VertexOut blit_vertex(uint vertexID [[vertex_id]]) {
    float2 positions[6] = {
        float2(-1, -1), float2(1, -1), float2(-1, 1),
        float2(-1, 1), float2(1, -1), float2(1, 1)
    };
    float2 texCoords[6] = {
        float2(0, 1), float2(1, 1), float2(0, 0), 
        float2(0, 0), float2(1, 1), float2(1, 0)
    };
    
    VertexOut out;
    out.position = float4(positions[vertexID], 0, 1);
    out.texCoord = texCoords[vertexID];
    return out;
}

fragment float4 blit_fragment(VertexOut in [[stage_in]],
                                texture2d<float> tex [[texture(0)]]) {
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::nearest);
    return tex.sample(s, in.texCoord);
}