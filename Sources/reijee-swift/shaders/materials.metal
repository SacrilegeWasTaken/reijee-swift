#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

struct Material {
    float3 albedo;
    float metallic;
    float roughness;
    float reflectivity;
};

constant Material materials[] = {
    {float3(1, 0, 0), 0.0, 0.5, 0.1},  // mat 0: красный матовый
    {float3(0.8, 0.8, 0.8), 1.0, 0.2, 0.9},  // mat 1: металл
    {float3(0, 1, 0), 0.0, 0.8, 0.0}   // mat 2: зеленый шершавый
};
