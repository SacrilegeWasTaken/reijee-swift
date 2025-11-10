#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

struct CameraData {
    float3 position;
    float3 forward;
    float3 right;
    float3 up;
    float fov;
    float aspect;
};

kernel void raytrace(
    texture2d<float, access::write> output [[texture(0)]],
    constant CameraData& camera [[buffer(0)]],
    primitive_acceleration_structure accelStructure [[buffer(1)]],
    uint2 tid [[thread_position_in_grid]]
) {
    if (tid.x >= output.get_width() || tid.y >= output.get_height()) {
        return;
    }
    
    float2 uv = (float2(tid) + 0.5) / float2(output.get_width(), output.get_height());
    uv = uv * 2.0 - 1.0;
    uv.y = -uv.y;
    
    float tanHalfFov = tan(camera.fov * 0.5);
    float3 direction = normalize(
        camera.forward + 
        camera.right * uv.x * tanHalfFov * camera.aspect +
        camera.up * uv.y * tanHalfFov
    );
    
    ray r;
    r.origin = camera.position;
    r.direction = direction;
    r.min_distance = 0.001;
    r.max_distance = 1000.0;
    
    intersector<triangle_data> i;
    i.assume_geometry_type(geometry_type::triangle);
    intersection_result<triangle_data> intersection = i.intersect(r, accelStructure);
    
    float3 color = float3(0.1, 0.1, 0.2); // Background color
    
    if (intersection.type == intersection_type::triangle) {
        // Use barycentric coordinates for coloring
        float2 bary = intersection.triangle_barycentric_coord;
        float u = bary.x;
        float v = bary.y;
        float w = 1.0 - u - v;
        
        // Create a color based on barycentric coordinates (like vertex colors)
        float3 baseColor = float3(w, u, v);
        
        // Simple lighting based on ray direction
        // This gives a nice "depth" effect where surfaces facing the camera are brighter
        float3 lightDir = normalize(float3(0.5, 1.0, 0.3));
        float facing = abs(dot(normalize(-r.direction), lightDir));
        float shading = mix(0.4, 1.0, facing);
        
        // Distance-based fog for depth perception
        float fog = exp(-intersection.distance * 0.05);
        
        color = baseColor * shading * fog;
    }
    
    output.write(float4(color, 1.0), tid);
}
