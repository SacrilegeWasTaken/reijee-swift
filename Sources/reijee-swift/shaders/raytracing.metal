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
    uint frameIndex;
};

struct Vertex {
    float3 position;
    float4 color;
};

struct LightData {
    float3 position;
    uint type;
    float3 color;
    float intensity;
    float3 direction;
    float2 size;
    float radius;
    float _padding;
};

// Compute normal from triangle vertices
float3 computeNormal(float3 v0, float3 v1, float3 v2) {
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    return normalize(cross(e1, e2));
}

// Simple pseudo-random function
float random(float2 st) {
    return fract(sin(dot(st, float2(12.9898, 78.233))) * 43758.5453);
}

// Trace shadow ray
bool traceShadow(ray r, primitive_acceleration_structure accelStructure, float maxDist) {
    intersector<triangle_data> shadowIntersector;
    shadowIntersector.assume_geometry_type(geometry_type::triangle);
    shadowIntersector.accept_any_intersection(true); // Stop at first hit
    
    r.max_distance = maxDist;
    auto shadowHit = shadowIntersector.intersect(r, accelStructure);
    return shadowHit.type != intersection_type::none;
}

// Simple ambient occlusion
float computeAO(float3 hitPos, float3 normal, primitive_acceleration_structure accelStructure, uint2 seed) {
    float ao = 0.0;
    const int samples = 4;
    
    for (int i = 0; i < samples; i++) {
        // Simple hemisphere sampling
        float r1 = random(float2(seed) + float(i) * 0.1);
        float r2 = random(float2(seed) + float(i) * 0.2);
        
        float phi = 2.0 * M_PI_F * r1;
        float cosTheta = sqrt(1.0 - r2);
        float sinTheta = sqrt(r2);
        
        float3 sampleDir = float3(
            cos(phi) * sinTheta,
            sin(phi) * sinTheta,
            cosTheta
        );
        
        // Orient to normal
        float3 tangent = normalize(cross(normal, float3(0.0, 1.0, 0.0)));
        if (length(tangent) < 0.1) {
            tangent = normalize(cross(normal, float3(1.0, 0.0, 0.0)));
        }
        float3 bitangent = cross(normal, tangent);
        sampleDir = normalize(tangent * sampleDir.x + bitangent * sampleDir.y + normal * sampleDir.z);
        
        ray aoRay;
        aoRay.origin = hitPos + normal * 0.001;
        aoRay.direction = sampleDir;
        aoRay.min_distance = 0.001;
        aoRay.max_distance = 1.0;
        
        if (!traceShadow(aoRay, accelStructure, 1.0)) {
            ao += 1.0;
        }
    }
    
    return ao / float(samples);
}

kernel void raytrace(
    texture2d<float, access::write> output [[texture(0)]],
    constant CameraData& camera [[buffer(0)]],
    primitive_acceleration_structure accelStructure [[buffer(1)]],
    constant Vertex* vertices [[buffer(2)]],
    constant uint16_t* indices [[buffer(3)]],
    constant uint& samplesPerPixel [[buffer(4)]],
    constant uint& lightCount [[buffer(5)]],
    constant LightData* lights [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]
) {
    if (tid.x >= output.get_width() || tid.y >= output.get_height()) {
        return;
    }
    
    float3 accumulatedColor = float3(0.0);
    
    for (uint sample = 0; sample < samplesPerPixel; sample++) {
        // Jitter для anti-aliasing
        float2 jitter = float2(
            random(float2(tid) + float(sample) * 0.1 + float(camera.frameIndex)),
            random(float2(tid) + float(sample) * 0.2 + float(camera.frameIndex))
        ) - 0.5;
        
        // Generate primary ray
        float2 uv = (float2(tid) + 0.5 + jitter) / float2(output.get_width(), output.get_height());
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
        
        float3 color = float3(0.0);
    
    if (intersection.type == intersection_type::triangle) {
        // Get barycentric coordinates
        float2 bary = intersection.triangle_barycentric_coord;
        float u = bary.x;
        float v = bary.y;
        float w = 1.0 - u - v;
        
        // Get triangle indices
        uint primitiveIndex = intersection.primitive_id;
        uint i0 = indices[primitiveIndex * 3 + 0];
        uint i1 = indices[primitiveIndex * 3 + 1];
        uint i2 = indices[primitiveIndex * 3 + 2];
        
        // Get vertices
        Vertex v0 = vertices[i0];
        Vertex v1 = vertices[i1];
        Vertex v2 = vertices[i2];
        
        // Interpolate vertex color
        float3 vertexColor = (v0.color.rgb * w + v1.color.rgb * u + v2.color.rgb * v);
        
        // Compute normal
        float3 normal = computeNormal(v0.position, v1.position, v2.position);
        
        // Flip normal if facing away from camera
        if (dot(normal, -r.direction) < 0.0) {
            normal = -normal;
        }
        
        // Hit position
        float3 hitPos = r.origin + r.direction * intersection.distance;
        
        // Lighting from all sources
        float3 totalLight = float3(0.0);
        
        for (uint lightIdx = 0; lightIdx < lightCount; lightIdx++) {
            LightData light = lights[lightIdx];
            float3 lightPos = light.position;
            float3 lightColor = light.color * light.intensity;
            float3 toLight = lightPos - hitPos;
            float distToLight = length(toLight);
            float3 lightDir = toLight / distToLight;
            
            // Diffuse
            float diffuse = max(0.0, dot(normal, lightDir));

            // Attenuation (inverse square law)
            float attenuation = 1.0 / (distToLight * distToLight);

            // Shadow
            float shadow = 1.0;
            if (diffuse > 0.0) {
                ray shadowRay;
                shadowRay.origin = hitPos + normal * 0.001;
                shadowRay.direction = lightDir;
                shadowRay.min_distance = 0.001;
                shadowRay.max_distance = distToLight - 0.001;
                shadow = traceShadow(shadowRay, accelStructure, distToLight - 0.001) ? 0.0 : 1.0;
            }

            totalLight += diffuse * shadow * lightColor * attenuation;

        }
        
        color = vertexColor * totalLight;
    }
        
        accumulatedColor += color;
    }
    
    accumulatedColor /= float(samplesPerPixel);
    
    // Draw light icons on top
    for (uint lightIdx = 0; lightIdx < lightCount; lightIdx++) {
        LightData light = lights[lightIdx];
        float3 lightPos = light.position;
        
        // Project light position to screen
        float3 toLight = lightPos - camera.position;
        float distToLight = length(toLight);
        float3 lightDir = toLight / distToLight;
        
        // Check if light is in front of camera
        float dotForward = dot(lightDir, camera.forward);
        if (dotForward > 0.0) {
            // Project to screen space
            float tanHalfFov = tan(camera.fov * 0.5);
            float3 relativePos = toLight;
            float rightDist = dot(relativePos, camera.right);
            float upDist = dot(relativePos, camera.up);
            float forwardDist = dot(relativePos, camera.forward);
            
            float screenX = (rightDist / (forwardDist * tanHalfFov * camera.aspect)) * 0.5 + 0.5;
            float screenY = -(upDist / (forwardDist * tanHalfFov)) * 0.5 + 0.5;
            
            float2 screenPos = float2(screenX * float(output.get_width()), screenY * float(output.get_height()));
            float2 pixelPos = float2(tid);
            float dist = length(screenPos - pixelPos);
            
            // Draw a small circle for the light

            if (dist < 10.0) {
                float falloff = 1.0 - (dist / 10.0);
                accumulatedColor = mix(accumulatedColor, light.color * light.intensity * 2.0, falloff);
            }

        }
    }
    
    output.write(float4(accumulatedColor, 1.0), tid);
}
