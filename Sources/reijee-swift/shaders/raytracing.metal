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
    float focus;
    uint softShadows;
    uint shadowSamples;
    float shadowRadius;
    float _padding;
};

float3 computeNormal(float3 v0, float3 v1, float3 v2) {
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    return normalize(cross(e1, e2));
}

// Better RNG using PCG hash
uint pcg_hash(uint seed) {
    uint state = seed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float random(uint seed) {
    return float(pcg_hash(seed)) / 4294967296.0;
}

float2 random2(uint seed) {
    return float2(random(seed), random(seed + 1u));
}

float3 random3(uint seed) {
    return float3(random(seed), random(seed + 1u), random(seed + 2u));
}

// Uniform sampling on unit sphere
float3 randomSpherePoint(uint seed) {
    float2 u = random2(seed);
    float z = 1.0 - 2.0 * u.x;
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = 2.0 * 3.14159265 * u.y;
    return float3(r * cos(phi), r * sin(phi), z);
}

// Cosine-weighted hemisphere sampling for better shadow quality
float3 randomCosineHemisphere(uint seed, float3 normal) {
    float2 u = random2(seed);
    float r = sqrt(u.x);
    float theta = 2.0 * 3.14159265 * u.y;
    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(max(0.0, 1.0 - u.x));
    
    float3 tangent = abs(normal.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 bitangent = normalize(cross(normal, tangent));
    tangent = cross(bitangent, normal);
    
    return normalize(tangent * x + bitangent * y + normal * z);
}

bool traceShadow(ray r, primitive_acceleration_structure accelStructure, float maxDist) {
    intersector<triangle_data> shadowIntersector;
    shadowIntersector.assume_geometry_type(geometry_type::triangle);
    shadowIntersector.accept_any_intersection(true);
    
    r.max_distance = maxDist;
    auto shadowHit = shadowIntersector.intersect(r, accelStructure);
    return shadowHit.type != intersection_type::none;
}

kernel void raytrace(
    texture2d<float, access::write> output [[texture(0)]],
    texture2d<float, access::read_write> accumulation [[texture(1)]],
    constant CameraData& camera [[buffer(0)]],
    primitive_acceleration_structure accelStructure [[buffer(1)]],
    constant Vertex* vertices [[buffer(2)]],
    constant uint16_t* indices [[buffer(3)]],
    constant uint& samplesPerPixel [[buffer(4)]],
    constant uint& lightCount [[buffer(5)]],
    constant LightData* lights [[buffer(6)]],
    constant uint& accumulatedSamples [[buffer(7)]],
    constant uint& aoEnabled [[buffer(8)]],
    constant uint& aoSamples [[buffer(9)]],
    constant float& aoRadius [[buffer(10)]],
    constant uint& giEnabled [[buffer(11)]],
    constant uint& giSamples [[buffer(12)]],
    constant uint& giBounces [[buffer(13)]],
    constant float& giIntensity [[buffer(14)]],
    constant float& giFalloff [[buffer(15)]],
    constant float& giMaxDistance [[buffer(16)]],
    constant float& giMinDistance [[buffer(17)]],
    constant float& giBias [[buffer(18)]],
    constant uint8_t* giSampleDistribution [[buffer(19)]],
    uint2 tid [[thread_position_in_grid]]
) {
    if (tid.x >= output.get_width() || tid.y >= output.get_height()) {
        return;
    }
    
    float3 accumulatedColor = float3(0.0);
    
    for (uint sample = 0; sample < samplesPerPixel; sample++) {
        uint seed = tid.x + tid.y * output.get_width() + sample * 12345u + camera.frameIndex * 67890u;
        float2 jitter = random2(seed) - 0.5;
        
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
        float2 bary = intersection.triangle_barycentric_coord;
        float u = bary.x;
        float v = bary.y;
        float w = 1.0 - u - v;
        
        uint primitiveIndex = intersection.primitive_id;
        uint i0 = indices[primitiveIndex * 3 + 0];
        uint i1 = indices[primitiveIndex * 3 + 1];
        uint i2 = indices[primitiveIndex * 3 + 2];
        
        Vertex v0 = vertices[i0];
        Vertex v1 = vertices[i1];
        Vertex v2 = vertices[i2];
        
        float3 vertexColor = (v0.color.rgb * w + v1.color.rgb * u + v2.color.rgb * v);
        float3 normal = computeNormal(v0.position, v1.position, v2.position);
        
        if (dot(normal, -r.direction) < 0.0) {
            normal = -normal;
        }
        
        float3 hitPos = r.origin + r.direction * intersection.distance;
        
        // Ambient Occlusion
        float ao = 1.0;
        
        if (aoEnabled == 1) {
            ao = 0.0;
            for (uint aoIdx = 0; aoIdx < aoSamples; aoIdx++) {
                uint aoSeed = tid.x + tid.y * output.get_width() + aoIdx * 7777u + sample * 8888u + camera.frameIndex * 9999u;
                float3 aoDir = randomCosineHemisphere(aoSeed, normal);
                
                ray aoRay;
                aoRay.origin = hitPos + normal * 0.001;
                aoRay.direction = aoDir;
                aoRay.min_distance = 0.001;
                aoRay.max_distance = aoRadius;
                
                if (!traceShadow(aoRay, accelStructure, aoRadius)) {
                    ao += 1.0;
                }
            }
            ao /= float(aoSamples);
        }
        
        float3 totalLight = float3(0.0);
        
        for (uint lightIdx = 0; lightIdx < lightCount; lightIdx++) {
            LightData light = lights[lightIdx];
            float3 lightPos = light.position;
            float3 lightColor = light.color * light.intensity;
            float3 toLight = lightPos - hitPos;
            float distToLight = length(toLight);
            float3 lightDir = toLight / distToLight;
            
            // Area light directional check
            if (light.type == 1) {
                float3 areaLightDir = normalize(light.direction);
                float directionDot = dot(-lightDir, areaLightDir);
                if (directionDot <= 0.0) {
                    continue;
                }
                // Focus falloff
                if (light.focus > 0.0) {
                    float focusFalloff = pow(max(0.0, directionDot), light.focus);
                    lightColor *= focusFalloff;
                }
            }
            
            float diffuse = max(0.0, dot(normal, lightDir));
            float attenuation = 1.0 / (distToLight * distToLight);
            
            float shadow = 1.0;
            if (diffuse > 0.0) {
                // Adaptive bias based on angle and distance
                float bias = max(0.001, 0.005 * (1.0 - abs(dot(normal, lightDir))));
                
                if (light.softShadows == 1) {
                    float shadowSum = 0.0;
                    uint samples = light.shadowSamples;
                    float totalWeight = 0.0;
                    
                    for (uint s = 0; s < samples; s++) {
                        uint shadowSeed = tid.x + tid.y * output.get_width() + s * 9876u + lightIdx * 5432u + camera.frameIndex * 11111u + sample * 22222u;
                        float3 samplePos = lightPos;
                        float weight = 1.0;
                        
                        if (light.type == 1) {
                            // Stratified sampling for area lights
                            float2 offset = random2(shadowSeed) - 0.5;
                            
                            float3 lightRight = normalize(cross(light.direction, float3(0, 1, 0)));
                            if (length(lightRight) < 0.1) {
                                lightRight = normalize(cross(light.direction, float3(1, 0, 0)));
                            }
                            float3 lightUp = cross(lightRight, light.direction);
                            samplePos += lightRight * offset.x * light.size.x * light.shadowRadius + lightUp * offset.y * light.size.y * light.shadowRadius;
                        } else if (light.type == 0) {
                            // Importance sampling for point lights - sample towards surface
                            float3 toSurface = normalize(hitPos - lightPos);
                            float3 randomDir = randomCosineHemisphere(shadowSeed, toSurface);
                            samplePos += randomDir * light.shadowRadius;
                            
                            // Weight by cosine for importance sampling
                            weight = max(0.1, dot(randomDir, toSurface));
                        }
                        
                        float3 toLightSample = samplePos - hitPos;
                        float distToSample = length(toLightSample);
                        float3 lightDirSample = normalize(toLightSample);
                        
                        // Check if sample is above surface
                        float sampleDot = dot(normal, lightDirSample);
                        if (sampleDot > 0.0) {
                            ray shadowRay;
                            shadowRay.origin = hitPos + normal * bias;
                            shadowRay.direction = lightDirSample;
                            shadowRay.min_distance = bias;
                            shadowRay.max_distance = distToSample - bias;
                            
                            float visibility = traceShadow(shadowRay, accelStructure, distToSample - bias) ? 0.0 : 1.0;
                            shadowSum += visibility * weight;
                            totalWeight += weight;
                        }
                    }
                    shadow = totalWeight > 0.0 ? shadowSum / totalWeight : 0.0;
                } else {
                    ray shadowRay;
                    shadowRay.origin = hitPos + normal * bias;
                    shadowRay.direction = lightDir;
                    shadowRay.min_distance = bias;
                    shadowRay.max_distance = distToLight - bias;
                    shadow = traceShadow(shadowRay, accelStructure, distToLight - bias) ? 0.0 : 1.0;
                }
            }
            
            totalLight += diffuse * shadow * lightColor * attenuation;
        }
        
        color = vertexColor * totalLight * ao;
        
        // Global Illumination: path tracing style with throughput and direct lighting at bounces
        if (giEnabled == 1) {
            float3 giColor = float3(0.0);
            uint pathSamples = max(1u, giSamples);
            uint giBaseSeed = pcg_hash(tid.x + tid.y * output.get_width() + sample * 33331u ^ camera.frameIndex * 0x9E3779B9u);

            for (uint s = 0; s < pathSamples; s++) {
                uint giSeed = pcg_hash(giBaseSeed + s * 0x85EBCA6Bu);
                float3 throughput = float3(1.0);
                float3 bounceOrigin = hitPos;
                float3 bounceNormal = normal;

                for (uint bounce = 0; bounce < giBounces; bounce++) {
                    // Sample next direction (cosine-weighted). For Lambertian, BRDF/pdf cancels out.
                    float3 bounceDirection = randomCosineHemisphere(giSeed, bounceNormal);
                    giSeed = pcg_hash(giSeed + 0x27D4EB2Du + bounce);

                    ray giRay;
                    giRay.origin = bounceOrigin + bounceNormal * giBias;
                    giRay.direction = bounceDirection;
                    giRay.min_distance = giMinDistance;
                    giRay.max_distance = giMaxDistance;

                    intersection_result<triangle_data> giIntersection = i.intersect(giRay, accelStructure);
                    if (giIntersection.type != intersection_type::triangle) {
                        break;
                    }

                    float2 giBary = giIntersection.triangle_barycentric_coord;
                    float giU = giBary.x;
                    float giV = giBary.y;
                    float giW = 1.0 - giU - giV;

                    uint giPrimitiveIndex = giIntersection.primitive_id;
                    uint giI0 = indices[giPrimitiveIndex * 3 + 0];
                    uint giI1 = indices[giPrimitiveIndex * 3 + 1];
                    uint giI2 = indices[giPrimitiveIndex * 3 + 2];

                    Vertex giV0 = vertices[giI0];
                    Vertex giV1 = vertices[giI1];
                    Vertex giV2 = vertices[giI2];

                    float3 giAlbedo = (giV0.color.rgb * giW + giV1.color.rgb * giU + giV2.color.rgb * giV);
                    float3 giNormal = computeNormal(giV0.position, giV1.position, giV2.position);
                    if (dot(giNormal, -giRay.direction) < 0.0) {
                        giNormal = -giNormal;
                    }

                    float3 giHitPos = giRay.origin + giRay.direction * giIntersection.distance;

                    // Next-Event Estimation: sample a single light and cast one shadow ray
                    const float INV_PI = 0.31830988618; // 1/pi
                    float3 albedoClamped = clamp(giAlbedo, 0.0, 1.0);
                    if (lightCount > 0u) {
                        uint lightIdx = pcg_hash(giSeed) % lightCount;
                        giSeed = pcg_hash(giSeed + 0xA24BAEDCu);
                        LightData light = lights[lightIdx];

                        float3 samplePos = light.position;
                        float3 lightDirSample;
                        float distToSample;
                        float3 lightColor = light.color * light.intensity;

                        bool validLightSample = true;
                        if (light.type == 1) {
                            // Sample a point on the area light rectangle
                            float2 u = random2(giSeed) - 0.5;
                            giSeed = pcg_hash(giSeed + 0x9E3779B9u);
                            float3 areaDir = normalize(light.direction);
                            float3 lightRight = normalize(cross(areaDir, float3(0, 1, 0)));
                            if (length(lightRight) < 0.1) {
                                lightRight = normalize(cross(areaDir, float3(1, 0, 0)));
                            }
                            float3 lightUp = cross(lightRight, areaDir);
                            samplePos = light.position + lightRight * (u.x * light.size.x) + lightUp * (u.y * light.size.y);

                            float3 toLightSample = samplePos - giHitPos;
                            distToSample = length(toLightSample);
                            lightDirSample = toLightSample / max(distToSample, 1e-4);

                            float dirDot = dot(-lightDirSample, areaDir);
                            if (dirDot <= 0.0) {
                                // Light faces away, skip this sample
                                validLightSample = false;
                            }
                            if (light.focus > 0.0) {
                                float focusFalloff = pow(max(0.0, dirDot), light.focus);
                                lightColor *= focusFalloff;
                            }
                        } else { // point light
                            float3 toLightSample = light.position - giHitPos;
                            distToSample = length(toLightSample);
                            lightDirSample = toLightSample / max(distToSample, 1e-4);
                        }
                        if (validLightSample) {
                            float nl = max(0.0, dot(giNormal, lightDirSample));
                            if (nl > 0.0) {
                                float attenuation = 1.0 / max(distToSample * distToSample, 1e-4);
                                float bias = max(0.001, 0.005 * (1.0 - abs(dot(giNormal, lightDirSample))));
                                ray shadowRay;
                                shadowRay.origin = giHitPos + giNormal * bias;
                                shadowRay.direction = lightDirSample;
                                shadowRay.min_distance = bias;
                                shadowRay.max_distance = max(distToSample - bias, 0.0);
                                float visibility = traceShadow(shadowRay, accelStructure, shadowRay.max_distance) ? 0.0 : 1.0;

                                float3 contrib = (albedoClamped * INV_PI) * (nl * visibility) * lightColor * attenuation;
                                // Scale for uniform light selection
                                contrib *= float(lightCount);
                                giColor += throughput * contrib;
                            }
                        }
                    }

                    // Update throughput for next bounce (Lambertian BRDF expectation under cosine sampling)
                    throughput *= (albedoClamped * INV_PI);
                    throughput = clamp(throughput, float3(0.0), float3(10.0));

                    // Russian roulette after a few bounces
                    if (bounce >= 2) {
                        float p = clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.05, 0.9);
                        if (random(giSeed) > p) {
                            break;
                        }
                        throughput /= p;
                        giSeed = pcg_hash(giSeed + 0x165667B1u);
                    }

                    bounceOrigin = giHitPos;
                    bounceNormal = giNormal;
                }
            }

            giColor = (giColor / float(pathSamples)) * giIntensity;
            color += giColor;
        }
    }
        
        accumulatedColor += color;
    }
    
    accumulatedColor /= float(samplesPerPixel);
    
    // Progressive accumulation
    float4 previousColor = accumulation.read(tid);
    float weight = 1.0 / float(accumulatedSamples);
    float3 blendedColor = mix(previousColor.rgb, accumulatedColor, weight);
    accumulation.write(float4(blendedColor, 1.0), tid);
    accumulatedColor = blendedColor;
    
    // Draw light icons
    for (uint lightIdx = 0; lightIdx < lightCount; lightIdx++) {
        LightData light = lights[lightIdx];
        
        if (light.type == 1) {
            float3 lightPos = light.position;
            float3 lightDir = normalize(light.direction);
            float2 lightSize = light.size;
            
            float3 toCamera = normalize(camera.position - lightPos);
            float facing = dot(toCamera, lightDir);
            
            if (facing > 0.0) {
                float3 lightRight = normalize(cross(lightDir, float3(0, 1, 0)));
                if (length(lightRight) < 0.1) {
                    lightRight = normalize(cross(lightDir, float3(1, 0, 0)));
                }
                float3 lightUp = cross(lightRight, lightDir);
                
                float3 corners[4];
                corners[0] = lightPos - lightRight * lightSize.x * 0.5 - lightUp * lightSize.y * 0.5;
                corners[1] = lightPos + lightRight * lightSize.x * 0.5 - lightUp * lightSize.y * 0.5;
                corners[2] = lightPos + lightRight * lightSize.x * 0.5 + lightUp * lightSize.y * 0.5;
                corners[3] = lightPos - lightRight * lightSize.x * 0.5 + lightUp * lightSize.y * 0.5;
                
                float tanHalfFov = tan(camera.fov * 0.5);
                float2 screenCorners[4];
                bool allInFront = true;
                
                for (int i = 0; i < 4; i++) {
                    float3 toCorner = corners[i] - camera.position;
                    float forwardDist = dot(toCorner, camera.forward);
                    
                    if (forwardDist <= 0.0) {
                        allInFront = false;
                        break;
                    }
                    
                    float rightDist = dot(toCorner, camera.right);
                    float upDist = dot(toCorner, camera.up);
                    
                    float screenX = (rightDist / (forwardDist * tanHalfFov * camera.aspect)) * 0.5 + 0.5;
                    float screenY = -(upDist / (forwardDist * tanHalfFov)) * 0.5 + 0.5;
                    
                    screenCorners[i] = float2(screenX * float(output.get_width()), screenY * float(output.get_height()));
                }
                
                if (allInFront) {
                    float2 pixelPos = float2(tid);
                    bool inside = true;
                    
                    for (int i = 0; i < 4; i++) {
                        float2 edge = screenCorners[(i + 1) % 4] - screenCorners[i];
                        float2 toPixel = pixelPos - screenCorners[i];
                        float cross = edge.x * toPixel.y - edge.y * toPixel.x;
                        if (cross < 0.0) {
                            inside = false;
                            break;
                        }
                    }
                    
                    if (inside) {
                        accumulatedColor = light.color * light.intensity * 0.5;
                    }
                }
            }
        } else if (light.type == 0) {
            float3 lightPos = light.position;
            float3 toLight = lightPos - camera.position;
            float distToLight = length(toLight);
            float3 lightDir = toLight / distToLight;
            
            float dotForward = dot(lightDir, camera.forward);
            if (dotForward > 0.0) {
                float tanHalfFov = tan(camera.fov * 0.5);
                float rightDist = dot(toLight, camera.right);
                float upDist = dot(toLight, camera.up);
                float forwardDist = dot(toLight, camera.forward);
                
                float screenX = (rightDist / (forwardDist * tanHalfFov * camera.aspect)) * 0.5 + 0.5;
                float screenY = -(upDist / (forwardDist * tanHalfFov)) * 0.5 + 0.5;
                
                float2 screenPos = float2(screenX * float(output.get_width()), screenY * float(output.get_height()));
                float2 pixelPos = float2(tid);
                float dist = length(screenPos - pixelPos);
                
                if (dist < 10.0) {
                    float falloff = 1.0 - (dist / 10.0);
                    accumulatedColor = mix(accumulatedColor, light.color * light.intensity * 0.5, falloff);
                }
            }
        }
    }
    
    output.write(float4(accumulatedColor, 1.0), tid);
}
