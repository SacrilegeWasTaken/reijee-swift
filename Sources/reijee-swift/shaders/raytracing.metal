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

struct PBRMaterial {
    float3 baseColor;
    float metallic;
    float roughness;
    float specular;
    float ior;
    float transmission;
    float clearcoat;
    float clearcoatRoughness;
    float3 emissiveColor;
    float emissiveIntensity;
};

// BRDF helpers (GGX + Schlick Fresnel + Disney Diffuse)
float3 fresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float distributionGGX(float3 N, float3 H, float alpha) {
    float a2 = alpha * alpha;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / max(3.14159265 * denom * denom, 1e-6);
}

float geometrySchlickGGX(float NdotV, float k) {
    return NdotV / max(NdotV * (1.0 - k) + k, 1e-6);
}

float geometrySmith(float3 N, float3 V, float3 L, float k) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx1 = geometrySchlickGGX(NdotV, k);
    float ggx2 = geometrySchlickGGX(NdotL, k);
    return ggx1 * ggx2;
}

float3 diffuseDisney(float3 baseColor, float roughness, float NdotL, float NdotV, float LdotH) {
    float FL = pow(1.0 - NdotL, 5.0);
    float FV = pow(1.0 - NdotV, 5.0);
    float Fd90 = 0.5 + 2.0 * LdotH * LdotH * roughness;
    float Fd = (1.0 + (Fd90 - 1.0) * FL) * (1.0 + (Fd90 - 1.0) * FV);
    return baseColor * Fd * 0.31830988618; // 1/pi
}

float3 computeNormal(float3 v0, float3 v1, float3 v2) {
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    return normalize(cross(e1, e2));
}

// Environment sampling helper (equirectangular -> direction) with full 3-axis rotation
constexpr sampler envSampler(coord::normalized, address::repeat, filter::linear);
inline float3 rotateDirection(float3 dir, float3 euler) {
    // euler: (pitch (X), yaw (Y), roll (Z))
    float cp = cos(euler.x); float sp = sin(euler.x);
    float cy = cos(euler.y); float sy = sin(euler.y);
    float cr = cos(euler.z); float sr = sin(euler.z);

    float3x3 Rx = float3x3(
        float3(1, 0, 0),
        float3(0, cp, -sp),
        float3(0, sp, cp)
    );
    float3x3 Ry = float3x3(
        float3(cy, 0, sy),
        float3(0, 1, 0),
        float3(-sy, 0, cy)
    );
    float3x3 Rz = float3x3(
        float3(cr, -sr, 0),
        float3(sr, cr, 0),
        float3(0, 0, 1)
    );
    // Compose rotation: roll * pitch * yaw
    float3x3 R = Rz * Rx * Ry;
    return normalize(R * dir);
}

inline float3 sampleEquirectangular(texture2d<float, access::sample> tex, float3 dir, float3 rotation) {
    float3 rdir = rotateDirection(dir, rotation);
    float u = 0.5 + atan2(rdir.z, rdir.x) / (2.0 * 3.14159265);
    float v = 0.5 - asin(clamp(rdir.y, -1.0, 1.0)) / 3.14159265;
    return tex.sample(envSampler, float2(u, v)).rgb;
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

// Stateful PCG RNG for decorrelated streams
inline uint pcg_next(thread uint &state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

inline float rand01(thread uint &state) {
    return float(pcg_next(state)) * (1.0 / 4294967296.0);
}

inline float2 rand2(thread uint &state) {
    return float2(rand01(state), rand01(state));
}

// Scrambled radical inverse (base-2 Van der Corput)
inline float radicalInverse_VdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 2^32
}

// Hammersley 2D (i/N, radicalInverse) with optional scramble
inline float2 hammersley2D(uint i, uint N, uint scramble) {
    return float2(float(i) / float(max(1u, N)), radicalInverse_VdC(i ^ scramble));
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
    texture2d<float, access::sample> envTexture [[texture(2)]],
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
    constant uint& materialCount [[buffer(20)]],
    const device PBRMaterial* materials [[buffer(21)]],
    const device uint* triMaterialIndices [[buffer(22)]],
    constant uint& envPresent [[buffer(23)]],
    constant float& envIntensity [[buffer(24)]],
    constant float3& envRotation [[buffer(25)]],
    constant float3& envTint [[buffer(26)]],
    constant uint& envRender [[buffer(27)]],
    constant uint& specularEnabled [[buffer(28)]],
    constant uint& specularBounces [[buffer(29)]],
    uint2 tid [[thread_position_in_grid]]
) {
    if (tid.x >= output.get_width() || tid.y >= output.get_height()) {
        return;
    }
    
    float3 accumulatedColor = float3(0.0);
    
    for (uint sample = 0; sample < samplesPerPixel; sample++) {
        uint seed = tid.x + tid.y * output.get_width() + sample * 12345u + camera.frameIndex * 67890u;
        thread uint rng = pcg_hash(seed);
        float2 jitter = rand2(rng) - 0.5;
        
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

        // Fetch material for this triangle
        uint triIndex = primitiveIndex; // 0-based triangle index
        uint matIndex = triMaterialIndices[triIndex];
        matIndex = matIndex < materialCount ? matIndex : 0u;
        PBRMaterial mat = materials[matIndex];

        float3 V = normalize(-r.direction);
        float3 baseColor = clamp(mat.baseColor, 0.0, 10.0);
        float metallic = clamp(mat.metallic, 0.0, 1.0);
        float roughness = clamp(mat.roughness, 0.02, 1.0);
        float specular = clamp(mat.specular, 0.0, 1.0);
        float ior = max(mat.ior, 1.0);
        float transmission = clamp(mat.transmission, 0.0, 1.0);
        float clearcoat = clamp(mat.clearcoat, 0.0, 1.0);
        float clearcoatRoughness = clamp(mat.clearcoatRoughness, 0.02, 1.0);
        float3 emissive = mat.emissiveColor * mat.emissiveIntensity;
        
        float F0_diel = pow((ior - 1.0) / (ior + 1.0), 2.0);
        float3 F0 = mix(float3(F0_diel * specular), baseColor, metallic);
        
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
        
        float3 totalLight = emissive;
        
        for (uint lightIdx = 0; lightIdx < lightCount; lightIdx++) {
            LightData light = lights[lightIdx];
            float3 lightPos = light.position;
            float3 baseLightColor = light.color * light.intensity;

            if (light.type == 1) {
                // Rectangular area light: sample the emitting rectangle instead of treating like a point
                float3 areaDir = normalize(light.direction);
                float3 lightRight = normalize(cross(areaDir, float3(0, 1, 0)));
                if (length(lightRight) < 0.1) {
                    lightRight = normalize(cross(areaDir, float3(1, 0, 0)));
                }
                float3 lightUp = cross(lightRight, areaDir);

                uint samples = max(1u, light.shadowSamples);
                float3 accum = float3(0.0);

                for (uint s = 0; s < samples; s++) {
                    // Hammersley + Cranley-Patterson rotation for low-pattern sampling on the rectangle
                    uint scramble = pcg_hash(tid.x * 0x9E3779B9u ^ tid.y * 0x85EBCA6Bu ^ camera.frameIndex * 0x27D4EB2Du ^ lightIdx * 0x165667B1u);
                    float jitter = rand01(rng);
                    float u0 = (float(s) + jitter) / float(samples);
                    float u1 = radicalInverse_VdC(s ^ scramble);
                    float2 shift = rand2(rng);
                    float2 u = fract(float2(u0, u1) + shift) - 0.5;

                    float3 samplePos = lightPos + lightRight * (u.x * light.size.x) + lightUp * (u.y * light.size.y);

                    float3 toLightSample = samplePos - hitPos;
                    float distToSample = length(toLightSample);
                    if (distToSample <= 1e-4) {
                        continue;
                    }
                    float3 wi = toLightSample / distToSample;

                    float nl = max(0.0, dot(normal, wi));
                    float nl_light = max(0.0, dot(-wi, areaDir));
                    if (nl <= 0.0 || nl_light <= 0.0) {
                        continue;
                    }

                    float3 lightColor = baseLightColor;
                    if (light.focus > 0.0) {
                        float focusFalloff = pow(nl_light, light.focus);
                        lightColor *= focusFalloff;
                    }

                    float bias = max(0.001, 0.005 * (1.0 - abs(dot(normal, wi))));
                    ray shadowRay;
                    shadowRay.origin = hitPos + normal * bias;
                    shadowRay.direction = wi;
                    shadowRay.min_distance = bias;
                    shadowRay.max_distance = distToSample - bias;
                    float visibility = traceShadow(shadowRay, accelStructure, shadowRay.max_distance) ? 0.0 : 1.0;

                    float attenuation = 1.0 / (distToSample * distToSample);
                    float3 H = normalize(V + wi);
                    float alpha = roughness * roughness;
                    float D = distributionGGX(normal, H, alpha);
                    float k = (roughness + 1.0);
                    k = (k * k) / 8.0;
                    float G = geometrySmith(normal, V, wi, k);
                    float3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
                    float3 specBRDF = (D * G * F) / max(4.0 * max(dot(normal, V), 0.0) * nl, 1e-4);
                    float3 kd = (1.0 - F) * (1.0 - metallic);
                    float3 diffBRDF = diffuseDisney(baseColor, roughness, nl, max(dot(normal, V), 0.0), max(dot(wi, H), 0.0));
                    float3 brdf = kd * diffBRDF + specBRDF;

                    // Light's cosine is nl_light; surface cosine is nl
                    float3 Li = lightColor * attenuation * nl_light * visibility;
                    accum += brdf * Li * nl;
                }

                totalLight += accum / float(samples);
            } else {
                // Point light fallback (existing behavior)
                float3 toLight = lightPos - hitPos;
                float distToLight = length(toLight);
                float3 lightDir = toLight / max(distToLight, 1e-4);

                float nl = max(0.0, dot(normal, lightDir));
                float attenuation = 1.0 / max(distToLight * distToLight, 1e-4);

                float shadow = 1.0;
                if (nl > 0.0) {
                    float bias = max(0.001, 0.005 * (1.0 - abs(dot(normal, lightDir))));

                    if (light.softShadows == 1) {
                        float shadowSum = 0.0;
                        uint samples = max(1u, light.shadowSamples);
                        float totalWeight = 0.0;

                        for (uint s = 0; s < samples; s++) {
                            uint shadowSeed = tid.x + tid.y * output.get_width() + s * 9876u + lightIdx * 5432u + camera.frameIndex * 11111u + sample * 22222u;
                            float3 samplePos = lightPos;
                            float weight = 1.0;

                            // Importance sampling for point lights - sample towards surface
                            float3 toSurface = normalize(hitPos - lightPos);
                            float3 randomDir = randomCosineHemisphere(shadowSeed, toSurface);
                            samplePos += randomDir * light.shadowRadius;

                            // Weight by cosine for importance sampling
                            weight = max(0.1, dot(randomDir, toSurface));

                            float3 toLightSample = samplePos - hitPos;
                            float distToSample = length(toLightSample);
                            float3 lightDirSample = normalize(toLightSample);

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
                        shadowRay.max_distance = max(distToLight - bias, 0.0);
                        shadow = traceShadow(shadowRay, accelStructure, shadowRay.max_distance) ? 0.0 : 1.0;
                    }
                }

                float3 H = normalize(V + lightDir);
                float alpha = roughness * roughness;
                float D = distributionGGX(normal, H, alpha);
                float k = (roughness + 1.0); k = (k * k) / 8.0;
                float G = geometrySmith(normal, V, lightDir, k);
                float3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
                float3 specBRDF = (D * G * F) / max(4.0 * max(dot(normal, V), 0.0) * nl, 1e-4);
                float3 kd = (1.0 - F) * (1.0 - metallic);
                float3 diffBRDF = diffuseDisney(baseColor, roughness, nl, max(dot(normal, V), 0.0), max(dot(lightDir, H), 0.0));
                float3 brdf = kd * diffBRDF + specBRDF;

                totalLight += brdf * (baseLightColor * attenuation) * nl * shadow;
            }
        }
        
        // Add environment lighting: when GI is disabled, approximate specular IBL from the HDRI.
        // When GI is enabled we rely on the path tracer to account for specular inter-reflections
        if (envPresent == 1 && giEnabled == 0) {
            float3 envR = sampleEquirectangular(envTexture, reflect(-V, normal), envRotation) * envIntensity * envTint;
            // For rough surfaces the env reflection should be blurred/prefiltered; approximate by scaling by (1 - roughness)
            float specularLodFactor = 1.0 - roughness;
            totalLight += envR * F0 * specularLodFactor;
        }

        // Base shading result from PBR (already includes emissive) with AO
        color = totalLight * ao;
        
        // Global Illumination: path tracing style with throughput and direct lighting at bounces
        if (giEnabled == 1) {
            float3 giColor = float3(0.0);
            const float MAX_SAMPLE_RADIANCE = 100.0;
            uint pathSamples = max(1u, giSamples);
            uint giBaseSeed = pcg_hash(tid.x + tid.y * output.get_width() + sample * 33331u ^ camera.frameIndex * 0x9E3779B9u);

            for (uint s = 0; s < pathSamples; s++) {
                uint giSeed = pcg_hash(giBaseSeed + s * 0x85EBCA6Bu);
                float3 throughput = float3(1.0);
                float3 bounceOrigin = hitPos;
                float3 bounceDir = V; // view direction from hit: V = -r.direction
                float3 bounceNormal = normal;
                uint specularCount = 0u;

                for (uint bounce = 0; bounce < giBounces; bounce++) {
                    // At the start of each bounce we have a current surface: bounceOrigin, bounceNormal
                    // Perform Next-Event Estimation (one light sample) using current surface
                    const float INV_PI = 0.31830988618; // 1/pi
                    // current material info for this surface
                    uint currentTriIndex = 0u; // not used for NEE here
                    float3 currentAlbedo = clamp((1.0 - clamp(mat.metallic, 0.0, 1.0)) * clamp(mat.baseColor, 0.0, 1.0), 0.0, 1.0);

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
                            uint scramble = pcg_hash(giSeed ^ lightIdx * 0x9E3779B9u ^ camera.frameIndex * 0x85EBCA6Bu);
                            float u0 = fract((float(s) + random(giSeed)) / max(1.0, float(pathSamples)) + random(giSeed + 17u));
                            float u1 = radicalInverse_VdC((s % max(1u, pathSamples)) ^ scramble);
                            float2 u = float2(u0, u1) - 0.5;
                            giSeed = pcg_hash(giSeed + 0x9E3779B9u);
                            float3 areaDir = normalize(light.direction);
                            float3 lightRight = normalize(cross(areaDir, float3(0, 1, 0)));
                            if (length(lightRight) < 0.1) {
                                lightRight = normalize(cross(areaDir, float3(1, 0, 0)));
                            }
                            float3 lightUp = cross(lightRight, areaDir);
                            samplePos = light.position + lightRight * (u.x * light.size.x) + lightUp * (u.y * light.size.y);

                            float3 toLightSample = samplePos - bounceOrigin;
                            distToSample = length(toLightSample);
                            lightDirSample = toLightSample / max(distToSample, 1e-4);

                            float dirDot = dot(-lightDirSample, areaDir);
                            if (dirDot <= 0.0) {
                                validLightSample = false;
                            }
                            if (light.focus > 0.0) {
                                float focusFalloff = pow(max(0.0, dirDot), light.focus);
                                lightColor *= focusFalloff;
                            }
                        } else {
                            float3 toLightSample = light.position - bounceOrigin;
                            distToSample = length(toLightSample);
                            lightDirSample = toLightSample / max(distToSample, 1e-4);
                        }
                        if (validLightSample) {
                            float nl = max(0.0, dot(bounceNormal, lightDirSample));
                            if (nl > 0.0) {
                                float attenuation = 1.0 / max(distToSample * distToSample, 1e-4);
                                float bias = max(0.001, 0.005 * (1.0 - abs(dot(bounceNormal, lightDirSample))));
                                ray shadowRay;
                                shadowRay.origin = bounceOrigin + bounceNormal * bias;
                                shadowRay.direction = lightDirSample;
                                shadowRay.min_distance = bias;
                                shadowRay.max_distance = max(distToSample - bias, 0.0);
                                float visibility = traceShadow(shadowRay, accelStructure, shadowRay.max_distance) ? 0.0 : 1.0;

                                if (light.type == 1) {
                                    float3 areaDirLocal = normalize(light.direction);
                                    float nl_light = max(0.0, dot(-lightDirSample, areaDirLocal));
                                    float area = max(1e-6, light.size.x * light.size.y);
                                    lightColor *= nl_light * area;
                                }

                                // Compute PDFs for MIS (power heuristic, beta=2)
                                float pdf_light = 1e6; // default large for point lights
                                if (light.type == 1) {
                                    // sampling the area uniformly -> pdf_area = 1/area
                                    float area = max(1e-6, light.size.x * light.size.y);
                                    float nl_light = max(0.0, dot(-lightDirSample, normalize(light.direction)));
                                    // convert area pdf to solid angle pdf
                                    pdf_light = (distToSample * distToSample) / (area * max(nl_light, 1e-6));
                                }

                                // Better BRDF sampling pdf: include a GGX specular pdf when material is glossy
                                float specularParamLocal = clamp(mat.specular, 0.0, 1.0);
                                float metallicLocal = clamp(mat.metallic, 0.0, 1.0);
                                float specularWeight = max(metallicLocal, specularParamLocal);
                                float diffuseWeight = 1.0 - specularWeight;

                                float pdf_brdf_diffuse = max(1e-6, nl * INV_PI);
                                // compute GGX specular PDF for sampling H -> L
                                float rough = clamp(mat.roughness, 0.02, 1.0);
                                float alphaSpec = rough * rough;
                                float3 Vlocal = normalize(-bounceDir);
                                float3 H_from_light = normalize(Vlocal + lightDirSample);
                                float NdotH_spec = max(dot(bounceNormal, H_from_light), 1e-6);
                                float VdotH_spec = max(dot(Vlocal, H_from_light), 1e-6);
                                float D_spec = distributionGGX(bounceNormal, H_from_light, alphaSpec);
                                float pdf_specular = (D_spec * NdotH_spec) / (4.0 * VdotH_spec + 1e-6);

                                float pdf_brdf = diffuseWeight * pdf_brdf_diffuse + specularWeight * pdf_specular + 1e-8;

                                // Power heuristic
                                float w_light = (pdf_light * pdf_light) / (pdf_light * pdf_light + pdf_brdf * pdf_brdf);

                                float3 contrib = (currentAlbedo * INV_PI) * (nl * visibility) * lightColor * attenuation;
                                contrib *= float(lightCount);
                                contrib *= w_light;

                                // Clamp per-sample contribution to avoid fireflies (roughness-aware)
                                const float MAX_SAMPLE_RADIANCE = 100.0;
                                float curRough = clamp(mat.roughness, 0.02, 1.0);
                                float clampScale = 0.2 + curRough * 0.8; // more aggressive clamp for low roughness
                                float3 maxClamp = float3(MAX_SAMPLE_RADIANCE * clampScale);
                                contrib = clamp(contrib, float3(0.0), maxClamp);

                                giColor += throughput * contrib;
                            }
                        }
                    }

                    // Sample next direction according to BRDF at current surface
                    float metallic = clamp(mat.metallic, 0.0, 1.0);
                    float specularParam = clamp(mat.specular, 0.0, 1.0);
                    float specProb = max(metallic, specularParam);

                    bool doSpecular = (specularEnabled == 1) && (specProb > 0.0);
                    float rrand = rand01(giSeed);

                    // Low-discrepancy Hammersley sample for this bounce/sample pair
                    uint hN = max(1u, giBounces * pathSamples);
                    uint hIndex = (s * giBounces + bounce) % hN;
                    uint scramble = pcg_hash(giSeed ^ 0x9E3779B9u);
                    float2 Xi = hammersley2D(hIndex, hN, scramble);
                    giSeed = pcg_hash(giSeed + 0x7ED55D16u);

                    float3 sampledDir;
                    if (doSpecular && rrand < specProb && specularCount < specularBounces) {
                        float roughness = clamp(mat.roughness, 0.02, 1.0);
                        float alpha = roughness * roughness;

                        // Importance sample GGX normal distribution using Xi (H sample)
                        float phi = 2.0 * 3.14159265 * Xi.x;
                        float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (alpha*alpha - 1.0) * Xi.y));
                        float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
                        float3 Ht = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

                        float3 up = fabs(bounceNormal.z) < 0.999 ? float3(0,0,1) : float3(1,0,0);
                        float3 tangent = normalize(cross(up, bounceNormal));
                        float3 bitangent = cross(bounceNormal, tangent);
                        float3 H = normalize(tangent * Ht.x + bitangent * Ht.y + bounceNormal * Ht.z);

                        float3 Vlocal = normalize(bounceDir);
                        sampledDir = normalize(reflect(-Vlocal, H));

                        // update throughput using importance-sampled microfacet formula (BRDF * cos / pdf)
                        float NdotH = max(dot(bounceNormal, H), 1e-6);
                        float VdotH = max(dot(Vlocal, H), 1e-6);
                        float NdotV = max(dot(bounceNormal, Vlocal), 1e-6);
                        float3 F0_local = mix(float3(pow((mat.ior - 1.0)/(mat.ior + 1.0), 2.0) * mat.specular), mat.baseColor, mat.metallic);
                        float3 F = fresnelSchlick(max(dot(H, Vlocal), 0.0), F0_local);
                        float G = geometrySmith(bounceNormal, Vlocal, sampledDir, (mat.roughness + 1.0));
                        float3 specContrib = F * G * VdotH / (NdotV * NdotH);

                        // account for discrete branch probability (we chose specular with probability specProb)
                        float invSpecProb = 1.0 / max(specProb, 1e-6);
                        throughput *= specContrib * invSpecProb;
                        specularCount += 1u;
                    } else {
                        // cosine-weighted hemisphere sampling using Hammersley Xi
                        float r = sqrt(Xi.x);
                        float theta = 2.0 * 3.14159265 * Xi.y;
                        float x = r * cos(theta);
                        float y = r * sin(theta);
                        float z = sqrt(max(0.0, 1.0 - Xi.x));

                        float3 up = fabs(bounceNormal.z) < 0.999 ? float3(0,0,1) : float3(1,0,0);
                        float3 tangent = normalize(cross(up, bounceNormal));
                        float3 bitangent = cross(bounceNormal, tangent);
                        sampledDir = normalize(tangent * x + bitangent * y + bounceNormal * z);

                        // account for discrete branch probability (we chose diffuse with 1-specProb)
                        float invDiffProb = 1.0 / max(1.0 - specProb, 1e-6);
                        throughput *= currentAlbedo * invDiffProb;
                    }

                    // Trace the sampled direction
                    ray giRay;
                    giRay.origin = bounceOrigin + bounceNormal * giBias;
                    giRay.direction = sampledDir;
                    giRay.min_distance = giMinDistance;
                    giRay.max_distance = giMaxDistance;

                    intersection_result<triangle_data> giIntersection = i.intersect(giRay, accelStructure);
                    if (giIntersection.type != intersection_type::triangle) {
                        // Miss: add environment contribution scaled by throughput
                        if (envPresent == 1) {
                            float3 envSample = sampleEquirectangular(envTexture, giRay.direction, envRotation) * envIntensity * envTint;
                            // Roughness-aware clamping for env contribution
                            float curRoughEnv = clamp(mat.roughness, 0.02, 1.0);
                            float envClampScale = 0.2 + curRoughEnv * 0.8;
                            float3 envMaxClamp = float3(MAX_SAMPLE_RADIANCE * envClampScale);
                            float3 envContrib = clamp(throughput * envSample, float3(0.0), envMaxClamp);
                            giColor += envContrib;
                        }
                        break;
                    }

                    // Update hit info for next bounce
                    float2 giBary = giIntersection.triangle_barycentric_coord;
                    uint giPrimitiveIndex = giIntersection.primitive_id;
                    uint giI0 = indices[giPrimitiveIndex * 3 + 0];
                    uint giI1 = indices[giPrimitiveIndex * 3 + 1];
                    uint giI2 = indices[giPrimitiveIndex * 3 + 2];

                    uint giMatIndex = triMaterialIndices[giPrimitiveIndex];
                    giMatIndex = giMatIndex < materialCount ? giMatIndex : 0u;
                    PBRMaterial giMat = materials[giMatIndex];
                    float3 giAlbedo = (1.0 - clamp(giMat.metallic, 0.0, 1.0)) * clamp(giMat.baseColor, 0.0, 1.0);
                    Vertex giV0 = vertices[giI0];
                    Vertex giV1 = vertices[giI1];
                    Vertex giV2 = vertices[giI2];
                    float3 giNormal = computeNormal(giV0.position, giV1.position, giV2.position);
                    if (dot(giNormal, -giRay.direction) < 0.0) {
                        giNormal = -giNormal;
                    }

                    float3 giHitPos = giRay.origin + giRay.direction * giIntersection.distance;

                    // Prepare for next bounce
                    bounceOrigin = giHitPos;
                    bounceNormal = giNormal;
                    // Update the material used for next NEE and sampling
                    mat = giMat;

                    // Russian roulette
                    if (bounce >= 2) {
                        float p = clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.05, 0.95);
                        if (random(giSeed) > p) { break; }
                        throughput /= p;
                        giSeed = pcg_hash(giSeed + 0x165667B1u);
                    }
                }
            }

            giColor = (giColor / float(pathSamples)) * giIntensity;
            color += giColor;
        }
        } else {
            // Miss: render environment map only if present AND render flag is set
            if (envPresent == 1 && envRender == 1) {
                color = sampleEquirectangular(envTexture, r.direction, envRotation) * envIntensity * envTint;
            } else {
                color = float3(0.0);
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
