import simd

struct Uniforms {
    var projectionMatrix: float4x4
    var viewMatrix: float4x4
}

// Camera data for raytracing
struct CameraData {
    var position: SIMD3<Float>
    var forward: SIMD3<Float>
    var right: SIMD3<Float>
    var up: SIMD3<Float>
    var fov: Float
    var aspect: Float
}
