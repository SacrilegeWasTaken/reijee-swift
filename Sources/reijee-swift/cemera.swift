import simd

class Camera {
    var position: SIMD3<Float> = [0, 0, 2]
    var target: SIMD3<Float> = [0, 0, 0]
    
    func viewMatrix() -> float4x4 {
        let z = simd_normalize(position - target)
        let x = simd_normalize(simd_cross([0, 1, 0], z))
        let y = simd_cross(z, x)
        
        return float4x4(
            [x.x, y.x, z.x, 0],
            [x.y, y.y, z.y, 0],
            [x.z, y.z, z.z, 0],
            [-simd_dot(x, position), -simd_dot(y, position), -simd_dot(z, position), 1]
        )
    }
    
    func projectionMatrix(fov: Float, aspect: Float, near: Float, far: Float) -> float4x4 {
        let y = 1 / tan(fov * 0.5)
        let x = y / aspect
        let z = far / (near - far)
        
        return float4x4(
            [x, 0, 0, 0],
            [0, y, 0, 0],
            [0, 0, z, -1],
            [0, 0, z * near, 0]
        )
    }
}
