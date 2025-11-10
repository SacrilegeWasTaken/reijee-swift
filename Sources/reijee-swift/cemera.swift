import simd

class Camera: @unchecked Sendable {
    private let lock = RwLock<CameraState>(CameraState())
    
    private struct CameraState {
        var position: SIMD3<Float> = [0, 0, 5]
        var yaw: Float = Float.pi
        var pitch: Float = 0
    }

    func rotate(yaw: Float, pitch: Float) {
        lock.write { state in
            state.yaw += yaw
            state.pitch = max(-Float.pi/2 + 0.1, min(Float.pi/2 - 0.1, state.pitch + pitch))
        }
    }

    func move(_ delta: SIMD3<Float>) {
        lock.write { state in
            let forward = SIMD3<Float>(
                sin(state.yaw),
                0,
                cos(state.yaw)
            )
            let right = SIMD3<Float>(
                cos(state.yaw),
                0,
                -sin(state.yaw)
            )
            let up = SIMD3<Float>(0, 1, 0)
            
            state.position += forward * delta.z + right * delta.x + up * delta.y
        }
    }
    
    func viewMatrix() -> float4x4 {
        lock.read { state in
            let forward = SIMD3<Float>(
                cos(state.pitch) * sin(state.yaw),
                sin(state.pitch),
                cos(state.pitch) * cos(state.yaw)
            )
            let target = state.position + forward
            
            let z = simd_normalize(state.position - target)
            let x = simd_normalize(simd_cross([0, 1, 0], z))
            let y = simd_cross(z, x)
            
            return float4x4(
                [x.x, y.x, z.x, 0],
                [x.y, y.y, z.y, 0],
                [x.z, y.z, z.z, 0],
                [-simd_dot(x, state.position), -simd_dot(y, state.position), -simd_dot(z, state.position), 1]
            )
        }
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
