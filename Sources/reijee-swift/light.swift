import simd
import Metal

enum LightType: UInt32 {
    case point  = 0
    case area   = 1
    case spot   = 2
    case dome   = 3
}

struct LightData {
    var position: SIMD3<Float>
    var type: UInt32
    var color: SIMD3<Float>
    var intensity: Float
    var direction: SIMD3<Float>   // Для area/dome
    var size: SIMD2<Float>        // Для area (width, height)
    var radius: Float             // Для dome
    var _padding: Float           // Выравнивание до 16 байт
}


class Light: @unchecked Sendable {
    private let lock: RwLock<LightState> 
    
    private struct LightState {
        var position: SIMD3<Float>
        var type: LightType
        var color: SIMD3<Float>
        var intensity: Float
        var direction: SIMD3<Float>
        var size: SIMD2<Float>
        var radius: Float
    }
    
    init(type: LightType, position: SIMD3<Float>, color: SIMD3<Float>, intensity: Float) {
        let initialState = LightState(
            position: position,
            type: type,
            color: color,
            intensity: intensity,
            direction: SIMD3<Float>(0, -1, 0),
            size: SIMD2<Float>(1, 1),
            radius: 1.0
        )
        self.lock = RwLock(initialState)
    }
    
    func toLightData() -> LightData {
        lock.read { state in
            LightData(
                position: state.position,
                type: state.type.rawValue,
                color: state.color,
                intensity: state.intensity,
                direction: state.direction,
                size: state.size,
                radius: state.radius,
                _padding: 0
            )
        }
    }
    
    // Для area light
    func setDirection(_ dir: SIMD3<Float>) {
        lock.write { state in
            state.direction = normalize(dir)
        }
    }
    
    func setSize(_ size: SIMD2<Float>) {
        lock.write { state in
            state.size = size
        }
    }
    
    // Для dome light
    func setRadius(_ radius: Float) {
        lock.write { state in
            state.radius = radius
        }
    }
}
