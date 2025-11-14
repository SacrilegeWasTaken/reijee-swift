import simd
import Metal

enum LightType: UInt32 {
    case point  = 0
    case area   = 1
    case spot   = 2
    case dome   = 3
}

struct ShadowConfig {
    var samples: UInt32
    var radius: Float
}

struct LightData {
    var position: SIMD3<Float>
    var type: UInt32
    var color: SIMD3<Float>
    var intensity: Float
    var direction: SIMD3<Float>
    var size: SIMD2<Float>
    var radius: Float
    var focus: Float
    var softShadows: UInt32
    var shadowSamples: UInt32
    var shadowRadius: Float
    var _padding: Float
}

protocol Light: Transformable, Sendable {
    func toLightData() -> LightData
}

struct PointLight: Light {
    private var position: SIMD3<Float>
    private var color: SIMD3<Float>
    private var intensity: Float
    private var softShadows: Bool
    private var shadowConfig: ShadowConfig
    
    init(
        position: SIMD3<Float>, 
        color: SIMD3<Float>, 
        intensity: Float, 
        softShadows: Bool = false, 
        shadowConfig: ShadowConfig = ShadowConfig(samples: 8, radius: 0.1)) 
    {
        self.position = position
        self.color = color
        self.intensity = intensity
        self.softShadows = softShadows
        self.shadowConfig = shadowConfig
    }
    
    func toLightData() -> LightData {
        LightData(
            position: position,
            type: LightType.point.rawValue,
            color: color,
            intensity: intensity,
            direction: SIMD3<Float>(0, -1, 0),
            size: SIMD2<Float>(0, 0),
            radius: 0,
            focus: 0,
            softShadows: softShadows ? 1 : 0,
            shadowSamples: shadowConfig.samples,
            shadowRadius: shadowConfig.radius,
            _padding: 0
        )
    }
    
    mutating func translate(_ delta: SIMD3<Float>) {
        position += delta
    }
    
    mutating func rotate(_ angle: Float, axis: SIMD3<Float>) {}
    mutating func scale(_ factor: Float) {}
}

struct AreaLight: Light {
    private var position: SIMD3<Float>
    private var color: SIMD3<Float>
    private var intensity: Float
    private var direction: SIMD3<Float>
    private var size: SIMD2<Float>
    private var focus: Float
    private var softShadows: Bool
    private var shadowConfig: ShadowConfig
    
    init(
        position: SIMD3<Float>, 
        color: SIMD3<Float>, 
        intensity: Float, 
        size: SIMD2<Float> = SIMD2<Float>(1, 1), 
        direction: SIMD3<Float> = SIMD3<Float>(0, 0, -1), 
        focus: Float = 1.0, 
        softShadows: Bool = false, 
        shadowConfig: ShadowConfig = ShadowConfig(samples: 16, radius: 1.0)) 
    {
        self.position = position
        self.color = color
        self.intensity = intensity
        self.size = size
        self.direction = normalize(direction)
        self.focus = focus
        self.softShadows = softShadows
        self.shadowConfig = shadowConfig
    }
    
    func toLightData() -> LightData {
        LightData(
            position: position,
            type: LightType.area.rawValue,
            color: color,
            intensity: intensity,
            direction: direction,
            size: size,
            radius: 0,
            focus: focus,
            softShadows: softShadows ? 1 : 0,
            shadowSamples: shadowConfig.samples,
            shadowRadius: shadowConfig.radius,
            _padding: 0
        )
    }
    
    mutating func translate(_ delta: SIMD3<Float>) {
        position += delta
    }
    
    mutating func rotate(_ angle: Float, axis: SIMD3<Float>) {
        let normalized = simd_normalize(axis)
        let c = cos(angle), s = sin(angle), t = 1 - c
        let x = normalized.x, y = normalized.y, z = normalized.z
        
        let d = direction
        direction.x = (c + x*x*t)*d.x + (x*y*t - z*s)*d.y + (x*z*t + y*s)*d.z
        direction.y = (y*x*t + z*s)*d.x + (c + y*y*t)*d.y + (y*z*t - x*s)*d.z
        direction.z = (z*x*t - y*s)*d.x + (z*y*t + x*s)*d.y + (c + z*z*t)*d.z
        direction = normalize(direction)
    }
    
    mutating func scale(_ factor: Float) {
        size *= factor
    }
}
