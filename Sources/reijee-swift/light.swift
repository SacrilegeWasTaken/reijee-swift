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

// Dome / HDRI environment light
struct DomeLight: Light {
    // Euler rotation in radians (pitch, yaw, roll) - rotation applied to environment sampling
    private var rotation: SIMD3<Float>
    private var intensity: Float
    // Tint multiplies HDRI color; useful for subtle color shifts. Default is white (no tint).
    private var tint: SIMD3<Float>
    private var textureName: String

    init(textureName: String, intensity: Float = 1.0, rotation: SIMD3<Float> = SIMD3<Float>(0,0,0), tint: SIMD3<Float> = SIMD3<Float>(1,1,1)) {
        self.textureName = textureName
        self.intensity = intensity
        self.rotation = rotation
        self.tint = tint
    }

    // Preferred initializer (matches API used in app): rotation, intensity, tint, textureName
    init(rotation: SIMD3<Float>, intensity: Float = 1.0, tint: SIMD3<Float> = SIMD3<Float>(1,1,1), textureName: String) {
        self.textureName = textureName
        self.intensity = intensity
        self.rotation = rotation
        self.tint = tint
    }

    func toLightData() -> LightData {
        // Environment lights are represented with type dome but don't provide positional lighting via LightData.
        return LightData(
            position: SIMD3<Float>(0,0,0),
            type: LightType.dome.rawValue,
            color: tint,
            intensity: intensity,
            direction: SIMD3<Float>(0, -1, 0),
            size: SIMD2<Float>(0,0),
            radius: 0,
            focus: 0,
            softShadows: 0,
            shadowSamples: 0,
            shadowRadius: 0,
            _padding: 0
        )
    }

    func getTextureName() -> String { textureName }
    func getRotation() -> SIMD3<Float> { rotation }
    func getIntensity() -> Float { intensity }
    func getTint() -> SIMD3<Float> { tint }

    mutating func translate(_ delta: SIMD3<Float>) {}
    mutating func rotate(_ angle: Float, axis: SIMD3<Float>) {
        // Rotate around given axis by adding to euler angles (simple approach)
        if axis.x == 1 { rotation.x += angle }
        if axis.y == 1 { rotation.y += angle }
        if axis.z == 1 { rotation.z += angle }
    }
    mutating func scale(_ factor: Float) {}
}
