import Foundation
import simd

protocol Geometry {
    func vertices() -> [Vertex]
    func indices() -> [UInt16]
}

protocol Transformable {
    mutating func translate(_ dxyz: SIMD3<Float>)
    mutating func rotate(_ angle: Float, axis: SIMD3<Float>)
    mutating func scale(_ factor: Float)
}

struct Triangle: Geometry, Transformable {

    private var _verticies = [
        Vertex(position: SIMD3<Float>(0.0, 0.5, 0.0), color: SIMD4<Float>(1, 0, 0, 1)),   // верх
        Vertex(position: SIMD3<Float>(-0.5, -0.5, 0.0), color: SIMD4<Float>(0, 1, 0, 1)), // лево-низ
        Vertex(position: SIMD3<Float>(0.5, -0.5, 0.0), color: SIMD4<Float>(0, 0, 1, 1))   // право-низ
    ]
    private let _indicies: [UInt16] = [ 0, 1, 2,    // CCW - conter clock-wise для вида впереди
                                        0, 2, 1 ]   // CW - clock-wise для вида сзади

    func vertices() -> [Vertex] {
        return _verticies
    }

    func indices() -> [UInt16] {
        return _indicies
    }

    mutating func translate(_ dxyz: SIMD3<Float>) {
        for index in 0..<_verticies.count {
            _verticies[index].position += dxyz
        }
    }

    mutating func rotate(_ angle: Float, axis: SIMD3<Float>) {
        // 1. Находим центр объекта
        var center = SIMD3<Float>(0, 0, 0)
        for vertex in _verticies {
            center += vertex.position
        }
        center /= Float(_verticies.count)
        
        // 2. Сдвигаем к началу координат
        for i in 0..<_verticies.count {
            _verticies[i].position -= center
        }
        
        // 3. Вращаем вокруг начала координат
        let normalizedAxis = simd_normalize(axis)
        let cos = cosf(angle)
        let sin = sinf(angle)  // ИСПРАВЛЕНО: было cosf!
        let oneMinusCos = 1.0 - cos

        let x = normalizedAxis.x
        let y = normalizedAxis.y
        let z = normalizedAxis.z

        for i in 0..<_verticies.count {
            let pos = _verticies[i].position
            
            _verticies[i].position.x = (cos + x * x * oneMinusCos) * pos.x +
                                        (x * y * oneMinusCos - z * sin) * pos.y +
                                        (x * z * oneMinusCos + y * sin) * pos.z
            
            _verticies[i].position.y = (y * x * oneMinusCos + z * sin) * pos.x +
                                        (cos + y * y * oneMinusCos) * pos.y +
                                        (y * z * oneMinusCos - x * sin) * pos.z
            
            _verticies[i].position.z = (z * x * oneMinusCos - y * sin) * pos.x +
                                        (z * y * oneMinusCos + x * sin) * pos.y +
                                        (cos + z * z * oneMinusCos) * pos.z
        }
        
        // 4. Сдвигаем обратно
        for i in 0..<_verticies.count {
            _verticies[i].position += center
        }
    }

    mutating func scale(_ factor: Float) {
        for i in 0..<_verticies.count {
            _verticies[i].position *= factor
        }
    }
}

struct Cube: Geometry, Transformable {

    private var _vertices = [
        // Front face (red)
        Vertex(position: SIMD3<Float>(-0.5, -0.5,  0.5), color: SIMD4<Float>(1, 0, 0, 1)),
        Vertex(position: SIMD3<Float>( 0.5, -0.5,  0.5), color: SIMD4<Float>(1, 0, 0, 1)),
        Vertex(position: SIMD3<Float>( 0.5,  0.5,  0.5), color: SIMD4<Float>(1, 0, 0, 1)),
        Vertex(position: SIMD3<Float>(-0.5,  0.5,  0.5), color: SIMD4<Float>(1, 0, 0, 1)),
        // Back face (green)
        Vertex(position: SIMD3<Float>(-0.5, -0.5, -0.5), color: SIMD4<Float>(0, 1, 0, 1)),
        Vertex(position: SIMD3<Float>( 0.5, -0.5, -0.5), color: SIMD4<Float>(0, 1, 0, 1)),
        Vertex(position: SIMD3<Float>( 0.5,  0.5, -0.5), color: SIMD4<Float>(0, 1, 0, 1)),
        Vertex(position: SIMD3<Float>(-0.5,  0.5, -0.5), color: SIMD4<Float>(0, 1, 0, 1))
    ]
    
    private let _indices: [UInt16] = [
        0,1,2, 0,2,3,  // front
        5,4,7, 5,7,6,  // back
        4,0,3, 4,3,7,  // left
        1,5,6, 1,6,2,  // right
        3,2,6, 3,6,7,  // top
        4,5,1, 4,1,0   // bottom
    ]
    
    func vertices() -> [Vertex] { _vertices }

    func indices() -> [UInt16] { _indices }
    
    mutating func translate(_ delta: SIMD3<Float>) {
        for i in 0..<_vertices.count {
            _vertices[i].position += delta
        }
    }
    
    mutating func rotate(_ angle: Float, axis: SIMD3<Float>) {
        let normalized = simd_normalize(axis)
        let c = cos(angle), s = sin(angle), t = 1 - c
        let x = normalized.x, y = normalized.y, z = normalized.z
        
        for i in 0..<_vertices.count {
            let p = _vertices[i].position
            _vertices[i].position.x = (c + x*x*t)*p.x + (x*y*t - z*s)*p.y + (x*z*t + y*s)*p.z
            _vertices[i].position.y = (y*x*t + z*s)*p.x + (c + y*y*t)*p.y + (y*z*t - x*s)*p.z
            _vertices[i].position.z = (z*x*t - y*s)*p.x + (z*y*t + x*s)*p.y + (c + z*z*t)*p.z
        }
    }
    
    mutating func scale(_ factor: Float) {
        for i in 0..<_vertices.count {
            _vertices[i].position *= factor
        }
    }
}