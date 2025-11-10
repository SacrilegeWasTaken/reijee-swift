import Foundation
import simd

protocol _2DGeometry {
    func vetricies() -> [Vertex]
    func indicies() -> [UInt16]
}

protocol _2DMovable {
    mutating func traslate(_ dxyz: SIMD3<Float>)
    mutating func rotate(_ angle: Float, axis: SIMD3<Float>)
    mutating func scale(_ factor: Float)
}

struct Triangle: _2DGeometry, _2DMovable {

    private var _verticies = [
        Vertex(position: SIMD3<Float>(0.0, 0.5, 0.0), color: SIMD4<Float>(1, 0, 0, 1)),   // верх
        Vertex(position: SIMD3<Float>(-0.5, -0.5, 0.0), color: SIMD4<Float>(0, 1, 0, 1)), // лево-низ
        Vertex(position: SIMD3<Float>(0.5, -0.5, 0.0), color: SIMD4<Float>(0, 0, 1, 1))   // право-низ
    ]
    private let _indicies: [UInt16] = [0, 1, 2]

    func vetricies() -> [Vertex] {
        return _verticies
    }

    func indicies() -> [UInt16] {
        return _indicies
    }

    mutating func traslate(_ dxyz: SIMD3<Float>) {
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