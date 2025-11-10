import AppKit
import MetalKit

// Объект сцены
class SceneObject {
    fileprivate var geometry: any _2DGeomtry & _2DMovable
    fileprivate let pipelineName: String
    fileprivate var vertexBuffer: MTLBuffer
    fileprivate let device: MTLDevice

    init(geometry: any _2DGeomtry & _2DMovable, pipelineName: String, vertexBuffer: MTLBuffer, device: MTLDevice) {
        self.geometry = geometry
        self.pipelineName = pipelineName
        self.vertexBuffer = vertexBuffer
        self.device = device
    }

    func updateBuffer() {
        let vertices = geometry.vetricies()
        vertexBuffer = device.makeBuffer(
            bytes: vertices,
            length: vertices.count * MemoryLayout<Vertex>.stride,
            options: []
        )!
    }

    func getVertexBuffer() -> MTLBuffer {
        return vertexBuffer
    }

    func getPipelineName() -> String {
        return pipelineName
    }
}


extension SceneObject: _2DMovable {
    func traslate(_ dxyz: SIMD3<Float>) {
        self.geometry.traslate(dxyz)
        updateBuffer()
    }
    func rotate(_ angle: Float, axis: SIMD3<Float>) {
        self.geometry.rotate(angle, axis: axis)
        updateBuffer()
    
    }
    func scale(_ factor: Float) {
        self.geometry.scale(factor)
        updateBuffer()
    }
}

extension SceneObject: _2DGeomtry {
    func vetricies() -> [Vertex] {
        self.geometry.vetricies()
    }
    func indicies() -> [UInt16] {
        self.geometry.indicies()
    }
}


// Менеджер сцены
class Scene {
    var objects: [String: SceneObject] = [:]
    
    func addObject(objectName: String, _ object: SceneObject) {
        objects[objectName] = object
    }
    
    func removeObject(at objectName: String) {
        objects.removeValue(forKey: objectName)
    }
    
    func getObjects() -> [SceneObject] {
        return Array(objects.values)
    }

    func getAllObjectIDs() -> [String] {
        return Array(objects.keys)
    }
    
    
    func getObject(id: String) -> SceneObject? {
        return objects[id]
    }
    
    func clear() {
        objects.removeAll()
    }
}