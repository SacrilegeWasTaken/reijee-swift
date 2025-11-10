import AppKit
import MetalKit

// Объект сцены
struct SceneObject {
    let geometry: any _2DGeomtry
    let pipelineName: String
    var vertexBuffer: MTLBuffer
}

// Менеджер сцены
class Scene {
    private var objects: [SceneObject] = []
    
    func addObject(_ object: SceneObject) {
        objects.append(object)
    }
    
    func removeObject(at index: Int) {
        objects.remove(at: index)
    }
    
    func getObjects() -> [SceneObject] {
        return objects
    }
    
    func clear() {
        objects.removeAll()
    }
}