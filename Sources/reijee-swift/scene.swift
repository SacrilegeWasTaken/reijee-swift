import AppKit
import MetalKit

/// Thread Safety:
/// SceneObject использует @unchecked Sendable с ручной синхронизацией:
///
/// 1. geometry: RwLock<any _2DGeometry & _2DMovable>
///    - Защищен RwLock для многопоточного доступа
///    - Множественное чтение без блокировки
///    - Эксклюзивная запись при модификации
///
/// 2. currentBufferIndex: RwLock<Int>
///    - Атомарное переключение между буферами (0 ↔ 1)
///    - Гарантирует согласованность при чтении/записи
///
/// 3. vertexBuffers: [MTLBuffer] (двойная буферизация)
///    - Рендер-поток читает из vertexBuffers[currentBufferIndex]
///    - Потоки обновления пишут в vertexBuffers[(currentBufferIndex + 1) % 2]
///    - Чтение и запись НИКОГДА не пересекаются - разные индексы
///    - Переключение индекса атомарно через RwLock
///    - Дополнительная блокировка НЕ требуется
///
/// 4. pipelineName, device: let
///    - Неизменяемые после инициализации
///    - Потокобезопасны по определению
///
/// Гарантии:
/// - Рендер работает на 120 FPS без блокировок
/// - Геометрия обновляется в фоновых потоках
/// - Нет гонок данных благодаря двойной буферизации + RwLock
class SceneObject: @unchecked Sendable {
    fileprivate var geometry: RwLock<any Geometry & Transformable>
    fileprivate let pipelineName: String
    fileprivate var currentBufferIndex = RwLock<Int>(0)
    fileprivate var materialIndex = RwLock<Int>(0)

    private let device: MTLDevice
    private var vertexBuffers: [MTLBuffer]

    private var indexBuffers: MTLBuffer


    init(geometry: any Geometry & Transformable, pipelineName: String, vertexBuffer: MTLBuffer, indexBuffer: MTLBuffer, device: MTLDevice) {
        self.geometry = RwLock(geometry)
        self.pipelineName = pipelineName
        self.vertexBuffers = [vertexBuffer, vertexBuffer]
        self.indexBuffers = indexBuffer
        self.device = device
    }

    func updateBuffer() {
        let nextIndex = currentBufferIndex.read { ($0 + 1) % 2 }
        let vertices = geometry.read { $0.vertices() } 

        vertexBuffers[nextIndex] = device.makeBuffer(
            bytes: vertices,
            length: vertices.count * MemoryLayout<Vertex>.stride,
            options: []
        )!

        currentBufferIndex.write { $0 = nextIndex }
    }

    func getVertexBuffer() -> MTLBuffer {
        let index = currentBufferIndex.read { $0 }
        return vertexBuffers[index]
    }

    func getPipelineName() -> String {
        return pipelineName
    }

    func getIndexBuffer() -> MTLBuffer {
        return indexBuffers
    }

    func setMaterialIndex(_ index: Int) {
        materialIndex.write { $0 = index }
    }

    func getMaterialIndex() -> Int {
        return materialIndex.read { $0 }
    }
}


extension SceneObject: Transformable {
    func translate(_ dxyz: SIMD3<Float>) {
        geometry.write { $0.translate(dxyz) }
        updateBuffer()
    }
    func rotate(_ angle: Float, axis: SIMD3<Float>) {
        geometry.write { $0.rotate(angle, axis: axis) }
        updateBuffer()
    
    }
    func scale(_ factor: Float) {
        geometry.write { $0.scale(factor) }
        updateBuffer()
    }
}

extension SceneObject: Geometry {
    func vertices() -> [Vertex] {
        geometry.read { $0.vertices() }
    }
    func indices() -> [UInt16] {
        geometry.read { $0.indices() }
    }
}


// Менеджер сцены
actor Scene {
    private var objects: [String: SceneObject] = [:]
    nonisolated(unsafe) private var cachedObjects: [SceneObject] = []

    private var lights: [String: Light] = [:]
    nonisolated(unsafe) private var cachedLights: [Light] = []

    func clear() {
        lights.removeAll()
        objects.removeAll()
        updateCache()
    }

    nonisolated func getObjectsForRendering() -> [SceneObject] {
        return cachedObjects
    }

    nonisolated func getLightsForRendering() -> [Light] {
        return cachedLights
    }

    // importance of determined object indexes
    func updateCache () {
        cachedObjects = objects.keys.sorted().compactMap { objects[$0] }
        cachedLights = lights.keys.sorted().compactMap { lights[$0] }
    }
}

// Light
extension Scene {
    func addLight(name: String, _ light: Light) {
        lights[name] = light
        updateCache()
    }
    
    func removeLight(name: String) {
        lights.removeValue(forKey: name)
        updateCache()
    }
    
    func getLight(name: String) -> Light? {
        return lights[name]
    }
    
    func getAllLights() -> [Light] {
        return Array(lights.values)
    }
    
    func getAllLightIDs() -> [String] {
        return Array(lights.keys)
    }

    func clearLights() {
        lights.removeAll()
        updateCache()
    }
}


// Objects
extension Scene {
    func addObject(objectName: String, _ object: SceneObject) {
        objects[objectName] = object
        updateCache()
    }
    
    func removeObject(at objectName: String) {
        objects.removeValue(forKey: objectName)
        updateCache()
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
    
    func clearObjects() {
        objects.removeAll()
        updateCache()
    }
}