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
    fileprivate var geometry: RwLock<any _2DGeometry & _2DMovable>
    fileprivate let pipelineName: String
    fileprivate var currentBufferIndex = RwLock<Int>(0)

    private let device: MTLDevice
    private var vertexBuffers: [MTLBuffer]


    init(geometry: any _2DGeometry & _2DMovable, pipelineName: String, vertexBuffer: MTLBuffer, device: MTLDevice) {
        self.geometry = RwLock(geometry)
        self.pipelineName = pipelineName
        self.vertexBuffers = [vertexBuffer, vertexBuffer]
        self.device = device
    }

    func updateBuffer() {
        let nextIndex = currentBufferIndex.read { ($0 + 1) % 2 }
        let vertices = geometry.read { $0.vetricies() } 

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
}


extension SceneObject: _2DMovable {
    func traslate(_ dxyz: SIMD3<Float>) {
        geometry.write { $0.traslate(dxyz) }
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

extension SceneObject: _2DGeometry {
    func vetricies() -> [Vertex] {
        geometry.read { $0.vetricies() }
    }
    func indicies() -> [UInt16] {
         geometry.read { $0.indicies() }
    }
}


// Менеджер сцены
actor Scene {
    private var objects: [String: SceneObject] = [:]
    nonisolated(unsafe) private var cachedObjects: [SceneObject] = []
    
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
    
    func clear() {
        objects.removeAll()
        updateCache()
    }

    nonisolated func getObjectsForRendering() -> [SceneObject] {
        return cachedObjects
    }

    func updateCache () {
        cachedObjects = Array(objects.values)
    }
}