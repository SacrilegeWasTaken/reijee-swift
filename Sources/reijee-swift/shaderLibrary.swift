import AppKit
import MetalKit

class ShaderLibrary {
    private let device: MTLDevice
    private var libraries: [String: MTLLibrary] = [:]
    private var pipelines: [String: MTLRenderPipelineState] = [:]
    
    init(device: MTLDevice) {
        self.device = device
    }
    
    // Загрузить библиотеку из файла
    func loadLibrary(name: String, path: String) {
        let source = try! String(contentsOfFile: path, encoding: .utf8)
        libraries[name] = try! device.makeLibrary(source: source, options: nil)
    }
    
    // Создать pipeline из конкретной библиотеки
    func createPipeline(name: String, libraryName: String, vertexFunction: String, fragmentFunction: String, pixelFormat: MTLPixelFormat) {
        guard let library = libraries[libraryName] else { return }
        
        let vertexFunc = library.makeFunction(name: vertexFunction)!
        let fragmentFunc = library.makeFunction(name: fragmentFunction)!
        
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertexFunc
        descriptor.fragmentFunction = fragmentFunc
        descriptor.colorAttachments[0].pixelFormat = pixelFormat
        
        pipelines[name] = try! device.makeRenderPipelineState(descriptor: descriptor)
    }
    
    func getPipeline(_ name: String) -> MTLRenderPipelineState? {
        return pipelines[name]
    }
}
