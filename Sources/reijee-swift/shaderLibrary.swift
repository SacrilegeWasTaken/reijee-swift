import AppKit
import MetalKit

class ShaderLibrary {
    private let device: MTLDevice
    private var libraries: [String: MTLLibrary] = [:]
    private var pipelines: [String: MTLRenderPipelineState] = [:]
    private var computePipelines: [String: MTLComputePipelineState] = [:]
    
    init(device: MTLDevice) {
        self.device = device
    }
    
    // Загрузить библиотеку из файла
    func loadLibrary(name: String, path: String) {
        // Пробуем загрузить как .metallib
        if path.hasSuffix(".metallib") {
            libraries[name] = try! device.makeLibrary(URL: URL(fileURLWithPath: path))
            return
        }
        
        // Иначе компилируем из исходников с поддержкой #include
        var source = try! String(contentsOfFile: path, encoding: .utf8)
        
        let dir = (path as NSString).deletingLastPathComponent
        let includePattern = try! NSRegularExpression(pattern: "#include \"(.+?)\"")
        let matches = includePattern.matches(in: source, range: NSRange(source.startIndex..., in: source))
        
        for match in matches.reversed() {
            let range = Range(match.range(at: 1), in: source)!
            let includePath = dir + "/" + String(source[range])
            let includeContent = try! String(contentsOfFile: includePath, encoding: .utf8)
            let fullRange = Range(match.range, in: source)!
            source.replaceSubrange(fullRange, with: includeContent)
        }
        
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
    
    // Create compute pipeline
    func createComputePipeline(name: String, libraryName: String, kernelFunction: String) {
        guard let library = libraries[libraryName] else { return }
        guard let function = library.makeFunction(name: kernelFunction) else { return }
        computePipelines[name] = try! device.makeComputePipelineState(function: function)
    }
    
    func getComputePipeline(_ name: String) -> MTLComputePipelineState? {
        return computePipelines[name]
    }
}
