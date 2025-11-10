import AppKit
import MetalKit

class Renderer: NSObject, MTKViewDelegate, @unchecked Sendable{
    fileprivate let device: MTLDevice 
    fileprivate let commandQueue: MTLCommandQueue
    fileprivate let shaderLibrary: ShaderLibrary
    fileprivate let scene: Scene

    init(_ device: MTLDevice) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        self.scene = Scene()

        let shaderPath = #filePath.replacingOccurrences(of: "renderer.swift", with: "shaders.metal")
        self.shaderLibrary = ShaderLibrary(device: device)
        self.shaderLibrary.loadLibrary(name: "triangle", path: shaderPath)
        
        // Регистрируем pipeline
        shaderLibrary.createPipeline(
            name: "triangle",
            libraryName: "triangle",
            vertexFunction: "vertex_main",
            fragmentFunction: "fragment_main",
            pixelFormat: .bgra8Unorm_srgb
        )

        super.init()
    }


    func registerLibrary(
        libraryName: String,
        shaderPath: String,
    ) {
        self.shaderLibrary.loadLibrary(name: libraryName, path: shaderPath)
    }


    func registerPipeline(
        pipelineName: String,
        libraryName: String,
        vertexFunction: String,
        fragmentFunction: String,
        pixelFormat: MTLPixelFormat
    ) {        
        shaderLibrary.createPipeline(
            name: pipelineName,
            libraryName: libraryName,
            vertexFunction: vertexFunction,
            fragmentFunction: fragmentFunction,
            pixelFormat: pixelFormat
        )
    }




    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable else { return }
        guard let descriptor = view.currentRenderPassDescriptor else { return }

        descriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0)
    
        // создаём командный буфер
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }

        // создаем рендер энкодер
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else { return }

        for object in scene.getObjectsForRendering() {
            guard let pipeline = shaderLibrary.getPipeline(object.getPipelineName()) else { 
                print("Pipeline \(object.getPipelineName()) not found")
                continue
             }

             encoder.setRenderPipelineState(pipeline)
             encoder.setVertexBuffer(object.getVertexBuffer(), offset: 0, index: 0)
             encoder.drawPrimitives(
                type: .triangle,
                vertexStart: 0,
                vertexCount: object.vetricies().count
             )
        }
        

        encoder.endEncoding()

        // отображаем результат
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }


    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {

    }
}


extension Renderer {
    func addObject(objectName: String, geometry: any _2DGeometry & _2DMovable, pipelineName: String) async {
        let vetricies = geometry.vetricies()
        let buffer = device.makeBuffer(
            bytes: vetricies,
            length: vetricies.count * MemoryLayout<Vertex>.stride,
            options: []
        )!

        let object = SceneObject(
            geometry: geometry,
            pipelineName: pipelineName,
            vertexBuffer: buffer,
            device: device
        )

        await scene.addObject(objectName: objectName, object)
    }
    
    func removeObject(at objectName: String) async {
       await scene.removeObject(at: objectName)
    }
    
    func getObjects() async -> [SceneObject] {
        return await scene.getObjects()
    }
    
    func getAllObjectIDs() async -> [String]  {
        return await scene.getAllObjectIDs()
    }
    
    func getObject(objectName: String) async -> SceneObject? {
        return await scene.getObject(id: objectName)
    }
    
    func clear() async {
        await scene.clear()
    }
}