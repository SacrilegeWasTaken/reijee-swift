import AppKit
import MetalKit

class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice 
    let commandQueue: MTLCommandQueue
    let shaderLibrary: ShaderLibrary
    let scene: Scene


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

    func addObject(geometry: any _2DGeomtry, pipelineName: String) {
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
        )

        scene.addObject(object)
    }


    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable else { return }
        guard let descriptor = view.currentRenderPassDescriptor else { return }

        descriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0)
    
        // создаём командный буфер
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }

        // создаем рендер энкодер
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else { return }

        for object in scene.getObjects() {
            guard let pipeline = shaderLibrary.getPipeline(object.pipelineName) else { 
                print("Pipeline \(object.pipelineName) not found")
                continue
             }

             encoder.setRenderPipelineState(pipeline)
             encoder.setVertexBuffer(object.vertexBuffer, offset: 0, index: 0)
             encoder.drawPrimitives(
                type: .triangle,
                vertexStart: 0,
                vertexCount: object.geometry.vetricies().count
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
