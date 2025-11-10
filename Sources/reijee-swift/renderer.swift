import AppKit
@preconcurrency import MetalKit

enum RenderMode {
    case rasterization
    case raytracing
}

class Renderer: NSObject, MTKViewDelegate, @unchecked Sendable{
    fileprivate let device: MTLDevice 
    fileprivate let commandQueue: MTLCommandQueue
    fileprivate let shaderLibrary: ShaderLibrary
    fileprivate let scene: Scene
    fileprivate let depthStencilState: MTLDepthStencilState
    fileprivate let camera: Camera

    private let pressedKeysProvider: () -> Set<UInt16>
    private let shiftProvider: () -> Bool
    private var cameraVelocity = SIMD3<Float>(0, 0, 0)

    private var renderMode = RwLock<RenderMode>(.rasterization)
    private var wasMPressed = false

    init(_ device: MTLDevice, pressedKeysProvider: @escaping () -> Set<UInt16>, shiftProvider: @escaping () -> Bool) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        self.scene = Scene()
        self.camera = Camera()
        self.shaderLibrary = ShaderLibrary(device: device)
        self.pressedKeysProvider = pressedKeysProvider
        self.shiftProvider = shiftProvider

        let depthDescriptor = MTLDepthStencilDescriptor()
        depthDescriptor.depthCompareFunction = .less
        depthDescriptor.isDepthWriteEnabled = true
        self.depthStencilState = device.makeDepthStencilState(descriptor: depthDescriptor)!

        super.init()
    }
    
    
    func toggleRenderMode() {
        renderMode.write { mode in
            mode = mode == .rasterization ? .raytracing : .rasterization
        }
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
        updateCamera()
        updateInput()
        guard let drawable = view.currentDrawable else { return }
        guard let descriptor = view.currentRenderPassDescriptor else { return }
        let mode = renderMode.read { $0 }
        
        switch mode {
        case .rasterization:
            drawRasterization(in: view, drawable: drawable, descriptor: descriptor)
        case .raytracing:
            drawRaytracing(in: view, drawable: drawable, descriptor: descriptor)
        }
    }


    func drawRaytracing(in view: MTKView, drawable: MTLDrawable, descriptor: MTLRenderPassDescriptor) {

        descriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.1, 0.1, 0.2, 1.0)
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else { return }
        encoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    
    func drawRasterization(in view: MTKView, drawable: MTLDrawable, descriptor: MTLRenderPassDescriptor) {

        descriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0)
        descriptor.depthAttachment.loadAction = .clear
        descriptor.depthAttachment.clearDepth = 1.0
    
        // создаём командный буфер
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }

        // создаем рендер энкодер
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else { return }
        let aspect = Float(descriptor.colorAttachments[0].texture!.width) / Float(descriptor.colorAttachments[0].texture!.height)
        var uniforms = Uniforms(
            projectionMatrix: camera.projectionMatrix(fov: .pi / 3, aspect: aspect, near: 0.1, far: 100),
            viewMatrix: camera.viewMatrix()
        )

        for object in scene.getObjectsForRendering() {
            guard let pipeline = shaderLibrary.getPipeline(object.getPipelineName()) else { 
                print("Pipeline \(object.getPipelineName()) not found")
                continue
             }

            encoder.setRenderPipelineState(pipeline)
            encoder.setVertexBuffer(object.getVertexBuffer(), offset: 0, index: 0)
            encoder.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
            encoder.setDepthStencilState(depthStencilState)
            encoder.drawIndexedPrimitives(
                type: .triangle, 
                indexCount: object.indices().count, 
                indexType: .uint16, 
                indexBuffer: object.getIndexBuffer(), 
                indexBufferOffset: 0
            )     
        }
        

        encoder.endEncoding()

        // отображаем результат
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }


    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {

    }

    func rotateCamera(yaw: Float, pitch: Float) {
        camera.rotate(yaw: yaw, pitch: pitch)
    }
    

    private func updateCamera() {
        let pressedKeys = pressedKeysProvider()
        let acceleration: Float = 0.008
        let damping: Float = 0.85
        var targetVelocity = SIMD3<Float>(0, 0, 0)
        
        if pressedKeys.contains(13) { targetVelocity.z += 1 } // W
        if pressedKeys.contains(1) { targetVelocity.z -= 1 }  // S
        if pressedKeys.contains(0) { targetVelocity.x += 1 }  // A
        if pressedKeys.contains(2) { targetVelocity.x -= 1 }  // D
        if pressedKeys.contains(49) { targetVelocity.y += 1 } // Space
        if shiftProvider() { targetVelocity.y -= 1 } // Shift
        
        cameraVelocity = cameraVelocity * damping + targetVelocity * acceleration
        
        if simd_length(cameraVelocity) > 0.001 {
            camera.move(cameraVelocity)
        }

        if pressedKeys.contains(123) { camera.rotate(yaw: -0.02, pitch: 0) } // Left arrow
        if pressedKeys.contains(124) { camera.rotate(yaw: 0.02, pitch: 0) }  // Right arrow
        if pressedKeys.contains(126) { camera.rotate(yaw: 0, pitch: 0.02) }  // Up arrow
        if pressedKeys.contains(125) { camera.rotate(yaw: 0, pitch: -0.02) } // Down arrow
    }

    private func updateInput() {
        let pressedKeys = pressedKeysProvider()
        let isMPressed = pressedKeys.contains(46) // M
        if isMPressed && !wasMPressed {
            toggleRenderMode()
        }
        wasMPressed = isMPressed
    }
}


extension Renderer {
    func addObject(objectName: String, geometry: any Geometry & Transformable, pipelineName: String) async {
        let vetricies = geometry.vertices()
        let indicies = geometry.indices()
        let vertexBuffer = device.makeBuffer(
            bytes: vetricies,
            length: vetricies.count * MemoryLayout<Vertex>.stride,
            options: []
        )!

        let indexBuffer = device.makeBuffer(
            bytes: indicies,
            length: indicies.count * MemoryLayout<UInt16>.stride,
            options: []
        )!

        

        let object = SceneObject(
            geometry: geometry,
            pipelineName: pipelineName,
            vertexBuffer: vertexBuffer,
            indexBuffer: indexBuffer,
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