import AppKit
@preconcurrency import MetalKit

enum RenderMode {
    case rasterization
    case raytracing
}

// Raytracing
extension Renderer {
    private func buildAccelerationStructure() {
        let objects = scene.getObjectsForRendering()
        
        guard !objects.isEmpty else {
            return
        }
        
        let geometries = objects.map { object -> MTLAccelerationStructureTriangleGeometryDescriptor in
            let descriptor = MTLAccelerationStructureTriangleGeometryDescriptor()
            descriptor.vertexBuffer = object.getVertexBuffer()
            descriptor.vertexStride = MemoryLayout<Vertex>.stride
            descriptor.indexBuffer = object.getIndexBuffer()
            descriptor.indexType = .uint16
            descriptor.triangleCount = object.indices().count / 3
            return descriptor
        }
        
        let accelDescriptor = MTLPrimitiveAccelerationStructureDescriptor()
        accelDescriptor.geometryDescriptors = geometries
        
        let sizes = device.accelerationStructureSizes(descriptor: accelDescriptor)
        guard let accelStructure = device.makeAccelerationStructure(size: sizes.accelerationStructureSize) else {
            return
        }
        
        guard let scratchBuffer = device.makeBuffer(length: sizes.buildScratchBufferSize, options: .storageModePrivate) else {
            return
        }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { 
            return 
        }
        guard let encoder = commandBuffer.makeAccelerationStructureCommandEncoder() else { 
            return 
        }
        
        encoder.build(accelerationStructure: accelStructure, descriptor: accelDescriptor, scratchBuffer: scratchBuffer, scratchBufferOffset: 0)
        encoder.endEncoding()
        
        // Don't block - let it complete asynchronously
        commandBuffer.commit()
        
        // Store immediately so it's available next frame
        self.accelerationStructure = accelStructure
    }

    func drawRaytracing(in view: MTKView, drawable: MTLDrawable, descriptor: MTLRenderPassDescriptor) {
        // Rebuild acceleration structure every frame to support animated geometry
        buildAccelerationStructure()
        
        if accelerationStructure == nil {
            // If build failed, show clear color
            descriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.1, 0.1, 0.2, 1.0)
            guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
            guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else { return }
            encoder.endEncoding()
            commandBuffer.present(drawable)
            commandBuffer.commit()
            return
        }
        
        guard let accelStructure = accelerationStructure else { 
            print("Acceleration structure not ready")
            return 
        }
        guard let computePipeline = shaderLibrary.getComputePipeline("raytracing") else { 
            print("Raytracing compute pipeline not found")
            return 
        }
        
        // Get texture info from descriptor instead of view to avoid MainActor issues
        guard let msaaTexture = descriptor.colorAttachments[0].texture else {
            print("No MSAA texture in descriptor")
            return
        }
        
        let width = msaaTexture.width
        let height = msaaTexture.height
        
        // Create intermediate texture without MSAA for compute shader output
        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.pixelFormat = msaaTexture.pixelFormat
        textureDescriptor.width = width
        textureDescriptor.height = height
        textureDescriptor.usage = [.shaderWrite, .shaderRead]
        textureDescriptor.storageMode = .private
        
        guard let outputTexture = device.makeTexture(descriptor: textureDescriptor) else {
            print("Failed to create output texture")
            return
        }
        
        // Setup camera data
        let aspect = Float(width) / Float(height)
        var cameraData = camera.getCameraData(fov: .pi / 3, aspect: aspect)
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { 
            print("Failed to create command buffer")
            return 
        }
        
        // Compute pass - write to intermediate texture
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { 
            print("Failed to create compute encoder")
            return 
        }
        
        computeEncoder.setComputePipelineState(computePipeline)
        computeEncoder.setTexture(outputTexture, index: 0)
        computeEncoder.setBytes(&cameraData, length: MemoryLayout<CameraData>.stride, index: 0)
        computeEncoder.setAccelerationStructure(accelStructure, bufferIndex: 1)
        
        let threadgroupSize = MTLSize(width: 8, height: 8, depth: 1)
        let threadgroups = MTLSize(
            width: (width + threadgroupSize.width - 1) / threadgroupSize.width,
            height: (height + threadgroupSize.height - 1) / threadgroupSize.height,
            depth: 1
        )
        
        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)
        computeEncoder.endEncoding()
        
        // Create a simple render pass descriptor to copy to drawable
        let copyDescriptor = MTLRenderPassDescriptor()
        copyDescriptor.colorAttachments[0].texture = descriptor.colorAttachments[0].texture
        copyDescriptor.colorAttachments[0].resolveTexture = descriptor.colorAttachments[0].resolveTexture
        copyDescriptor.colorAttachments[0].loadAction = .dontCare
        copyDescriptor.colorAttachments[0].storeAction = descriptor.colorAttachments[0].resolveTexture != nil ? .multisampleResolve : .store
        
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: copyDescriptor) else {
            print("Failed to create copy render encoder")
            return
        }
        
        // Create a blit pipeline if we don't have one
        if let blitPipeline = getOrCreateBlitPipeline() {
            renderEncoder.setRenderPipelineState(blitPipeline)
            renderEncoder.setFragmentTexture(outputTexture, index: 0)
            renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
        }
        
        renderEncoder.endEncoding()
        
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
    
    private func getOrCreateBlitPipeline() -> MTLRenderPipelineState? {
        // Check if we already have blit pipeline
        if let existing = shaderLibrary.getPipeline("blitRT") {
            return existing
        }
        
        // Create inline blit shader
        let blitShaderSource = """
        #include <metal_stdlib>
        using namespace metal;
        
        struct VertexOut {
            float4 position [[position]];
            float2 texCoord;
        };
        
        vertex VertexOut blit_vertex(uint vertexID [[vertex_id]]) {
            float2 positions[6] = {
                float2(-1, -1), float2(1, -1), float2(-1, 1),
                float2(-1, 1), float2(1, -1), float2(1, 1)
            };
            float2 texCoords[6] = {
                float2(0, 1), float2(1, 1), float2(0, 0),                float2(0, 0), float2(1, 1), float2(1, 0)
            };
            
            VertexOut out;
            out.position = float4(positions[vertexID], 0, 1);
            out.texCoord = texCoords[vertexID];
            return out;
        }
        
        fragment float4 blit_fragment(VertexOut in [[stage_in]],
                                      texture2d<float> tex [[texture(0)]]) {
            constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::nearest);
            return tex.sample(s, in.texCoord);
        }
        """
        
        do {
            let library = try device.makeLibrary(source: blitShaderSource, options: nil)
            let vertexFunc = library.makeFunction(name: "blit_vertex")!
            let fragmentFunc = library.makeFunction(name: "blit_fragment")!
            
            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.vertexFunction = vertexFunc
            pipelineDescriptor.fragmentFunction = fragmentFunc
            pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
            pipelineDescriptor.rasterSampleCount = 4 // Match MSAA
            
            let pipeline = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
            return pipeline
        } catch {
            print("Failed to create blit pipeline: \(error)")
            return nil
        }
    }
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

    // Raytrace
    private var accelerationStructure: MTLAccelerationStructure?

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
    
    func registerComputePipeline(
        pipelineName: String,
        libraryName: String,
        kernelFunction: String
    ) {
        shaderLibrary.createComputePipeline(
            name: pipelineName,
            libraryName: libraryName,
            kernelFunction: kernelFunction
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


    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {

    }
}

// Render mode switching
extension Renderer {
    func toggleRenderMode() {
        renderMode.write { mode in
            mode = mode == .rasterization ? .raytracing : .rasterization
        }
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

// Movement and camera
extension Renderer {
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

    func rotateCamera(yaw: Float, pitch: Float) {
        camera.rotate(yaw: yaw, pitch: pitch)
    }
}

// OBJECT CONTROLS
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

// Rasterization render mode
extension Renderer {
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
}
