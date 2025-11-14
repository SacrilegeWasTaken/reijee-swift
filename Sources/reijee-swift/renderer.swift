@preconcurrency import AppKit
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

    // Raytrace
    private var accelerationStructure: MTLAccelerationStructure?
    private var combinedVertexBuffer: MTLBuffer?
    private var combinedIndexBuffer: MTLBuffer?
    private var asBuilding = false
    private var asReady = false
    private var samplesPerPixel: Int = 1
    private var threadGroupSizeOneDimension: Int = 16

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

// Raytracing
extension Renderer {
    private func buildAccelerationStructure() {
        guard !asBuilding else { return }
        asBuilding = true
        
        let objects = scene.getObjectsForRendering()
        
        guard !objects.isEmpty else {
            asBuilding = false
            return
        }
        
        // Combine all vertices and indices into single buffers
        var allVertices: [Vertex] = []
        var allIndices: [UInt16] = []
        var indexOffset: UInt16 = 0
        
        // GEOMETRY DESCRIPTORS FOR EACH OBJECT IN THE SCENE
        let geometries = objects.map { object -> MTLAccelerationStructureTriangleGeometryDescriptor in
            let vertices = object.vertices()
            let indices = object.indices()
            
            // Add vertices
            allVertices.append(contentsOf: vertices)
            
            // Add indices with offset
            let offsetIndices = indices.map { $0 + indexOffset }
            allIndices.append(contentsOf: offsetIndices)
            indexOffset += UInt16(vertices.count)
            
            // Create geometry descriptor
            let descriptor = MTLAccelerationStructureTriangleGeometryDescriptor()
            descriptor.vertexBuffer = object.getVertexBuffer()
            descriptor.vertexStride = MemoryLayout<Vertex>.stride
            descriptor.indexBuffer = object.getIndexBuffer()
            descriptor.indexType = .uint16
            descriptor.triangleCount = indices.count / 3
            return descriptor
        }
        
        // Create combined buffers
        combinedVertexBuffer = device.makeBuffer(
            bytes: allVertices,
            length: allVertices.count * MemoryLayout<Vertex>.stride,
            options: []
        )
        
        combinedIndexBuffer = device.makeBuffer(
            bytes: allIndices,
            length: allIndices.count * MemoryLayout<UInt16>.stride,
            options: []
        )
        
        let accelDescriptor = MTLPrimitiveAccelerationStructureDescriptor()
        accelDescriptor.geometryDescriptors = geometries
        
        
        let sizes = device.accelerationStructureSizes(descriptor: accelDescriptor)
        
        // CREATING CONTAINER FOR BVH-TREES
        guard let accelStructure = device.makeAccelerationStructure(size: sizes.accelerationStructureSize) else {
            asBuilding = false
            return
        }
        
        // TEMPORARY BUFFER FOR INTERMIDIATE COMPUTATIONS (while constructing BVH it's computing bboxes / sotring triangles and building hierarchy)
        guard let scratchBuffer = device.makeBuffer(length: sizes.buildScratchBufferSize, options: .storageModePrivate) else {
            asBuilding = false
            return
        }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            asBuilding = false
            return
        }

        // Special acceleration scructure command encoder
        guard let encoder = commandBuffer.makeAccelerationStructureCommandEncoder() else {
            asBuilding = false
            return
        }
        
        encoder.build(
            accelerationStructure: accelStructure, // WHERE TO BUILD
            descriptor: accelDescriptor, // WHAT TO BUILD
            scratchBuffer: scratchBuffer, // TEMPORARY MEMORY
            scratchBufferOffset: 0 // OFFSET IN THE SCRATCH BUFFER
        )
        encoder.endEncoding() // BUILDING BVH TREE
        
        commandBuffer.addCompletedHandler { [weak self] _ in // ASYNC BUILDING
            self?.accelerationStructure = accelStructure
            self?.asReady = true
            self?.asBuilding = false
        }
        
        commandBuffer.commit()
    }

    func drawRaytracing(in view: MTKView, drawable: MTLDrawable, descriptor: MTLRenderPassDescriptor) {
        if accelerationStructure == nil && !asBuilding {
            buildAccelerationStructure()
        }
        
        guard asReady, let accelStructure = accelerationStructure else {
            drawRasterization(in: view, drawable: drawable, descriptor: descriptor)
            return
        }
        
        // getting our raytracing shader
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
        
        // getting our combined buffers which are created in `buildAccelerationStructure` func
        guard let vertexBuffer = combinedVertexBuffer else {
            print("No vertex buffer")
            return
        }
        guard let indexBuffer = combinedIndexBuffer else {
            print("No index buffer")
            return
        }

        var samples = UInt32(samplesPerPixel)
        
        // arguments for a compute shader
        computeEncoder.setComputePipelineState(computePipeline) // need to be first 
        computeEncoder.setTexture(outputTexture, index: 0)
        computeEncoder.setBytes(&cameraData, length: MemoryLayout<CameraData>.stride, index: 0)
        computeEncoder.setAccelerationStructure(accelStructure, bufferIndex: 1)
        computeEncoder.setBuffer(vertexBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(indexBuffer, offset: 0, index: 3)
        computeEncoder.setBytes(&samples, length: MemoryLayout<UInt32>.stride, index: 4)
        
        let threadgroupSize = MTLSize(width: threadGroupSizeOneDimension, height: threadGroupSizeOneDimension, depth: 1) // TODO: make it configurable
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
        copyDescriptor.colorAttachments[0].resolveTexture = descriptor.colorAttachments[0].resolveTexture // final texture
        copyDescriptor.colorAttachments[0].loadAction = .dontCare
        copyDescriptor.colorAttachments[0].storeAction = descriptor.colorAttachments[0].resolveTexture != nil ? .multisampleResolve : .store
        
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: copyDescriptor) else {
            print("Failed to create copy render encoder")
            return
        }
        

        guard let blitPipeline = shaderLibrary.getPipeline("blit") else {
            print("Blit pipeline not found")
            return
        
        }

        // Create a blit pipeline if we don't have one
        renderEncoder.setRenderPipelineState(blitPipeline)
        renderEncoder.setFragmentTexture(outputTexture, index: 0)
        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
        
        renderEncoder.endEncoding()
        
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}