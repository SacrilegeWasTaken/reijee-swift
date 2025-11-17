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
    private var mKeyPressed = RwLock<Bool>(false)
    private var gKeyPressed = RwLock<Bool>(false)

    // Raytrace
    private var accelerationStructure: MTLAccelerationStructure?
    private var combinedVertexBuffer: MTLBuffer?
    private var combinedIndexBuffer: MTLBuffer?
    private var asBuilding = false
    private var asReady = false
    private var samplesPerPixel: Int = 1
    private var threadGroupSizeOneDimension: Int = 16
    private var frameIndex: UInt32 = 0
    
    // Progressive rendering
    private var accumulationTexture: MTLTexture?
    private var accumulatedSamples: UInt32 = 0
    private var lastCameraData: CameraData?
    private let maxAccumulatedSamples: UInt32 = 1024
    
    // AO settings
    private var aoEnabled: Bool = true
    private var aoSamples: UInt32 = 8
    private var aoRadius: Float = 1.0
 
    // Global Illumination settings
    private var giEnabled: Bool = true
    private var giSamples: UInt32 = 4
    private var giBounces: UInt32 = 4
    private var giIntensity: Float = 0.33
    private var giFalloff: Float = 1.0 // Added GI falloff parameter
    private var giMaxDistance: Float = 1000.0
    private var giMinDistance: Float = 0.001
    private var giBias: Float = 0.001
    private var giSampleDistribution: [UInt8] = Array("cosine".utf8) // Options: "uniform", "cosine"

    // Materials
    private let materialLibrary = MaterialLibrary()
    private var triangleMaterialIndexBuffer: MTLBuffer?
    private var materialsBuffer: MTLBuffer?
    
    // Environment / HDRI cache
    private var envTextures: [String: MTLTexture] = [:]
    private var textureLoader: MTKTextureLoader!


    init(_ device: MTLDevice, pressedKeysProvider: @escaping () -> Set<UInt16>, shiftProvider: @escaping () -> Bool) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!
        self.scene = Scene()
        self.camera = Camera()
        self.shaderLibrary = ShaderLibrary(device: device)
        self.textureLoader = MTKTextureLoader(device: device)
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
        invalidateAccelerationStructure()
        resetAccumulation()
    }
    
    func toggleGI() {
        giEnabled.toggle()
        resetAccumulation()
    }
    
    private func invalidateAccelerationStructure() {
        accelerationStructure = nil
        combinedVertexBuffer = nil
        combinedIndexBuffer = nil
        asReady = false
        asBuilding = false
    }

    private func updateInput() {
        // Input now handled asynchronously via handleKeyPress
    }
    
    func handleKeyPress(_ keyCode: UInt16) {
        if keyCode == 46 { // M key
            let wasPressed = mKeyPressed.read { $0 }
            if !wasPressed {
                mKeyPressed.write { $0 = true }
                toggleRenderMode()
            }
        }
        if keyCode == 5 { // G key
            let wasPressed = gKeyPressed.read { $0 }
            if !wasPressed {
                gKeyPressed.write { $0 = true }
                toggleGI()
            }
        }
    }
    
    func handleKeyRelease(_ keyCode: UInt16) {
        if keyCode == 46 { // M key
            mKeyPressed.write { $0 = false }
        }
        if keyCode == 5 { // G key
            gKeyPressed.write { $0 = false }
        }
    }
}

// Movement and camera
extension Renderer {
    private func updateCamera() {
        let pressedKeys = pressedKeysProvider()
        let acceleration: Float = 0.008
        let damping: Float = 0.85
        var targetVelocity = SIMD3<Float>(0, 0, 0)
        var cameraMoved = false
        
        if pressedKeys.contains(13) { targetVelocity.z += 1 } // W
        if pressedKeys.contains(1) { targetVelocity.z -= 1 }  // S
        if pressedKeys.contains(0) { targetVelocity.x += 1 }  // A
        if pressedKeys.contains(2) { targetVelocity.x -= 1 }  // D
        if pressedKeys.contains(49) { targetVelocity.y += 1 } // Space
        if shiftProvider() { targetVelocity.y -= 1 } // Shift
        
        cameraVelocity = cameraVelocity * damping + targetVelocity * acceleration
        
        if simd_length(cameraVelocity) > 0.001 {
            camera.move(cameraVelocity)
            cameraMoved = true
        }

        if pressedKeys.contains(123) { camera.rotate(yaw: -0.02, pitch: 0); cameraMoved = true } // Left arrow
        if pressedKeys.contains(124) { camera.rotate(yaw: 0.02, pitch: 0); cameraMoved = true }  // Right arrow
        if pressedKeys.contains(126) { camera.rotate(yaw: 0, pitch: 0.02); cameraMoved = true }  // Up arrow
        if pressedKeys.contains(125) { camera.rotate(yaw: 0, pitch: -0.02); cameraMoved = true } // Down arrow
        
        if cameraMoved {
            resetAccumulation()
        }
    }

    func rotateCamera(yaw: Float, pitch: Float) {
        camera.rotate(yaw: yaw, pitch: pitch)
        resetAccumulation()
    }
    
    private func resetAccumulation() {
        accumulatedSamples = 0
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

        // default material 0
        object.setMaterialIndex(0)

        await scene.addObject(objectName: objectName, object)
        invalidateAccelerationStructure()
        resetAccumulation()
    }
    
    private func cameraDataEqual(_ a: CameraData, _ b: CameraData) -> Bool {
        return simd_distance(a.position, b.position) < 0.0001 &&
               simd_distance(a.forward, b.forward) < 0.0001 &&
               simd_distance(a.right, b.right) < 0.0001 &&
               simd_distance(a.up, b.up) < 0.0001
    }
    
    func removeObject(at objectName: String) async {
       await scene.removeObject(at: objectName)
       invalidateAccelerationStructure()
       resetAccumulation()
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

// MATERIALS
extension Renderer {
    @discardableResult
    func addMaterial(_ mat: PBRMaterial) -> Int {
        let index = materialLibrary.add(mat)
        // Re-upload materials buffer next frame; no need to rebuild AS
        resetAccumulation()
        return index
    }

    func setObjectMaterial(objectName: String, materialIndex: Int) async {
        guard let object = await scene.getObject(id: objectName) else { return }
        object.setMaterialIndex(materialIndex)
        // Rebuild triangle->material mapping for AS
        invalidateAccelerationStructure()
        resetAccumulation()
    }
}

// LIGHT CONTROLS
extension Renderer {
    func addLight(name: String, light: Light) async {
        await scene.addLight(name: name, light)
        resetAccumulation()
    }
    
    func removeLight(name: String) async {
        await scene.removeLight(name: name)
        resetAccumulation()
    }
    
    func getLight(name: String) async -> Light? {
        return await scene.getLight(name: name)
    }
    
    func getAllLights() async -> [Light] {
        return await scene.getAllLights()
    }
    
    func getAllLightIDs() async -> [String] {
        return await scene.getAllLightIDs()
    }
    
    func clearLights() async {
        await scene.clearLights()
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
        var triMaterialIndices: [UInt32] = []
        
        for object in objects {
            let vertices = object.vertices()
            let indices = object.indices()
            
            let vertexOffset = UInt16(allVertices.count)
            allVertices.append(contentsOf: vertices)
            
            let offsetIndices = indices.map { $0 + vertexOffset }
            allIndices.append(contentsOf: offsetIndices)

            // per-triangle material index duplicated for each triangle in this object
            let triangleCount = indices.count / 3
            let matIndex = UInt32(max(0, object.getMaterialIndex()))
            triMaterialIndices.append(contentsOf: Array(repeating: matIndex, count: triangleCount))
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

        // Build triangle->material index buffer
        triangleMaterialIndexBuffer = device.makeBuffer(
            bytes: triMaterialIndices,
            length: triMaterialIndices.count * MemoryLayout<UInt32>.stride,
            options: []
        )
        
        // Single geometry descriptor for all objects
        let geometryDescriptor = MTLAccelerationStructureTriangleGeometryDescriptor()
        geometryDescriptor.vertexBuffer = combinedVertexBuffer
        geometryDescriptor.vertexStride = MemoryLayout<Vertex>.stride
        geometryDescriptor.indexBuffer = combinedIndexBuffer
        geometryDescriptor.indexType = .uint16
        geometryDescriptor.triangleCount = allIndices.count / 3
        
        let geometries = [geometryDescriptor]
        
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

    // Load or return cached environment (HDRI) texture by name. Tries bundle/resource then relative Assets/ path.
    private func getEnvironmentTexture(_ name: String) -> MTLTexture? {
        if let tex = envTextures[name] {
            return tex
        }

        // If the provided name is already an absolute path or exists as-is, try it first
        func tryLoad(_ url: URL) -> MTLTexture? {
            if FileManager.default.fileExists(atPath: url.path) {
                do {
                    let options: [MTKTextureLoader.Option: Any] = [
                        .SRGB: false,
                        .generateMipmaps: NSNumber(value: true)
                    ]
                    let tex = try textureLoader.newTexture(URL: url, options: options)
                    print("[Renderer] Loaded environment texture \(url.path)")
                    envTextures[name] = tex
                    return tex
                } catch {
                    print("[Renderer] Failed to load environment texture at \(url): \(error)")
                }
            }
            return nil
        }

        // Direct path (absolute or relative) provided
        if name.hasPrefix("file://"), let fileURL = URL(string: name) {
            if let tex = tryLoad(fileURL) { return tex }
        }

        if FileManager.default.fileExists(atPath: name) {
            let fileURL = URL(fileURLWithPath: name).standardizedFileURL
            if let tex = tryLoad(fileURL) { return tex }
        }

        // Build candidate URLs: bundle resource and project Assets
        var urlCandidates: [URL] = []
        if !name.hasPrefix("/") {
            if let res = Bundle.main.resourceURL {
                urlCandidates.append(res.appendingPathComponent(name))
            }
            let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            urlCandidates.append(cwd.appendingPathComponent("Assets").appendingPathComponent(name))
        } else {
            // If name is absolute but earlier check failed, try standardized variant
            let std = URL(fileURLWithPath: name).standardizedFileURL
            urlCandidates.append(std)
        }

        for url in urlCandidates {
            if let tex = tryLoad(url) { return tex }
        }

        // Debug: print tried paths
        print("[Renderer] Environment texture not found for name '\(name)'. Tried:")
        for url in urlCandidates { print("  - \(url.path)") }
        return nil
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
        var cameraData = camera.getCameraData(fov: .pi / 3, aspect: aspect, frameIndex: frameIndex)
        
        // Check if camera moved
        if let lastCamera = lastCameraData {
            if !cameraDataEqual(lastCamera, cameraData) {
                resetAccumulation()
            }
        }
        lastCameraData = cameraData
        
        // Create or recreate accumulation texture if needed
        if accumulationTexture == nil || accumulationTexture!.width != width || accumulationTexture!.height != height {
            let accumulationDescriptor = MTLTextureDescriptor()
            accumulationDescriptor.pixelFormat = .rgba32Float
            accumulationDescriptor.width = width
            accumulationDescriptor.height = height
            accumulationDescriptor.usage = [.shaderWrite, .shaderRead]
            accumulationDescriptor.storageMode = .private
            accumulationTexture = device.makeTexture(descriptor: accumulationDescriptor)
            resetAccumulation()
        }
        
        // Increment samples if not at max
        if accumulatedSamples < maxAccumulatedSamples {
            accumulatedSamples += 1
            frameIndex = frameIndex &+ 1
        }


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
        guard let triMatBuffer = triangleMaterialIndexBuffer else {
            print("No triangle->material buffer")
            return
        }

        var samples = UInt32(samplesPerPixel)
        
        // Setup lights
        let lights = scene.getLightsForRendering()
        var lightDataArray = lights.map { $0.toLightData() }
        var lightCount = UInt32(lightDataArray.count)

        // Setup materials
        let mats = materialLibrary.getAll()
        var materialCount = UInt32(mats.count)
        if mats.count > 0 {
            materialsBuffer = device.makeBuffer(bytes: mats, length: mats.count * MemoryLayout<PBRMaterial>.stride, options: [])
        }
        
        // arguments for a compute shader
        computeEncoder.setComputePipelineState(computePipeline) // need to be first 
        computeEncoder.setTexture(outputTexture, index: 0)
        computeEncoder.setTexture(accumulationTexture, index: 1)
        computeEncoder.setBytes(&cameraData, length: MemoryLayout<CameraData>.stride, index: 0)
        computeEncoder.setAccelerationStructure(accelStructure, bufferIndex: 1)
        computeEncoder.setBuffer(vertexBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(indexBuffer, offset: 0, index: 3)
        computeEncoder.setBytes(&samples, length: MemoryLayout<UInt32>.stride, index: 4)
        computeEncoder.setBytes(&lightCount, length: MemoryLayout<UInt32>.stride, index: 5)
        if !lightDataArray.isEmpty {
            computeEncoder.setBytes(&lightDataArray, length: MemoryLayout<LightData>.stride * lightDataArray.count, index: 6)
        }
        computeEncoder.setBytes(&accumulatedSamples, length: MemoryLayout<UInt32>.stride, index: 7)
        
        var aoEnabledInt = aoEnabled ? UInt32(1) : UInt32(0)
        var aoSamplesVar = aoSamples
        var aoRadiusVar = aoRadius
        computeEncoder.setBytes(&aoEnabledInt, length: MemoryLayout<UInt32>.stride, index: 8)
        computeEncoder.setBytes(&aoSamplesVar, length: MemoryLayout<UInt32>.stride, index: 9)
        computeEncoder.setBytes(&aoRadiusVar, length: MemoryLayout<Float>.stride, index: 10)
        
        var giEnabledInt = giEnabled ? UInt32(1) : UInt32(0)
        var giSamplesVar = giSamples
        var giBouncesVar = giBounces
        var giIntensityVar = giIntensity
        var giFalloffVar = giFalloff // New falloff variable
        var giMaxDistanceVar = giMaxDistance
        var giMinDistanceVar = giMinDistance
        var giBiasVar = giBias
        var giSampleDistributionVar = giSampleDistribution
        computeEncoder.setBytes(&giEnabledInt, length: MemoryLayout<UInt32>.stride, index: 11)
        computeEncoder.setBytes(&giSamplesVar, length: MemoryLayout<UInt32>.stride, index: 12)
        computeEncoder.setBytes(&giBouncesVar, length: MemoryLayout<UInt32>.stride, index: 13)
        computeEncoder.setBytes(&giIntensityVar, length: MemoryLayout<Float>.stride, index: 14)
        computeEncoder.setBytes(&giFalloffVar, length: MemoryLayout<Float>.stride, index: 15) // Set falloff variable
        computeEncoder.setBytes(&giMaxDistanceVar, length: MemoryLayout<Float>.stride, index: 16)
        computeEncoder.setBytes(&giMinDistanceVar, length: MemoryLayout<Float>.stride, index: 17)
        computeEncoder.setBytes(&giBiasVar, length: MemoryLayout<Float>.stride, index: 18)
        computeEncoder.setBytes(&giSampleDistributionVar, length: MemoryLayout<String>.stride, index: 19)

        // Environment (HDRI) setup: find first DomeLight and bind its texture + params
        var envPresent: UInt32 = 0
        var envIntensityVar: Float = 1.0
        var envRotationVar = SIMD3<Float>(0,0,0)
        var envTintVar = SIMD3<Float>(1,1,1)
        var envTexture: MTLTexture? = nil
        for light in lights {
            if let dome = light as? DomeLight {
                envIntensityVar = dome.getIntensity()
                envRotationVar = dome.getRotation()
                envTintVar = dome.getTint()
                envTexture = getEnvironmentTexture(dome.getTextureName())
                if envTexture != nil {
                    envPresent = 1
                }
                break
            }
        }
        if let envTexture = envTexture {
            computeEncoder.setTexture(envTexture, index: 2)
        }
        computeEncoder.setBytes(&envPresent, length: MemoryLayout<UInt32>.stride, index: 23)
        computeEncoder.setBytes(&envIntensityVar, length: MemoryLayout<Float>.stride, index: 24)
        computeEncoder.setBytes(&envRotationVar, length: MemoryLayout<SIMD3<Float>>.stride, index: 25)
        computeEncoder.setBytes(&envTintVar, length: MemoryLayout<SIMD3<Float>>.stride, index: 26)

        // Materials buffers (20+)
        computeEncoder.setBytes(&materialCount, length: MemoryLayout<UInt32>.stride, index: 20)
        if let materialsBuffer = materialsBuffer {
            computeEncoder.setBuffer(materialsBuffer, offset: 0, index: 21)
        }
        computeEncoder.setBuffer(triMatBuffer, offset: 0, index: 22)
        
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