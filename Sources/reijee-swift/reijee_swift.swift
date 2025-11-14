import Cocoa
import MetalKit


@main
@MainActor
class AppDelegate: NSObject, NSApplicationDelegate {
    static func main() {
        let app = NSApplication.shared
        app.setActivationPolicy(.regular)
        
        let menubar = NSMenu()
        let appMenuItem = NSMenuItem()
        menubar.addItem(appMenuItem)
        app.mainMenu = menubar
        
        let appMenu = NSMenu()
        let quitMenuItem = NSMenuItem(title: "Quit", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appMenu.addItem(quitMenuItem)
        appMenuItem.submenu = appMenu
        
        let delegate = AppDelegate()
        app.delegate = delegate
        app.run()
    }

    var window: NSWindow!
    var renderer: Renderer!
    var animationTimer: Timer?
    var pressedKeys: Set<UInt16> = []
    var isShiftPressed = false

    func applicationDidFinishLaunching(_ notification: Notification) {
        setenv("MTL_HUD_ENABLED", "1", 1)
        let contentRect = NSRect(x: 0, y: 0, width: 800, height: 800)
        window = NSWindow(contentRect: contentRect,
                          styleMask: [.titled, .closable, .miniaturizable, .resizable],
                          backing: .buffered,
                          defer: false)
        window.title = "reijee-renderer"
        window.acceptsMouseMovedEvents = true


        // создаём Metal view
        let metalView = KeyboardView(frame: contentRect)
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        // cоздаем делегат для mtkView
        renderer = Renderer(device, pressedKeysProvider: { [weak self] in
            self?.pressedKeys ?? []
        }, shiftProvider: { [weak self] in
            self?.isShiftPressed ?? false
        })

        setupRenderer()
        // назначаем device и указываем формат пикселя
        metalView.device = device
        metalView.colorPixelFormat = .bgra8Unorm_srgb
        // назначаем делегат для отрисовки (должен наследоваться протокол MTKViewDelegate)
        metalView.delegate = renderer
        metalView.enableSetNeedsDisplay = false
        metalView.clearDepth = 1.0
        metalView.sampleCount = 4 // MSAA 4x
        metalView.depthStencilPixelFormat = .depth32Float
        metalView.isPaused = false
        metalView.preferredFramesPerSecond = 120

        metalView.onKeyDown = { [weak self] keyCode in
            self?.pressedKeys.insert(keyCode)
        }
        
        metalView.onKeyUp = { [weak self] keyCode in
            self?.pressedKeys.remove(keyCode)
        }
        
        metalView.onMouseMove = { [weak self] deltaX, deltaY in
            self?.renderer.rotateCamera(yaw: deltaX, pitch: deltaY)
        }
        
        metalView.onFlagsChanged = { [weak self] flags in
            self?.isShiftPressed = flags.contains(.shift)
        }
        
        window.contentView = metalView

        window.makeKeyAndOrderFront(nil)
        window.makeFirstResponder(metalView)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }

   func setupRenderer() {
        let renderer = self.renderer!

        registerRasterizationShaders()
        registerRaytracingShaders()
        
        // Добавляем треугольник в сцену
        Task {
            // var triangle = Triangle()
            // triangle.translate(SIMD3<Float>(1.5,0.0,0.0))
            // await renderer.addObject(objectName: "triangle", geometry: triangle, pipelineName: "coloredTriangle")
            var cube = Cube()
            cube.translate(SIMD3<Float>(-0.5,2.0,0.0))
            var cube2 = Cube()
            cube2.translate(SIMD3<Float>(-3,0.0,0.0))
            await renderer.addObject(objectName: "cube", geometry: cube, pipelineName: "coloredTriangle")
            await renderer.addObject(objectName: "cube2", geometry: cube2, pipelineName: "coloredTriangle")
            let grid = Grid()
            await renderer.addObject(objectName: "grid", geometry: grid, pipelineName: "gridUnlimited")
            
            let pointLight = PointLight(
                position: SIMD3<Float>(2, 5, 2),
                color: SIMD3<Float>(1, 1, 1),
                intensity: 5,
                softShadows: true,
                shadowConfig: ShadowConfig(samples: 8, radius: 0.2)
            )
            await renderer.addLight(name: "pointLight", light: pointLight)
            
            // let areaLight = AreaLight(position: SIMD3<Float>(5, 2, 0), color: SIMD3<Float>(1, 1, 1), intensity: 5, size: SIMD2<Float>(2, 2), direction: SIMD3<Float>(-1, 0, 0), focus: 2.0, softShadows: true)
            // await renderer.addLight(name: "areaLight", light: areaLight)
            
            // Запускаем анимацию в главном потоке
            await MainActor.run {
                self.animationTimer = Timer.scheduledTimer(withTimeInterval: 0.016, repeats: true) { [renderer] _ in
                    Task {
                        // guard let object = await renderer.getObject(objectName: "triangle") else { return }
                        // object.rotate(0.02, axis: SIMD3<Float>(0, 1, 0))
                        guard let object = await renderer.getObject(objectName: "cube") else { return }
                        object.rotate(0.02, axis: SIMD3<Float>(0, 1, 0))
                    }
                }
            }
        }
    }

    func registerRasterizationShaders() {
        let triangleShaderPath = #filePath.replacingOccurrences(of: "reijee_swift.swift", with: "shaders/shaders.metal")
        renderer.registerLibrary(libraryName: "triangle", shaderPath: triangleShaderPath)
        let gridShaderPath = #filePath.replacingOccurrences(of: "reijee_swift.swift", with: "shaders/grid.metal")
        renderer.registerLibrary(libraryName: "grid", shaderPath: gridShaderPath)

        renderer.registerPipeline(
            pipelineName: "coloredTriangle",
            libraryName: "triangle",
            vertexFunction: "vertex_main",
            fragmentFunction: "fragment_main",
            pixelFormat: .bgra8Unorm_srgb
        )

        renderer.registerPipeline(
            pipelineName: "gridUnlimited",
            libraryName: "grid",
            vertexFunction: "grid_vertex",
            fragmentFunction: "grid_fragment",
            pixelFormat: .bgra8Unorm_srgb
        )
    }

    func registerRaytracingShaders() {
        let raytracingShaderPath = #filePath.replacingOccurrences(of: "reijee_swift.swift", with: "shaders/raytracing.metal")
        renderer.registerLibrary(libraryName: "raytracing", shaderPath: raytracingShaderPath)
        let blitShaderPath = #filePath.replacingOccurrences(of: "reijee_swift.swift", with: "shaders/blit.metal")
        renderer.registerLibrary(libraryName: "blit", shaderPath: blitShaderPath)

         renderer.registerPipeline(
            pipelineName: "blit",
            libraryName: "blit",
            vertexFunction: "blit_vertex",
            fragmentFunction: "blit_fragment",
            pixelFormat: .bgra8Unorm_srgb
        )
    
        // Регистрируем compute pipeline для raytracing
        renderer.registerComputePipeline(
            pipelineName: "raytracing",
            libraryName: "raytracing",
            kernelFunction: "raytrace"
        )
    }
}
