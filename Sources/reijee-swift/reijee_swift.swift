import Cocoa
import MetalKit


@main
class AppDelegate: NSObject, NSApplicationDelegate {
    static func main() {
        let app = NSApplication.shared
        let delegate = AppDelegate()
        app.delegate = delegate
        app.run()
    }

    var window: NSWindow!
    var renderer: Renderer!

    func applicationDidFinishLaunching(_ notification: Notification) {
        let contentRect = NSRect(x: 0, y: 0, width: 1280, height: 800)
        window = NSWindow(contentRect: contentRect,
                          styleMask: [.titled, .closable, .miniaturizable, .resizable],
                          backing: .buffered,
                          defer: false)
        window.title = "Metal AppKit Beginner"


        // создаём Metal view
        let metalView = MTKView(frame: contentRect)
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        // cоздаем делегат для mtkView
        renderer = Renderer(device)

        setupRenderer()
        // назначаем device и указываем формат пикселя
        metalView.device = device
        metalView.colorPixelFormat = .bgra8Unorm_srgb
        // назначаем делегат для отрисовки (должен наследоваться протокол MTKViewDelegate)
        metalView.delegate = renderer
        metalView.enableSetNeedsDisplay = false
        metalView.isPaused = false
        metalView.preferredFramesPerSecond = 120

        window.contentView = metalView

        window.makeKeyAndOrderFront(nil)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }

    func setupRenderer() {
        let shaderPath = #filePath.replacingOccurrences(of: "reijee_swift.swift", with: "shaders.metal")
        renderer.registerLibrary(libraryName: "basic", shaderPath: shaderPath)
        
        // Регистрируем pipeline
        renderer.registerPipeline(
            pipelineName: "coloredTriangle",
            libraryName: "basic",
            vertexFunction: "vertex_main",
            fragmentFunction: "fragment_main",
            pixelFormat: .bgra8Unorm_srgb
        )
        
        // Добавляем треугольник в сцену
        let triangle = Triangle()
        renderer.addObject(objectName: "triangle", geometry: triangle, pipelineName: "coloredTriangle")
        let object = renderer.getObject(objectName: "triangle")!
        object.traslate(SIMD3<Float>(0.3, 0.1, 0.1))
        object.rotate(.pi / 2, axis: SIMD3<Float>(SIMD3<Float>(0.0, 0.0, 1.0)))
        object.scale(0.4)
    }

}
