import Cocoa
import MetalKit


@main
@MainActor
class AppDelegate: NSObject, NSApplicationDelegate {
    static func main() {
        let app = NSApplication.shared
        let delegate = AppDelegate()
        app.delegate = delegate
        app.run()
    }

    var window: NSWindow!
    var renderer: Renderer!
    var animationTimer: Timer?

    func applicationDidFinishLaunching(_ notification: Notification) {
        let contentRect = NSRect(x: 0, y: 0, width: 800, height: 800)
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
        metalView.clearDepth = 1.0
        metalView.depthStencilPixelFormat = .depth32Float
        metalView.isPaused = false
        metalView.preferredFramesPerSecond = 120

        window.contentView = metalView

        window.makeKeyAndOrderFront(nil)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }

   func setupRenderer() {
        let shaderPath = #filePath.replacingOccurrences(of: "reijee_swift.swift", with: "shaders/shaders.metal")
        renderer.registerLibrary(libraryName: "basic", shaderPath: shaderPath)
        
        // Регистрируем pipeline
        renderer.registerPipeline(
            pipelineName: "coloredTriangle",
            libraryName: "basic",
            vertexFunction: "vertex_main",
            fragmentFunction: "fragment_main",
            pixelFormat: .bgra8Unorm_srgb
        )
        
        let renderer = self.renderer!
        // Добавляем треугольник в сцену
        Task {
            var triangle = Triangle()
            // triangle.traslate(SIMD3<Float>(0,0,0.5))
            await renderer.addObject(objectName: "triangle", geometry: triangle, pipelineName: "coloredTriangle")
            
            // Запускаем анимацию в главном потоке
            await MainActor.run {
                self.animationTimer = Timer.scheduledTimer(withTimeInterval: 0.016, repeats: true) { [renderer] _ in
                    Task {
                        guard let object = await renderer.getObject(objectName: "triangle") else { return }
                        object.rotate(0.02, axis: SIMD3<Float>(0, 1, 0))
                    }
                }
            }
        }
    }


}
