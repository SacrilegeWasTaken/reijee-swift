import Cocoa
import MetalKit

class KeyboardView: MTKView {
    var onKeyDown: ((UInt16) -> Void)?
    var onKeyUp: ((UInt16) -> Void)?
    var onMouseMove: ((Float, Float) -> Void)?
    var onFlagsChanged: ((NSEvent.ModifierFlags) -> Void)?
    
    private var keyDownMonitor: Any?
    private var keyUpMonitor: Any?
    private var flagsMonitor: Any?
    
    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        
        guard window != nil else {
            if let monitor = keyDownMonitor {
                NSEvent.removeMonitor(monitor)
            }
            if let monitor = keyUpMonitor {
                NSEvent.removeMonitor(monitor)
            }
            if let monitor = flagsMonitor {
                NSEvent.removeMonitor(monitor)
            }
            return
        }
        
        keyDownMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            self?.onKeyDown?(event.keyCode)
            return nil
        }
        
        keyUpMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyUp) { [weak self] event in
            self?.onKeyUp?(event.keyCode)
            return nil
        }
        
        flagsMonitor = NSEvent.addLocalMonitorForEvents(matching: .flagsChanged) { [weak self] event in
            self?.onFlagsChanged?(event.modifierFlags)
            return event
        }
    }
    
    override func mouseDragged(with event: NSEvent) {
        let sensitivity: Float = 0.002
        onMouseMove?(Float(-event.deltaX) * sensitivity, Float(-event.deltaY) * sensitivity)
    }
}
