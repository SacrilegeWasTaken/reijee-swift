import Foundation

class RwLock<T>: @unchecked Sendable {
    private var value: T
    private let queue = DispatchQueue(label: "com.reijee.rwlock", attributes: .concurrent)
    
    init(_ value: T) {
        self.value = value
    }
    
    // Чтение (много потоков одновременно)
    func read<R>(_ body: (T) -> R) -> R {
        return queue.sync {
            body(value)
        }
    }
    
    // Запись (эксклюзивный доступ)
    func write<R>(_ body: (inout T) -> R) -> R {
        return queue.sync(flags: .barrier) {
            body(&value)
        }
    }
}