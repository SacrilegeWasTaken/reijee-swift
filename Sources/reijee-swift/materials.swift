import Foundation
import simd

// CPU-side material representation mirroring the GPU layout
public struct PBRMaterial: Sendable {
	public var baseColor: SIMD3<Float>       // RGB
	public var metallic: Float               // 0..1
	public var roughness: Float              // 0..1 (perceptual)
	public var specular: Float               // 0..1 (dielectric specular level)
	public var ior: Float                    // Index of refraction for reflectance (~1.5 default)
	public var transmission: Float           // 0..1 (Refraction weight)
	public var clearcoat: Float              // 0..1
	public var clearcoatRoughness: Float     // 0..1
	public var emissiveColor: SIMD3<Float>   // RGB
	public var emissiveIntensity: Float      // cd or arbitrary units

	public init(
		baseColor: SIMD3<Float> = SIMD3<Float>(1, 1, 1),
		metallic: Float = 0.0,
		roughness: Float = 0.5,
		specular: Float = 0.5,
		ior: Float = 1.5,
		transmission: Float = 0.0,
		clearcoat: Float = 0.0,
		clearcoatRoughness: Float = 0.1,
		emissiveColor: SIMD3<Float> = SIMD3<Float>(0, 0, 0),
		emissiveIntensity: Float = 0.0
	) {
		self.baseColor = baseColor
		self.metallic = metallic
		self.roughness = roughness
		self.specular = specular
		self.ior = ior
		self.transmission = transmission
		self.clearcoat = clearcoat
		self.clearcoatRoughness = clearcoatRoughness
		self.emissiveColor = emissiveColor
		self.emissiveIntensity = emissiveIntensity
	}
}

// Simple library to store materials and hand out indices
public final class MaterialLibrary: @unchecked Sendable {
	private var materials: [PBRMaterial] = []

	public init() {
		// Add a reasonable default dielectric
		materials.append(PBRMaterial())
	}

	public func add(_ mat: PBRMaterial) -> Int {
		materials.append(mat)
		return materials.count - 1
	}

	public func update(_ index: Int, with mat: PBRMaterial) {
		guard materials.indices.contains(index) else { return }
		materials[index] = mat
	}

	public func getAll() -> [PBRMaterial] { materials }
	public func count() -> Int { materials.count }
}
