pub use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Metal};

/// Full-precision backend (f32) — used for VAE and default pipeline.
pub type B = Metal;

/// Half-precision backend (f16) — used for text encoder + transformer when --f16 is set.
/// Same Metal GPU device, just stores and computes in f16 instead of f32.
pub type BHalf = Metal<half::f16>;

/// Training backend: Metal with automatic differentiation.
pub type BTrain = Autodiff<Metal>;

/// Fusion-wrapped f32 Metal backend — same GPU device, but elementwise chains
/// are JIT-fused into single kernels. Used by the qwen3tts decode loop, whose
/// cost is kernel-launch overhead, not FLOPs (voxtral's decode loop rides
/// the same alias). Declared locally (instead of
/// burn-wgpu's global `fusion` flag) so the raw `Metal` alias — and gemma's
/// zero-copy CubeTensor seam — stay untouched.
#[cfg(any(feature = "qwen3tts", feature = "voxtral"))]
pub type BFused = burn_fusion::Fusion<
    burn::backend::wgpu::CubeBackend<burn::backend::wgpu::WgpuRuntime, f32, i32, u8>,
>;

/// Half-precision sibling of [`BFused`] — same fusion wrapper, f16 storage +
/// compute on the GPU. Halves the talker's per-step weight traffic; the CPU
/// stages (code predictor, codec-head gemv) stay f32 regardless.
#[cfg(any(feature = "qwen3tts", feature = "voxtral"))]
pub type BFusedHalf = burn_fusion::Fusion<
    burn::backend::wgpu::CubeBackend<burn::backend::wgpu::WgpuRuntime, half::f16, i32, u8>,
>;

pub type FloatElem = f32;

/// The VOICE's backend family: the talker (raw, f16 by default), the speaker
/// encoder (raw) and the codec (fusion-wrapped f32), plus their device. On the
/// Mac these are the Metal aliases above; on Linux they are CUDA, because the
/// wgpu lane there runs the talker at ~164 ms a frame (2026-09-03, GB10 via
/// Vulkan: no tensor cores under it) where the Mac's Metal lane runs it at 28
/// and the same GB10 on CUDA has more to give than either. The rest of mary
/// keeps `B`/`BHalf`/`WgpuDevice` untouched; only `speak` reads these.
#[cfg(all(feature = "qwen3tts", target_os = "linux"))]
pub mod speak {
    pub use burn::backend::cuda::CudaDevice as Device;
    use burn_cubecl::cubecl;

    pub type Raw = burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime, f32, i32, u8>;
    pub type RawHalf = burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime, half::f16, i32, u8>;
    pub type Fused = burn_fusion::Fusion<Raw>;
    pub type FusedHalf = burn_fusion::Fusion<RawHalf>;
}
#[cfg(all(feature = "qwen3tts", not(target_os = "linux")))]
pub mod speak {
    pub use super::WgpuDevice as Device;
    pub type Raw = super::B;
    pub type RawHalf = super::BHalf;
    pub type Fused = super::BFused;
    pub type FusedHalf = super::BFusedHalf;
}
