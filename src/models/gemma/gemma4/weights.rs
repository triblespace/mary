//! Load Gemma 4 weights from HuggingFace safetensors format.
//!
//! Builds model structs directly from tensors — no random init allocation.

use std::collections::HashMap;
#[cfg(feature = "import")]
use std::path::Path;

use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNormConfig};
use burn::prelude::*;
#[cfg(feature = "import")]
use memmap2::Mmap;
#[cfg(feature = "import")]
use safetensors::SafeTensors;

use super::config::Gemma4Config;
use super::decoder::{Gemma4Decoder, Gemma4Model};
use super::layers::{Gemma4Attention, Gemma4DecoderLayer, Gemma4MLP, PerLayerInput};
#[cfg(feature = "import")]
use crate::models::gemma::weights::bytes_to_f32_pub;
use crate::models::gemma::weights::{make_rms_norm, set_linear_weight};

/// A source of named tensors: either mmap'd safetensors shards or a pile-derived
/// keymap (name → (f32 data, shape)). The loader resolves names/aliases via
/// `resolve_name`/`tensor_exists` and then fetches the resolved name through
/// `get_f32`/`has` — both backends behave identically once the name is exact.
pub(crate) enum TensorSource<'a> {
    #[cfg(feature = "import")]
    Safe(HashMap<String, &'a SafeTensors<'a>>),
    Pile(HashMap<String, (Vec<f32>, Vec<usize>)>),
    /// Streaming pile load: `get` resolves one tensor's f32 data from the pile on
    /// demand (peak CPU = one tensor, not the whole keymap); `names` is the cheap
    /// handle-index keys for existence checks. Lets the 31B load from a pile.
    Stream {
        #[allow(clippy::type_complexity)]
        get: Box<dyn Fn(&str) -> Option<(Vec<f32>, Vec<usize>)> + 'a>,
        names: std::collections::HashSet<String>,
    },
}

impl<'a> TensorSource<'a> {
    /// Fetch by EXACT name (caller resolves aliases). Returns (f32 data, shape).
    fn get_f32(&self, name: &str) -> Option<(Vec<f32>, Vec<usize>)> {
        match self {
            #[cfg(feature = "import")]
            TensorSource::Safe(map) => map.values().find_map(|st| {
                st.tensor(name)
                    .ok()
                    .map(|v| (bytes_to_f32_pub(v.data(), v.dtype()), v.shape().to_vec()))
            }),
            TensorSource::Pile(map) => map.get(name).cloned(),
            TensorSource::Stream { get, .. } => get(name),
        }
    }

    /// Does a tensor with this EXACT name exist?
    fn has(&self, name: &str) -> bool {
        match self {
            #[cfg(feature = "import")]
            TensorSource::Safe(map) => map.values().any(|st| st.tensor(name).is_ok()),
            TensorSource::Pile(map) => map.contains_key(name),
            TensorSource::Stream { names, .. } => names.contains(name),
        }
    }
}

/// Resolve a tensor name across multiple prefix conventions.
/// The per-weight build strategy threaded through the loader. `has` answers raw
/// existence of an EXACT name; `get` builds the flat base tensor + shape for an
/// already-resolved EXACT name. The f32/streaming paths build via `from_floats`
/// (a copy); the aliased path maps the weight's mmap'd f16 pile blob straight
/// onto the GPU (zero-copy). Keeping the loader generic over this lets one model
/// structure serve both — the aliased builder is just Metal-monomorphized.
pub(crate) struct WeightCtx<'a, B: Backend> {
    #[allow(clippy::type_complexity)]
    pub has: Box<dyn Fn(&str) -> bool + 'a>,
    #[allow(clippy::type_complexity)]
    pub get: Box<dyn Fn(&str, &B::Device) -> Option<(Tensor<B, 1>, Vec<usize>)> + 'a>,
    /// Optional raw f32 access (present for the f32/streaming paths, absent for the
    /// aliased path). Used only where building the full GPU tensor would blow the
    /// buffer cap — notably the huge PLE table, which the f32 path slices on CPU
    /// while the aliased path slices the mmap-backed (no-alloc) tensor on GPU.
    #[allow(clippy::type_complexity)]
    pub raw: Option<Box<dyn Fn(&str) -> Option<(Vec<f32>, Vec<usize>)> + 'a>>,
}

/// Build the f32/streaming context that wraps a [`TensorSource`] — the existing
/// behavior (resolve a name, fetch f32 data, `from_floats` it).
pub(crate) fn f32_ctx<'a, B: Backend>(source: &'a TensorSource<'a>) -> WeightCtx<'a, B> {
    WeightCtx {
        has: Box::new(move |name: &str| source.has(name)),
        get: Box::new(move |name: &str, device: &B::Device| {
            let (data, shape) = source.get_f32(name)?;
            Some((Tensor::<B, 1>::from_floats(&data[..], device), shape))
        }),
        raw: Some(Box::new(move |name: &str| source.get_f32(name))),
    }
}

fn resolve_name<B: Backend>(ctx: &WeightCtx<B>, name: &str) -> String {
    for prefix in [
        "",
        "model.",
        "model.language_model.",
        "language_model.model.",
        "language_model.",
    ] {
        let full = format!("{}{}", prefix, name);
        if (ctx.has)(&full) {
            return full;
        }
        // Try with .linear.weight suffix for clipped linears (vision encoder)
        if name.ends_with(".weight") {
            let base = &name[..name.len() - 7]; // strip ".weight"
            let full = format!("{}{}.linear.weight", prefix, base);
            if (ctx.has)(&full) {
                return full;
            }
        }
    }
    panic!("Tensor not found: {name}");
}

fn tensor_exists<B: Backend>(ctx: &WeightCtx<B>, name: &str) -> bool {
    for prefix in [
        "",
        "model.",
        "model.language_model.",
        "language_model.model.",
        "language_model.",
    ] {
        let full = format!("{}{}", prefix, name);
        if (ctx.has)(&full) {
            return true;
        }
        if name.ends_with(".weight") {
            let base = &name[..name.len() - 7];
            let full = format!("{}{}.linear.weight", prefix, base);
            if (ctx.has)(&full) {
                return true;
            }
        }
    }
    false
}

/// Load a 2D tensor from the source, optionally transposing.
fn load_2d<B: Backend>(
    ctx: &WeightCtx<B>,
    name: &str,
    device: &B::Device,
    transpose: bool,
) -> Tensor<B, 2> {
    let actual = resolve_name(ctx, name);
    let (tensor, shape) = (ctx.get)(&actual, device).unwrap();
    let tensor = tensor.reshape([shape[0], shape[1]]);
    if transpose {
        tensor.swap_dims(0, 1)
    } else {
        tensor
    }
}

/// Load a 1D tensor from the source.
fn load_1d<B: Backend>(ctx: &WeightCtx<B>, name: &str, device: &B::Device) -> Tensor<B, 1> {
    let actual = resolve_name(ctx, name);
    (ctx.get)(&actual, device).unwrap().0
}

/// Build a Linear module directly from a weight tensor.
fn make_linear<B: Backend>(weight: Tensor<B, 2>, device: &B::Device) -> Linear<B> {
    let [d_in, d_out] = weight.dims();
    let mut linear = LinearConfig::new(d_in, d_out).with_bias(false).init(device);
    set_linear_weight(&mut linear, weight);
    linear
}

/// Build an Embedding module directly from a weight tensor.
fn make_embedding<B: Backend>(weight: Tensor<B, 2>, device: &B::Device) -> Embedding<B> {
    let [vocab, dim] = weight.dims();
    let embed = EmbeddingConfig::new(vocab, dim).init(device);
    let record = burn::nn::EmbeddingRecord {
        weight: burn::module::Param::from_tensor(weight),
    };
    embed.load_record(record)
}

/// Load a 3D tensor from the source (for position embedding table etc.)
fn load_3d<B: Backend>(ctx: &WeightCtx<B>, name: &str, device: &B::Device) -> Tensor<B, 3> {
    let actual = resolve_name(ctx, name);
    let (tensor, shape) = (ctx.get)(&actual, device).unwrap();
    tensor.reshape([shape[0], shape[1], shape[2]])
}

/// Load clip bounds for a clipped linear layer (returns None if not present).
fn load_clip_bounds<B: Backend>(
    ctx: &WeightCtx<B>,
    prefix: &str,
    device: &B::Device,
) -> Option<super::vision::ClipBounds> {
    // Try to find input_min/max/output_min/max scalars
    let try_scalar = |name: &str| -> Option<f32> {
        let full_names = [
            format!("{}.{}", prefix, name),
            format!("model.{}.{}", prefix, name),
            format!("model.vision_tower.{}.{}", prefix, name),
        ];
        for full in &full_names {
            if let Some((tensor, _shape)) = (ctx.get)(full, device) {
                return Some(tensor.slice([0..1]).into_scalar().elem::<f32>());
            }
        }
        None
    };

    let input_min = try_scalar("input_min")?;
    let input_max = try_scalar("input_max")?;
    let output_min = try_scalar("output_min")?;
    let output_max = try_scalar("output_max")?;

    Some(super::vision::ClipBounds {
        input_min,
        input_max,
        output_min,
        output_max,
    })
}

/// Load Gemma 4 vision encoder weights from safetensors.
pub(crate) fn load_vision_encoder<B: Backend>(
    config: &super::vision::Gemma4VisionConfig,
    _text_hidden_size: usize,
    ctx: &WeightCtx<B>,
    device: &B::Device,
) -> super::vision::Gemma4VisionEncoder<B> {
    let eps = config.rms_norm_eps;
    let head_dim = config.head_dim;

    println!(
        "  Loading vision encoder ({} layers)...",
        config.num_hidden_layers
    );

    // Patch embedder
    // Vision input_proj: NOT a clipped linear, weight stored as [768, 768]
    // Burn convention: [in, out], PyTorch: [out, in] → transpose
    let input_proj = make_linear(
        load_2d::<B>(
            ctx,
            "vision_tower.patch_embedder.input_proj.weight",
            device,
            true,
        ),
        device,
    );
    let position_embedding_table = load_3d::<B>(
        ctx,
        "vision_tower.patch_embedder.position_embedding_table",
        device,
    );

    let patch_embedder = super::vision::Gemma4PatchEmbedder {
        input_proj,
        position_embedding_table,
    };

    // Encoder layers
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        let p = format!("vision_tower.encoder.layers.{i}");

        layers.push(super::vision::Gemma4VisionLayer {
            q_proj: make_linear(
                load_2d::<B>(ctx, &format!("{p}.self_attn.q_proj.weight"), device, true),
                device,
            ),
            q_clip: load_clip_bounds(ctx, &format!("{p}.self_attn.q_proj"), device),
            k_proj: make_linear(
                load_2d::<B>(ctx, &format!("{p}.self_attn.k_proj.weight"), device, true),
                device,
            ),
            k_clip: load_clip_bounds(ctx, &format!("{p}.self_attn.k_proj"), device),
            v_proj: make_linear(
                load_2d::<B>(ctx, &format!("{p}.self_attn.v_proj.weight"), device, true),
                device,
            ),
            v_clip: load_clip_bounds(ctx, &format!("{p}.self_attn.v_proj"), device),
            o_proj: make_linear(
                load_2d::<B>(ctx, &format!("{p}.self_attn.o_proj.weight"), device, true),
                device,
            ),
            o_clip: load_clip_bounds(ctx, &format!("{p}.self_attn.o_proj"), device),
            q_norm: make_rms_norm(
                load_1d::<B>(ctx, &format!("{p}.self_attn.q_norm.weight"), device),
                eps,
                device,
            ),
            k_norm: make_rms_norm(
                load_1d::<B>(ctx, &format!("{p}.self_attn.k_norm.weight"), device),
                eps,
                device,
            ),
            v_norm: RmsNormConfig::new(head_dim).with_epsilon(eps).init(device), // no learned weights
            gate_proj: make_linear(
                load_2d::<B>(ctx, &format!("{p}.mlp.gate_proj.weight"), device, true),
                device,
            ),
            gate_clip: load_clip_bounds(ctx, &format!("{p}.mlp.gate_proj"), device),
            up_proj: make_linear(
                load_2d::<B>(ctx, &format!("{p}.mlp.up_proj.weight"), device, true),
                device,
            ),
            up_clip: load_clip_bounds(ctx, &format!("{p}.mlp.up_proj"), device),
            down_proj: make_linear(
                load_2d::<B>(ctx, &format!("{p}.mlp.down_proj.weight"), device, true),
                device,
            ),
            down_clip: load_clip_bounds(ctx, &format!("{p}.mlp.down_proj"), device),
            input_layernorm: make_rms_norm(
                load_1d::<B>(ctx, &format!("{p}.input_layernorm.weight"), device),
                eps,
                device,
            ),
            post_attention_layernorm: make_rms_norm(
                load_1d::<B>(ctx, &format!("{p}.post_attention_layernorm.weight"), device),
                eps,
                device,
            ),
            pre_feedforward_layernorm: make_rms_norm(
                load_1d::<B>(
                    ctx,
                    &format!("{p}.pre_feedforward_layernorm.weight"),
                    device,
                ),
                eps,
                device,
            ),
            post_feedforward_layernorm: make_rms_norm(
                load_1d::<B>(
                    ctx,
                    &format!("{p}.post_feedforward_layernorm.weight"),
                    device,
                ),
                eps,
                device,
            ),
            n_heads: config.num_attention_heads,
            head_dim,
        });
    }

    // Embedding projection (vision → text): norm + linear
    let embedding_pre_projection_norm = RmsNormConfig::new(config.hidden_size)
        .with_epsilon(eps)
        .init(device); // No learned weights
    let embedding_projection = make_linear(
        load_2d::<B>(
            ctx,
            "embed_vision.embedding_projection.weight",
            device,
            true,
        ),
        device,
    );

    // Standardize buffers (31B+): per-channel shift/scale applied after pool.
    let (std_bias, std_scale) = if config.standardize {
        let bias = try_resolve_1d::<B>(ctx, &["vision_tower.std_bias", "std_bias"], device);
        let scale = try_resolve_1d::<B>(ctx, &["vision_tower.std_scale", "std_scale"], device);
        (bias, scale)
    } else {
        (None, None)
    };

    println!("  Vision encoder loaded.");

    super::vision::Gemma4VisionEncoder {
        patch_embedder,
        layers,
        embedding_pre_projection_norm,
        embedding_projection,
        std_bias,
        std_scale,
        config: config.clone(),
    }
}

/// Try loading a 1D tensor by a list of candidate names (under any of the
/// vision-path prefixes); returns `None` if none match.
fn try_resolve_1d<B: Backend>(
    ctx: &WeightCtx<B>,
    bases: &[&str],
    device: &B::Device,
) -> Option<Tensor<B, 1>> {
    let prefixes = ["", "model.", "model.vision_tower.", "vision_tower."];
    for base in bases {
        for prefix in prefixes {
            let name = format!("{prefix}{base}");
            if let Some((tensor, shape)) = (ctx.get)(&name, device) {
                return Some(tensor.reshape([shape[0]]));
            }
        }
    }
    None
}

/// Load Gemma 4 text decoder weights from safetensors shards on disk.
/// Builds structs directly from tensors — no random init overhead.
///
/// Thin wrapper: mmaps the paths, builds a [`TensorSource::Safe`], and delegates
/// to [`load_gemma4_from_source`]. This frame owns the mmaps + `SafeTensors` so
/// the borrows in `Safe` stay alive across the call.
#[cfg(feature = "import")]
pub fn load_gemma4<B: Backend>(
    config: Gemma4Config,
    paths: &[&Path],
    device: &B::Device,
) -> (
    Gemma4Model<B>,
    Option<super::vision::Gemma4VisionEncoder<B>>,
) {
    let files: Vec<_> = paths
        .iter()
        .map(|p| {
            let file = std::fs::File::open(p).unwrap_or_else(|e| panic!("Can't open {:?}: {e}", p));
            unsafe { Mmap::map(&file) }.unwrap_or_else(|e| panic!("Can't mmap {:?}: {e}", p))
        })
        .collect();

    let safetensors: Vec<_> = files
        .iter()
        .map(|mmap| {
            SafeTensors::deserialize(mmap)
                .unwrap_or_else(|e| panic!("Can't parse safetensors: {e}"))
        })
        .collect();

    let mut tensors: HashMap<String, &SafeTensors<'_>> = HashMap::new();
    for st in &safetensors {
        for name in st.names() {
            tensors.insert(name.to_string(), st);
        }
    }
    println!(
        "Loading {} tensors from {} shards",
        tensors.len(),
        paths.len()
    );

    let source = TensorSource::Safe(tensors);
    let ctx = f32_ctx::<B>(&source);
    load_gemma4_from_source(config, &ctx, device)
}

/// Load a Gemma 4 model from a pile-derived keymap (`name → (f32 data, shape)`),
/// e.g. produced by [`crate::ingest::load_keymap`]. The in-substrate load path:
/// behaviorally identical to the safetensors path, just a different tensor source.
pub fn load_gemma4_from_keymap<B: Backend>(
    config: Gemma4Config,
    keymap: HashMap<String, (Vec<f32>, Vec<usize>)>,
    device: &B::Device,
) -> (
    Gemma4Model<B>,
    Option<super::vision::Gemma4VisionEncoder<B>>,
) {
    let source = TensorSource::Pile(keymap);
    let ctx = f32_ctx::<B>(&source);
    load_gemma4_from_source(config, &ctx, device)
}

/// Build the model by STREAMING tensors from a pile: given a precomputed index
/// of leaves (`crate::ingest::index_keymap`, unioned across shards), each
/// tensor's f32 data is materialized on demand and dropped after upload — peak
/// CPU is one tensor, not the whole ~120 GB f32 keymap. This is what makes
/// weights-as-tribles scale to the dense 31B.
///
/// The index costs handles and tensor headers, not weights: a leaf's payload is
/// a view over the pile's mapping.
pub fn load_gemma4_streaming<B: Backend>(
    config: Gemma4Config,
    index: HashMap<String, crate::leaf::Leaf>,
    device: &B::Device,
) -> (
    Gemma4Model<B>,
    Option<super::vision::Gemma4VisionEncoder<B>>,
) {
    let names: std::collections::HashSet<String> = index.keys().cloned().collect();
    let source = TensorSource::Stream {
        get: Box::new(move |name: &str| index.get(name).map(crate::leaf::Leaf::to_f32_shape)),
        names,
    };
    let ctx = f32_ctx::<B>(&source);
    load_gemma4_from_source(config, &ctx, device)
}

/// Build the full Gemma 4 model from a resolved [`TensorSource`] — the shared
/// body behind both [`load_gemma4`] (safetensors) and [`load_gemma4_from_keymap`]
/// (pile). All tensor access goes through `tensors` so the two paths are parity-exact.
pub(crate) fn load_gemma4_from_source<B: Backend>(
    config: Gemma4Config,
    ctx: &WeightCtx<B>,
    device: &B::Device,
) -> (
    Gemma4Model<B>,
    Option<super::vision::Gemma4VisionEncoder<B>>,
) {
    let tc = &config.text_config;
    let eps = tc.rms_norm_eps;

    // Embedding: single [vocab, hidden] tensor. 31B lands at 5.6 GiB (past
    // wgpu's default 4 GiB binding cap) — use `gaze::metal_device` to build
    // a raised-limit device before calling this.
    eprintln!("  Loading embeddings...");
    let embed = make_embedding::<B>(
        load_2d::<B>(ctx, "embed_tokens.weight", device, false),
        device,
    );

    // PLE shared embedding: [vocab_size, ple_dim * num_layers]
    // This is the largest single tensor (8.8GB f32 for E2B) — load and split per-layer.
    // Each layer gets a [vocab_size, ple_dim] slice.
    let ple_slices: Option<Vec<Tensor<B, 2>>> = if tc.has_ple() {
        eprintln!("  Loading PLE embedding (per-layer slices)...");
        let actual = resolve_name(ctx, "embed_tokens_per_layer.weight");
        let ple_dim = tc.hidden_size_per_layer_input;
        // Scale by sqrt(ple_dim) — matches Gemma4TextScaledWordEmbedding behavior.
        let ple_scale = (ple_dim as f64).sqrt() as f32;
        let slices: Vec<Tensor<B, 2>> = if let Some(raw) = &ctx.raw {
            // f32/streaming: slice the CPU f32 data into one [vocab, ple_dim]
            // tensor per layer — never materializes the full (multi-GB) table on
            // the GPU at once (which would blow the buffer cap).
            let (full_data, shape) = raw(&actual).unwrap();
            let (vocab, total_dim) = (shape[0], shape[1]);
            let n_layers = total_dim / ple_dim;
            (0..n_layers)
                .map(|l| {
                    let start = l * ple_dim;
                    let mut slice_data = Vec::with_capacity(vocab * ple_dim);
                    for v in 0..vocab {
                        let row_start = v * total_dim + start;
                        slice_data.extend_from_slice(&full_data[row_start..row_start + ple_dim]);
                    }
                    Tensor::<B, 1>::from_floats(&slice_data[..], device)
                        .reshape([vocab, ple_dim])
                        .mul_scalar(ple_scale)
                })
                .collect()
        } else {
            // aliased: the full table IS the mmap (no GPU alloc); slice each
            // layer's [vocab, ple_dim] column-block on the GPU.
            let (full, shape) = (ctx.get)(&actual, device).unwrap();
            let (vocab, total_dim) = (shape[0], shape[1]);
            let n_layers = total_dim / ple_dim;
            let full = full.reshape([vocab, total_dim]);
            (0..n_layers)
                .map(|l| {
                    let start = l * ple_dim;
                    full.clone()
                        .slice([0..vocab, start..start + ple_dim])
                        .mul_scalar(ple_scale)
                })
                .collect()
        };
        Some(slices)
    } else {
        None
    };
    // We don't store the full embedding — layers index into their slices directly.
    let _embed_per_layer: Option<Embedding<B>> = None;

    // Layers
    let mut layers = Vec::with_capacity(tc.num_hidden_layers);
    for i in 0..tc.num_hidden_layers {
        let p = format!("layers.{i}");
        eprintln!("  Loading layer {i}/{}", tc.num_hidden_layers);

        let layer_type = tc.layer_type(i);
        let (n_kv_heads, head_dim) = match layer_type {
            super::config::LayerType::SlidingAttention => (tc.num_key_value_heads, tc.head_dim),
            super::config::LayerType::FullAttention => (tc.global_kv_heads(), tc.global_head_dim()),
        };

        // With attention_k_eq_v enabled (31B+), full-attention layers omit
        // v_proj from the checkpoint — Python reuses k_proj as v. Fall
        // back to loading k_proj weights for v_proj so the attention
        // forward's `k = v.clone()` still produces the right tensor.
        let layer_k_eq_v =
            tc.attention_k_eq_v && layer_type == super::config::LayerType::FullAttention;
        let v_proj_name = format!("{p}.self_attn.v_proj.weight");
        let v_weight = if layer_k_eq_v && !tensor_exists(ctx, &v_proj_name) {
            load_2d::<B>(ctx, &format!("{p}.self_attn.k_proj.weight"), device, true)
        } else {
            load_2d::<B>(ctx, &v_proj_name, device, true)
        };
        let attention = Gemma4Attention {
            q_proj: make_linear(
                load_2d::<B>(ctx, &format!("{p}.self_attn.q_proj.weight"), device, true),
                device,
            ),
            k_proj: make_linear(
                load_2d::<B>(ctx, &format!("{p}.self_attn.k_proj.weight"), device, true),
                device,
            ),
            v_proj: make_linear(v_weight, device),
            o_proj: make_linear(
                load_2d::<B>(ctx, &format!("{p}.self_attn.o_proj.weight"), device, true),
                device,
            ),
            q_norm: make_rms_norm(
                load_1d::<B>(ctx, &format!("{p}.self_attn.q_norm.weight"), device),
                eps,
                device,
            ),
            k_norm: make_rms_norm(
                load_1d::<B>(ctx, &format!("{p}.self_attn.k_norm.weight"), device),
                eps,
                device,
            ),
            // v_norm has no learned weights — just RMSNorm with gamma=1.0 (initialized by default)
            v_norm: RmsNormConfig::new(head_dim).with_epsilon(eps).init(device),
            n_heads: tc.num_attention_heads,
            n_kv_heads,
            head_dim,
            layer_type,
            sliding_window: tc.sliding_window,
            k_eq_v: tc.attention_k_eq_v && layer_type == super::config::LayerType::FullAttention,
            softcap: tc.final_logit_softcapping,
            rope_dim: tc.rope_dim(layer_type),
            is_kv_shared: i >= tc.first_shared_kv_layer(),
        };

        let mlp = Gemma4MLP {
            gate_proj: make_linear(
                load_2d::<B>(ctx, &format!("{p}.mlp.gate_proj.weight"), device, true),
                device,
            ),
            up_proj: make_linear(
                load_2d::<B>(ctx, &format!("{p}.mlp.up_proj.weight"), device, true),
                device,
            ),
            down_proj: make_linear(
                load_2d::<B>(ctx, &format!("{p}.mlp.down_proj.weight"), device, true),
                device,
            ),
        };

        // layer_scalar is saved on every decoder layer (Python registers it
        // as a non-persistent buffer on Gemma4TextDecoderLayer, which means
        // it IS in the checkpoint even without PLE). Apply in the layer's
        // forward, not inside PLE.
        // Read dtype-agnostically: the backend element may be f16 (half-precision
        // weight path), so don't assume f32 in to_vec.
        let layer_scalar: f32 = load_1d::<B>(ctx, &format!("{p}.layer_scalar"), device)
            .slice([0..1])
            .into_scalar()
            .elem();

        let ple = if let Some(ref slices) = ple_slices {
            Some(PerLayerInput {
                embed_slice: slices[i].clone(),
                gate: make_linear(
                    load_2d::<B>(
                        ctx,
                        &format!("{p}.per_layer_input_gate.weight"),
                        device,
                        true,
                    ),
                    device,
                ),
                projection: make_linear(
                    load_2d::<B>(
                        ctx,
                        &format!("{p}.per_layer_projection.weight"),
                        device,
                        true,
                    ),
                    device,
                ),
                post_norm: make_rms_norm(
                    load_1d::<B>(
                        ctx,
                        &format!("{p}.post_per_layer_input_norm.weight"),
                        device,
                    ),
                    eps,
                    device,
                ),
                layer_scalar: 1.0, // unused now; see Gemma4DecoderLayer::forward
            })
        } else {
            None
        };

        // Optional MoE block (26B-A4B). Enabled per-model via config flag.
        let moe = if tc.has_moe() {
            Some(load_moe_block::<B>(ctx, &p, tc, device))
        } else {
            None
        };

        layers.push(Gemma4DecoderLayer {
            input_layernorm: make_rms_norm(
                load_1d::<B>(ctx, &format!("{p}.input_layernorm.weight"), device),
                eps,
                device,
            ),
            attention,
            post_attention_layernorm: make_rms_norm(
                load_1d::<B>(ctx, &format!("{p}.post_attention_layernorm.weight"), device),
                eps,
                device,
            ),
            pre_feedforward_layernorm: make_rms_norm(
                load_1d::<B>(
                    ctx,
                    &format!("{p}.pre_feedforward_layernorm.weight"),
                    device,
                ),
                eps,
                device,
            ),
            mlp,
            post_feedforward_layernorm: make_rms_norm(
                load_1d::<B>(
                    ctx,
                    &format!("{p}.post_feedforward_layernorm.weight"),
                    device,
                ),
                eps,
                device,
            ),
            ple,
            moe,
            layer_scalar,
        });
    }

    // Model-level PLE projection (E2B/E4B)
    let (per_layer_model_projection, per_layer_projection_norm, ple_proj_scale, ple_input_scale) =
        if tc.has_ple() {
            eprintln!("  Loading PLE model projection...");
            let proj = make_linear(
                load_2d::<B>(ctx, "per_layer_model_projection.weight", device, true),
                device,
            );
            let norm = make_rms_norm(
                load_1d::<B>(ctx, "per_layer_projection_norm.weight", device),
                eps,
                device,
            );
            // Scale factors: 1/sqrt(hidden_size) and 1/sqrt(2)
            let proj_scale = 1.0 / (tc.hidden_size as f64).sqrt() as f32;
            let input_scale = (0.5f64).sqrt() as f32; // 1/sqrt(2)
            (Some(proj), Some(norm), proj_scale, input_scale)
        } else {
            (None, None, 1.0, 1.0)
        };

    // Final norm
    eprintln!("  Loading output head...");
    let norm = make_rms_norm(load_1d::<B>(ctx, "norm.weight", device), eps, device);

    // LM head. When tied to the embedding we don't allocate anything new —
    // forward_inner routes through `embed.lm_head()` directly so the
    // chunked storage is reused. Only non-tied setups load a separate head.
    let lm_head = if tc.tie_word_embeddings {
        None
    } else {
        Some(make_linear(
            load_2d::<B>(ctx, "lm_head.weight", device, true),
            device,
        ))
    };

    let decoder = Gemma4Decoder {
        embed,
        embed_per_layer: None, // Per-layer slices are in each layer's PLE struct
        per_layer_model_projection,
        per_layer_projection_norm,
        per_layer_model_projection_scale: ple_proj_scale,
        per_layer_input_scale: ple_input_scale,
        layers,
        norm,
        lm_head,
    };
    // Optionally load vision encoder
    let vision_encoder = config.vision_config.as_ref().map(|vc| {
        let vision_config: super::vision::Gemma4VisionConfig =
            serde_json::from_value(vc.clone()).expect("Failed to parse vision config");
        load_vision_encoder::<B>(&vision_config, tc.hidden_size, ctx, device)
    });

    eprintln!("All Gemma 4 weights loaded.");

    (
        Gemma4Model {
            decoder,
            config: config.text_config,
        },
        vision_encoder,
    )
}

/// Build a Gemma4MoeBlock (26B-A4B) for a single decoder layer. `p` is the
/// layer prefix (`layers.{i}`); the router + experts + extra norms live
/// directly under it.
fn load_moe_block<B: Backend>(
    ctx: &WeightCtx<B>,
    p: &str,
    tc: &super::config::Gemma4TextConfig,
    device: &B::Device,
) -> super::layers::Gemma4MoeBlock<B> {
    use super::layers::{Gemma4Experts, Gemma4MoeBlock, Gemma4Router};
    let eps = tc.rms_norm_eps;
    let num_experts = tc.num_experts.expect("num_experts must be set for MoE");
    let top_k = tc.top_k_experts.expect("top_k_experts must be set for MoE");
    let moe_inter = tc
        .expert_intermediate_size
        .expect("moe intermediate size must be set");
    let h = tc.hidden_size;

    // Router: norm has no learned scale (Python's with_scale=False).
    let router_norm = RmsNormConfig::new(h).with_epsilon(eps).init(device);
    let router_proj = make_linear(
        load_2d::<B>(ctx, &format!("{p}.router.proj.weight"), device, true),
        device,
    );
    let router_scale = load_1d::<B>(ctx, &format!("{p}.router.scale"), device);
    let router_per_expert = load_1d::<B>(ctx, &format!("{p}.router.per_expert_scale"), device);

    let router = Gemma4Router {
        norm: router_norm,
        proj: router_proj,
        scale: router_scale,
        per_expert_scale: router_per_expert,
        inv_sqrt_hidden: (h as f32).powf(-0.5),
        top_k,
        num_experts,
    };

    // Expert fused tensors: [E, 2*I, H] and [E, H, I].
    let gate_up_proj = load_3d::<B>(ctx, &format!("{p}.experts.gate_up_proj"), device);
    let down_proj = load_3d::<B>(ctx, &format!("{p}.experts.down_proj"), device);
    let experts = Gemma4Experts {
        gate_up_proj,
        down_proj,
        hidden_size: h,
        intermediate_size: moe_inter,
        num_experts,
    };

    Gemma4MoeBlock {
        router,
        experts,
        post_ffn_norm_1: make_rms_norm(
            load_1d::<B>(
                ctx,
                &format!("{p}.post_feedforward_layernorm_1.weight"),
                device,
            ),
            eps,
            device,
        ),
        pre_ffn_norm_2: make_rms_norm(
            load_1d::<B>(
                ctx,
                &format!("{p}.pre_feedforward_layernorm_2.weight"),
                device,
            ),
            eps,
            device,
        ),
        post_ffn_norm_2: make_rms_norm(
            load_1d::<B>(
                ctx,
                &format!("{p}.post_feedforward_layernorm_2.weight"),
                device,
            ),
            eps,
            device,
        ),
    }
}
