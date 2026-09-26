//! Import one Qwen3-TTS checkpoint cohort into Mary's native model collection.
//!
//! The result is ordinary model roots in one append-only pile:
//!
//! - the variant's exact base checkpoint;
//! - the exact codec checkpoint shared by both Qwen sizes;
//! - on macOS, the variant's filtered f16 talker tensors;
//! - on macOS, the variant's versioned, pre-folded f16 talker tensors.
//!
//! The last two are the Metal zero-copy layout. Every other target loads the
//! talker from the exact roots, so there the pile holds the first two only.
//!
//! The folded root is derived through the same production `Talker` load and
//! readback used by the zero-copy lane, then published with the same tensor-leaf
//! schema as every other model. There is no Repository branch, sibling pile, or
//! filename-based runtime relationship.
//!
//! Every publication is content-derived and signer-stable. If derivation is
//! interrupted, rerunning with the same checkpoint and key resumes
//! idempotently from the partial append-only cohort; changing the bytes under
//! an existing source coordinate is deliberately rejected as ambiguity.
//!
//! ```text
//! cargo run --release --features speak,import --bin qwen3tts_persist -- \
//!   <model-dir> <pile-path> <signing-key>
//! ```

mod imp {
    use mary::ingest::LeafDtype;
    #[cfg(target_os = "macos")]
    use mary::models::qwen3tts::talker::Talker;
    #[cfg(target_os = "macos")]
    use mary::nn::backend::{BFusedHalf, WgpuDevice};
    #[cfg(target_os = "macos")]
    use mary::nn::weight_loader::{AliasedPile, WeightLoader};
    #[cfg(target_os = "macos")]
    use mary::speak::QUANTIZATION_F16;
    use mary::speak::{Qwen3TtsVariant, Qwen3TtsWeights};
    use std::path::Path;
    use std::time::Instant;
    use triblespace::core::repo::pile::Pile;
    use triblespace::core::signing_key_file;
    #[cfg(target_os = "macos")]
    use triblespace::prelude::{ExclusiveId, entity};

    /// Everything the GPU talker loads, excluding the code predictor and
    /// codec-head CPU stages which deliberately remain exact f32.
    #[cfg(target_os = "macos")]
    fn is_gpu_talker_tensor(name: &str) -> bool {
        name.starts_with("talker.")
            && !name.starts_with("talker.code_predictor.")
            && name != "talker.codec_head.weight"
    }

    #[cfg(target_os = "macos")]
    fn select_index(
        snapshot: &mary::model_collection::ModelPileSnapshot,
        root: triblespace::prelude::Id,
    ) -> anyhow::Result<std::collections::HashMap<String, mary::leaf::Leaf>> {
        mary::selection::index_keymap_for_root(snapshot.facts(), snapshot.store(), root)
    }

    fn import_cohort(
        pile: &mut Pile,
        signing_key: &ed25519_dalek::SigningKey,
        model_dir: &Path,
        variant: Qwen3TtsVariant,
    ) -> anyhow::Result<()> {
        let base_source = variant.base_source();

        eprintln!("[qwen3tts] importing exact base {base_source}");
        let (base_root, _base_commit) = mary::persist::import_model_to_collection(
            pile,
            signing_key,
            model_dir,
            LeafDtype::F32,
            &base_source,
            mary::persist::QUANTIZATION_NATIVE,
        )?;

        eprintln!(
            "[qwen3tts] importing shared exact codec {}",
            Qwen3TtsVariant::codec_source()
        );
        let (codec_root, _codec_commit) = mary::persist::import_model_to_collection(
            pile,
            signing_key,
            &model_dir.join("speech_tokenizer"),
            LeafDtype::F32,
            Qwen3TtsVariant::codec_source(),
            mary::persist::QUANTIZATION_NATIVE,
        )?;

        #[cfg(target_os = "macos")]
        import_metal_talker(pile, signing_key, model_dir, variant, base_root, codec_root)?;
        #[cfg(not(target_os = "macos"))]
        {
            // Every other target loads the talker from the exact roots; the
            // f16 talker and its folded layout are Metal's zero-copy path, so
            // they are neither derived nor published here.
            let complete = mary::model_collection::snapshot_model_collection_local_latest(pile)?;
            let weights = Qwen3TtsWeights::from_snapshot(complete, variant)?;
            let (exact_count, _, _) = weights.counts();
            eprintln!(
                "[qwen3tts] native cohort valid for this target: base={base_root}, \
                 codec={codec_root}; {exact_count} exact tensors"
            );
        }
        Ok(())
    }

    /// The Metal zero-copy layout: the filtered f16 talker and its folded
    /// derivative, derived through the production Metal talker.
    #[cfg(target_os = "macos")]
    fn import_metal_talker(
        pile: &mut Pile,
        signing_key: &ed25519_dalek::SigningKey,
        model_dir: &Path,
        variant: Qwen3TtsVariant,
        base_root: triblespace::prelude::Id,
        codec_root: triblespace::prelude::Id,
    ) -> anyhow::Result<()> {
        let talker_source = variant.talker_f16_source();
        let folded_source = variant.folded_f16_source();
        eprintln!("[qwen3tts] importing filtered f16 talker {talker_source}");
        let (talker_root, _talker_commit) =
            mary::persist::import_safetensors_file_filtered_to_collection(
                pile,
                signing_key,
                &model_dir.join("model.safetensors"),
                LeafDtype::F16,
                &talker_source,
                QUANTIZATION_F16,
                is_gpu_talker_tensor,
            )?;
        mary::model_collection::publish_model_fragment(
            pile,
            signing_key,
            entity! { ExclusiveId::force_ref(&talker_root) @
                mary::format::attrs::parent: base_root,
            },
        )
        .map_err(|error| anyhow::anyhow!("publish filtered talker ancestry: {error}"))?;

        // Freeze one admitted collection cover, then bind the fold to the two
        // exact content roots returned by this invocation. Other collection
        // members, including the shared codec and stale coordinate conflicts,
        // cannot widen root-addressed selection.
        let source_snapshot = mary::model_collection::snapshot_model_collection_local_latest(pile)?;
        let exact = select_index(&source_snapshot, base_root)?;
        let talker_f16 = select_index(&source_snapshot, talker_root)?;
        drop(source_snapshot);

        eprintln!("[qwen3tts] deriving versioned folded f16 talker {folded_source}");
        let loader =
            WeightLoader::Aliased(AliasedPile::new(talker_f16, exact, WgpuDevice::default()));
        let talker = Talker::<BFusedHalf>::load(&loader, &WgpuDevice::default());
        drop(loader);
        let tensors = mary::persist::qwen3tts_folded_readback(&talker);
        drop(talker);
        let tensor_count = tensors.len();
        let tensor_bytes: usize = tensors.iter().map(|(_, bits, _)| bits.len() * 2).sum();

        let (members, facts) = mary::ingest::ingest_tensors(
            tensors.into_iter().map(|(name, bits, dims)| {
                (
                    name,
                    bits.into_iter().map(|value| value.to_f32()).collect(),
                    dims.into_iter().map(|dim| dim as usize).collect(),
                )
            }),
            pile,
            LeafDtype::F16,
        )
        .map_err(|error| anyhow::anyhow!("ingest folded talker: {error}"))?;
        let mut folded = mary::ingest::build_model_root(
            pile,
            &folded_source,
            QUANTIZATION_F16,
            members,
            facts,
            &[],
        )
        .map_err(|error| anyhow::anyhow!("build folded model root: {error}"))?;
        let folded_root = folded.root().expect("folded model root");
        folded += entity! { ExclusiveId::force_ref(&folded_root) @
            mary::format::attrs::parent*: [base_root, talker_root],
        };
        let _folded_commit =
            mary::model_collection::publish_model_fragment(pile, signing_key, folded)
                .map_err(|error| anyhow::anyhow!("publish folded model root: {error}"))?;

        // Gate the same locally admitted prefix Voice will observe, not merely
        // the four commits returned by this invocation. Stale coordinate
        // conflicts and invalid matching records therefore fail here too.
        let complete = mary::model_collection::snapshot_model_collection_local_latest(pile)?;
        let weights = Qwen3TtsWeights::from_snapshot(complete, variant)?;
        let (exact_count, f16_count, folded_count) = weights.counts();
        weights.validate_runtime_cohort()?;

        eprintln!(
            "[qwen3tts] native cohort valid: base={base_root}, codec={codec_root}, \
             talker-f16={talker_root}, folded={folded_root}"
        );
        eprintln!(
            "[qwen3tts] indexes: {exact_count} exact, {f16_count} talker f16, \
             {folded_count} folded f16; folded {:.2} GiB ({tensor_count} tensors)",
            tensor_bytes as f64 / (1_u64 << 30) as f64,
        );
        Ok(())
    }

    pub fn run() -> anyhow::Result<()> {
        let args: Vec<String> = std::env::args().collect();
        if args.len() != 4 {
            eprintln!("usage: qwen3tts_persist <model-dir> <pile-path> <signing-key>");
            std::process::exit(2);
        }
        let model_dir = Path::new(&args[1]);
        let pile_path = Path::new(&args[2]);
        let key_path = Path::new(&args[3]);
        let variant = Qwen3TtsVariant::detect(model_dir)?;
        let signing_key = signing_key_file::load_existing(key_path)?;

        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(pile_path)
        {
            Ok(_) => eprintln!("qwen3tts_persist: created new empty pile {pile_path:?}"),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.into()),
        }

        let started = Instant::now();
        let mut pile = Pile::open(pile_path)
            .map_err(|error| anyhow::anyhow!("open model pile {pile_path:?}: {error}"))?;
        let imported = import_cohort(&mut pile, &signing_key, model_dir, variant);
        let close = pile
            .close()
            .map_err(|error| anyhow::anyhow!("close model pile {pile_path:?}: {error}"));
        match (imported, close) {
            (Ok(()), Ok(())) => {}
            (Err(error), Ok(())) => return Err(error),
            (Ok(()), Err(error)) => return Err(error),
            (Err(error), Err(close_error)) => {
                return Err(
                    error.context(format!("import also failed to close pile: {close_error}"))
                );
            }
        }

        let size = std::fs::metadata(pile_path)?.len();
        println!(
            "Qwen3-TTS {variant:?} native pile {pile_path:?}: {:.2} GiB in {:.1}s",
            size as f64 / (1_u64 << 30) as f64,
            started.elapsed().as_secs_f64(),
        );
        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    imp::run()
}
