//! The hearing seam: 16 kHz waveform → understanding, in-process.
//!
//! `Hearing` bundles the parity-gated audio path (feature extractor → audio
//! tower → multimodal embedder, see `gemma_audio_parity`) with the text
//! decoder and tokenizer into one warm handle. `gemma_hear` (one-shot file)
//! and `gemma_listen` (live utterance loop) both call [`Hearing::understand`];
//! neither re-implements the merge/prefill/decode dance.
//!
//! # Two handovers, one path
//!
//! [`Hearing::embed`] stops after the parity-gated part and hands back the
//! AUDIO ROWS — exactly the rows [`Hearing::understand`] then writes over the
//! audio-soft-token positions before prefill. Embeddings are the better
//! handover for a consumer that has its own decoder: text throws away tone,
//! hesitation and mood, and the throwing-away happens at the greedy argmax in
//! `understand`, which is the last place anyone can still get them back.
//! `understand` is now literally `embed` plus a chat frame plus a decode, so
//! the two cannot describe different audio.

use burn::prelude::*;
use tokenizers::Tokenizer;

use super::audio::{AudioEmbedder, AudioModel};
use super::audio_preprocess::AudioFeatureExtractor;
use super::decoder::Gemma4Model;
use crate::models::gemma::rope::RopeTable;

/// Gemma 4 special token ids for the audio chat frame (E2B/E4B tokenizer).
pub const AUDIO_SOFT_TOKEN_ID: i64 = 258881; // <audio_soft_token>
pub const BOA_TOKEN_ID: i64 = 256000; // <|audio>
pub const EOA_TOKEN_ID: i64 = 258883; // <audio|>
pub const EOS_TOKEN_ID: u32 = 1;

/// The audio rows for one utterance, already projected into the decoder's
/// text width: `n_tokens` rows of `hidden` f32, row-major. These are soft
/// tokens, not a transcript — a consumer splices them into its own token
/// embeddings the way [`Hearing::understand`] does.
#[derive(Debug, Clone)]
pub struct AudioEmbeddings {
    /// Audio soft tokens produced for this utterance.
    pub n_tokens: usize,
    /// Decoder hidden width (row stride).
    pub hidden: usize,
    /// `n_tokens * hidden` values, row-major.
    pub rows: Vec<f32>,
}

impl AudioEmbeddings {
    /// One row, or `None` past the end.
    pub fn row(&self, index: usize) -> Option<&[f32]> {
        if index >= self.n_tokens {
            return None;
        }
        Some(&self.rows[index * self.hidden..(index + 1) * self.hidden])
    }
}

/// A warm hearing stack: decoder + audio tower + embedder + tokenizer + the
/// precomputed RoPE tables. Build once, understand many utterances.
pub struct Hearing<B: Backend> {
    pub model: Gemma4Model<B>,
    pub tower: AudioModel<B>,
    pub embedder: AudioEmbedder<B>,
    pub tokenizer: Tokenizer,
    pub fe: AudioFeatureExtractor,
    rope_sliding: RopeTable<B>,
    rope_global: RopeTable<B>,
    device: B::Device,
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i;
        }
    }
    best
}

impl<B: Backend> Hearing<B> {
    pub fn new(
        model: Gemma4Model<B>,
        tower: AudioModel<B>,
        embedder: AudioEmbedder<B>,
        tokenizer: Tokenizer,
        device: B::Device,
    ) -> Self {
        let (rope_sliding, rope_global) = model.rope_tables(&device);
        Hearing {
            model,
            tower,
            embedder,
            tokenizer,
            fe: AudioFeatureExtractor::new(),
            rope_sliding,
            rope_global,
            device,
        }
    }

    /// The parity-gated audio path ONLY: 16 kHz mono f32 (≤ 30 s — longer
    /// input is truncated by the feature extractor) → log-mel → audio tower →
    /// multimodal embedder, returning the rows in the decoder's own width.
    ///
    /// This is the handover seam for a consumer that has its own decoder: the
    /// rows are what [`Hearing::understand`] writes over its audio-soft-token
    /// positions, so taking them here keeps everything a transcript discards —
    /// tone, hesitation, mood — and costs the caller nothing downstream, since
    /// splicing embeddings is the operation the model already performs.
    pub fn embed(&self, wave: &[f32]) -> AudioEmbeddings {
        let device = &self.device;
        let (feat, _mask, n_frames) = self.fe.extract(wave);
        let input_features = Tensor::<B, 1>::from_floats(&feat[..], device).reshape([
            1,
            n_frames,
            self.fe.feature_size,
        ]);
        let tower_out = self.tower.forward(input_features);
        let [_, n_tokens, multi_hidden] = tower_out.dims();
        let projected = self
            .embedder
            .forward(tower_out.reshape([n_tokens, multi_hidden]));
        let [rows_out, hidden] = projected.dims();
        debug_assert_eq!(rows_out, n_tokens);
        AudioEmbeddings {
            n_tokens,
            hidden,
            rows: projected.to_data().to_vec().unwrap(),
        }
    }

    /// Run one utterance through stt + decoder. `wave` is 16 kHz mono f32
    /// (≤ 30 s — longer input is truncated by the feature extractor). Greedy
    /// decode up to `max_new` tokens; each decoded piece is also streamed to
    /// `on_token` so callers can print as generation happens. Returns the
    /// full response text.
    pub fn understand(
        &self,
        wave: &[f32],
        prompt: &str,
        max_new: usize,
        on_token: impl FnMut(&str),
    ) -> String {
        let audio = self.embed(wave);
        self.understand_embeddings(&audio, prompt, max_new, on_token)
    }

    /// Decode audio rows produced by this hearing stack without encoding the
    /// waveform a second time. Useful when a caller needs both rows and text.
    pub fn understand_embeddings(
        &self,
        audio: &AudioEmbeddings,
        prompt: &str,
        max_new: usize,
        mut on_token: impl FnMut(&str),
    ) -> String {
        if max_new == 0 {
            return String::new();
        }
        let device = &self.device;
        let n_audio_tokens = audio.n_tokens;

        // --- Chat frame ---
        //   <bos><|turn>user\n<|audio>[audio_soft × N]<audio|>{prompt}<turn|>\n<|turn>model\n
        let pre = self
            .tokenizer
            .encode("<bos><|turn>user\n<|audio>", false)
            .unwrap();
        let post_str = format!("<audio|>{prompt}<turn|>\n<|turn>model\n");
        let post = self.tokenizer.encode(post_str.as_str(), false).unwrap();
        assert_eq!(*pre.get_ids().last().unwrap() as i64, BOA_TOKEN_ID);
        assert_eq!(post.get_ids()[0] as i64, EOA_TOKEN_ID);

        let mut ids: Vec<i64> = Vec::new();
        ids.extend(pre.get_ids().iter().map(|&x| x as i64));
        let audio_start = ids.len();
        ids.extend(std::iter::repeat(AUDIO_SOFT_TOKEN_ID).take(n_audio_tokens));
        let audio_end = ids.len();
        ids.extend(post.get_ids().iter().map(|&x| x as i64));
        let n_chat = ids.len();

        // --- Merge audio soft tokens into the input embeddings ---
        let scale = (self.model.config.hidden_size as f64).sqrt() as f32;
        let tok_i32: Vec<i32> = ids.iter().map(|&x| x as i32).collect();
        let tokens = Tensor::<B, 1, Int>::from_ints(&tok_i32[..], device).reshape([1, n_chat]);
        let mut emb = self
            .model
            .decoder
            .embed
            .forward(tokens.clone())
            .mul_scalar(scale);
        {
            let [_, _, h] = emb.dims();
            let mut d: Vec<f32> = emb.to_data().to_vec().unwrap();
            assert_eq!(
                audio.hidden, h,
                "audio rows must already be in the decoder's width"
            );
            for i in 0..n_audio_tokens {
                let off = (audio_start + i) * h;
                d[off..off + h].copy_from_slice(audio.row(i).expect("audio row"));
            }
            emb = Tensor::<B, 1>::from_floats(&d[..], device).reshape([1, n_chat, h]);
        }

        // --- Prefill + greedy decode ---
        let mut caches = self.model.new_caches();
        let l = self.model.forward_embeds(
            emb,
            tokens,
            &self.rope_sliding,
            &self.rope_global,
            &mut caches,
            &[(audio_start, audio_end)],
            None,
        );
        let [_, sl, vv] = l.dims();
        let last: Vec<f32> = l
            .slice([0..1, (sl - 1)..sl, 0..vv])
            .reshape([vv])
            .to_data()
            .to_vec()
            .unwrap();

        // Stop at end-of-sequence OR end-of-turn — greedy decode otherwise
        // rambles past the model's own `<turn|>` into a hallucinated next turn.
        let eot: u32 = self
            .tokenizer
            .encode("<turn|>", false)
            .ok()
            .and_then(|e| e.get_ids().first().copied())
            .unwrap_or(EOS_TOKEN_ID);
        let stop = |id: u32| id == EOS_TOKEN_ID || id == eot;

        let mut out = String::new();
        let mut cur = argmax(&last);
        if !stop(cur as u32) {
            let piece = self
                .tokenizer
                .decode(&[cur as u32], false)
                .unwrap_or_default();
            on_token(&piece);
            out.push_str(&piece);
            for _ in 1..max_new {
                let inp = Tensor::<B, 1, Int>::from_ints([cur as i32], device).reshape([1, 1]);
                let l = self.model.forward_cached(
                    inp,
                    &self.rope_sliding,
                    &self.rope_global,
                    &mut caches,
                );
                let [_, _, vv] = l.dims();
                let d: Vec<f32> = l.reshape([vv]).to_data().to_vec().unwrap();
                cur = argmax(&d);
                if stop(cur as u32) {
                    break;
                }
                let piece = self
                    .tokenizer
                    .decode(&[cur as u32], false)
                    .unwrap_or_default();
                on_token(&piece);
                out.push_str(&piece);
            }
        }
        out
    }
}
