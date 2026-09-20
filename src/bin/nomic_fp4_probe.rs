//! How much does FP4 cost an embedding index? A night probe, 2026-09-11.
//!
//! Two questions JP asked: could the nomic text model run in NVFP4, and could
//! the output vectors be NVFP4 too, plain or as the sixteen-lane fractional
//! code the Inkling carrier pilot used. This binary measures both on our own
//! prose, the wiki fragments and journal summaries of the self pile, against
//! the f32 model and f32 vectors, by top-10 neighbour recall and cosine error.
//!
//! ```text
//! nomic_fp4_probe extract --pile <self.pile> --wiki <handle> --journal <handle> --out corpus.jsonl
//! nomic_fp4_probe probe --model <nomic_text.pile> --corpus corpus.jsonl [--queries 200]
//! ```
//!
//! Everything numerical about NVFP4 here is a CPU reference: E2M1 codes
//! {0, 0.5, 1, 1.5, 2, 3, 4, 6}, E4M3 block scales over sixteen elements, one
//! f32 scale per vector or per weight row. It is a measurement of rounding,
//! not of any kernel.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use triblespace::prelude::*;

// ── NVFP4 reference arithmetic ───────────────────────────────────────────

const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
const BLOCK: usize = 16;
const E4M3_MAX: f32 = 448.0;

/// Nearest E2M1 magnitude code for `m >= 0`, ties to the even code.
fn e2m1_code(m: f32) -> usize {
    let mut best = 0usize;
    let mut best_err = f32::INFINITY;
    for (code, value) in E2M1.iter().enumerate() {
        let err = (m - value).abs();
        if err < best_err || (err == best_err && code % 2 == 0) {
            best = code;
            best_err = err;
        }
    }
    best
}

/// Round a positive scale to E4M3: three mantissa bits, exponents 2^-6..2^8,
/// subnormals in steps of 2^-9, saturating at 448.
fn e4m3(v: f32) -> f32 {
    if !(v > 0.0) {
        return 0.0;
    }
    if v >= E4M3_MAX {
        return E4M3_MAX;
    }
    let e = v.log2().floor().max(-6.0);
    let step = 2f32.powf(e - 3.0);
    let rounded = (v / step).round() * step;
    rounded.min(E4M3_MAX)
}

/// One NVFP4 vector: per-vector f32 scale, E4M3 block scales, E2M1 codes.
/// Returns the dequantized values and, per element, the lower and upper E2M1
/// neighbours in the same block scaling (for the sixteen-lane code).
fn nvfp4_quantize(x: &[f32]) -> (Vec<f32>, Vec<(f32, f32)>) {
    nvfp4_quantize_with(x, false)
}

/// Candidate shrink factors for the per-block scale search: the block
/// maximum maps to 6 at 1.0 and saturates below it, trading the largest
/// element's error for a finer grid under everything else.
const SCALE_SEARCH: [f32; 9] = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6];

/// Squared error of one block quantized at `unit` per E2M1 step.
fn block_error(block: &[f32], unit: f32) -> f32 {
    block
        .iter()
        .map(|&v| {
            let m = (v.abs() / unit).min(6.0);
            let q = E2M1[e2m1_code(m)] * unit * v.signum();
            (v - q) * (v - q)
        })
        .sum()
}

/// `search` picks each block's E4M3 scale by least squared error over the
/// candidates in `SCALE_SEARCH` instead of always mapping the block maximum
/// to the top code.
fn nvfp4_quantize_with(x: &[f32], search: bool) -> (Vec<f32>, Vec<(f32, f32)>) {
    let absmax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let tensor_scale = if absmax > 0.0 {
        absmax / (6.0 * E4M3_MAX)
    } else {
        1.0
    };
    let mut out = Vec::with_capacity(x.len());
    let mut neighbours = Vec::with_capacity(x.len());
    for block in x.chunks(BLOCK) {
        let block_max = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let nearest = e4m3(block_max / (6.0 * tensor_scale));
        let scale = if search && nearest > 0.0 {
            let mut best = nearest;
            let mut best_err = block_error(block, nearest * tensor_scale);
            for factor in SCALE_SEARCH.iter().skip(1) {
                let candidate = e4m3(block_max * factor / (6.0 * tensor_scale));
                if candidate <= 0.0 || candidate == best {
                    continue;
                }
                let err = block_error(block, candidate * tensor_scale);
                if err < best_err {
                    best = candidate;
                    best_err = err;
                }
            }
            best
        } else {
            nearest
        };
        let unit = scale * tensor_scale;
        for &v in block {
            if unit == 0.0 {
                out.push(0.0);
                neighbours.push((0.0, 0.0));
                continue;
            }
            let m = (v.abs() / unit).min(6.0);
            let code = e2m1_code(m);
            let q = E2M1[code] * unit * v.signum();
            out.push(q);
            // Neighbours bracketing m on the E2M1 ladder.
            let lo = E2M1.iter().rev().find(|&&l| l <= m).copied().unwrap_or(0.0);
            let hi = E2M1.iter().find(|&&h| h >= m).copied().unwrap_or(6.0);
            neighbours.push((lo * unit * v.signum(), hi * unit * v.signum()));
        }
    }
    (out, neighbours)
}

/// Sixteen FP4 lanes: for each element, k of sixteen lanes take the upper
/// E2M1 neighbour and the rest the lower one, so the lane mean lands within
/// one sixteenth of the step. Storage is 16 x 4 bits per element.
fn fp4_lanes16(x: &[f32]) -> Vec<f32> {
    let (_, neighbours) = nvfp4_quantize(x);
    x.iter()
        .zip(neighbours)
        .map(|(&v, (lo, hi))| {
            if hi == lo {
                return lo;
            }
            let f = ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
            let k = (f * 16.0).round();
            lo + (hi - lo) * k / 16.0
        })
        .collect()
}

/// Two-stage residual NVFP4: quantize, then quantize the residual with its
/// own scales, and add. This is the shape of mary's QuantizedRow, done here
/// in the same reference arithmetic so the comparison is like for like.
fn nvfp4_two_stage(x: &[f32]) -> Vec<f32> {
    let (first, _) = nvfp4_quantize(x);
    let residual: Vec<f32> = x.iter().zip(&first).map(|(v, q)| v - q).collect();
    let (second, _) = nvfp4_quantize(&residual);
    first.iter().zip(second).map(|(a, b)| a + b).collect()
}

fn int8(x: &[f32]) -> Vec<f32> {
    let absmax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    if absmax == 0.0 {
        return x.to_vec();
    }
    let unit = absmax / 127.0;
    x.iter().map(|v| (v / unit).round() * unit).collect()
}

fn binary(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|v| if *v >= 0.0 { 1.0 } else { -1.0 })
        .collect()
}

fn l2_normalize(x: &mut [f32]) {
    let n = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    if n > 0.0 {
        for v in x.iter_mut() {
            *v /= n;
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// ── corpus ───────────────────────────────────────────────────────────────

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn json_unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('t') => out.push('\t'),
            Some('"') => out.push('"'),
            Some('\\') => out.push('\\'),
            Some('u') => {
                let hex: String = chars.by_ref().take(4).collect();
                if let Some(ch) = u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
                    out.push(ch);
                }
            }
            Some(other) => out.push(other),
            None => {}
        }
    }
    out
}

/// Read `{"id": "...", "source": "...", "text": "..."}` lines written by `extract`.
fn read_corpus(path: &Path) -> Result<Vec<(String, String, String)>> {
    let mut rows = Vec::new();
    for line in fs::read_to_string(path)?.lines() {
        let field = |name: &str| -> Option<String> {
            let key = format!("\"{name}\": \"");
            let start = line.find(&key)? + key.len();
            let rest = &line[start..];
            // The value ends at the first unescaped quote.
            let mut end = 0;
            let bytes = rest.as_bytes();
            while end < bytes.len() {
                if bytes[end] == b'\\' {
                    end += 2;
                    continue;
                }
                if bytes[end] == b'"' {
                    break;
                }
                end += 1;
            }
            Some(json_unescape(&rest[..end]))
        };
        if let (Some(id), Some(source), Some(text)) = (field("id"), field("source"), field("text"))
        {
            rows.push((id, source, text));
        }
    }
    Ok(rows)
}

fn extract(
    pile: &Path,
    sources: &[(String, [u8; 16], String)],
    out: &Path,
    max_chars: usize,
) -> Result<()> {
    use anybytes::View;
    use triblespace::core::blob::encodings::simplearchive::SimpleArchive;
    use triblespace::core::blob::encodings::utf8string::UTF8String;
    use triblespace::core::blob::{Blob, IntoBlob, TryFromBlob};
    use triblespace::core::collection::records::CollectionHandle;
    use triblespace::core::collection::{Collection, CollectionSnapshotExt};
    use triblespace::core::inline::Inline;
    use triblespace::core::inline::encodings::hash::Handle;
    use triblespace::core::repo::pile::Pile;
    use triblespace::core::repo::{BlobStoreGet, SnapshotSource};
    use triblespace::core::trible::{TRIBLE_LEN, TribleSet};

    let mut pile = Pile::open(pile).map_err(|e| anyhow!("open pile: {e:?}"))?;
    let snapshot = pile.snapshot().map_err(|e| anyhow!("snapshot: {e:?}"))?;
    let mut lines = Vec::new();
    for (name, attribute, handle_hex) in sources {
        let mut raw = [0u8; 32];
        hex_decode(handle_hex, &mut raw)?;
        let handle = CollectionHandle::new(raw);
        let collection: Collection<SimpleArchive> = Collection::open(&snapshot, handle)
            .map_err(|e| anyhow!("open collection {name}: {e}"))?;
        let facts: TribleSet = snapshot
            .collection(collection)
            .map_err(|e| anyhow!("attach {name}: {e:?}"))?
            .view::<TribleSet>()
            .map_err(|e| anyhow!("view {name}: {e:?}"))?;
        let blob: Blob<SimpleArchive> = facts.to_blob();
        let mut seen = std::collections::BTreeSet::new();
        let mut count = 0usize;
        for trible in blob.bytes.as_ref().chunks_exact(TRIBLE_LEN) {
            if trible[16..32] != attribute[..] {
                continue;
            }
            let value: [u8; 32] = trible[32..].try_into().unwrap();
            if !seen.insert(value) {
                continue;
            }
            let text_blob: Blob<UTF8String> =
                match snapshot.get(Inline::<Handle<UTF8String>>::new(value)) {
                    Ok(blob) => blob,
                    Err(_) => continue,
                };
            let text: View<str> = match View::try_from_blob(text_blob) {
                Ok(view) => view,
                Err(_) => continue,
            };
            let mut text: String = text.to_string();
            if text.trim().len() < 40 {
                continue;
            }
            if text.len() > max_chars {
                let mut cut = max_chars;
                while !text.is_char_boundary(cut) {
                    cut -= 1;
                }
                text.truncate(cut);
            }
            let id: String = value.iter().map(|b| format!("{b:02x}")).collect();
            lines.push(format!(
                "{{\"id\": \"{id}\", \"source\": \"{name}\", \"text\": \"{}\"}}",
                json_escape(&text)
            ));
            count += 1;
        }
        eprintln!("{name}: {count} texts");
    }
    fs::write(out, lines.join("\n") + "\n")?;
    eprintln!("wrote {} texts to {}", lines.len(), out.display());
    Ok(())
}

fn hex_decode(hex: &str, out: &mut [u8]) -> Result<()> {
    let hex = hex.trim().trim_start_matches("blake3:");
    if hex.len() != out.len() * 2 {
        return Err(anyhow!(
            "expected {} hex digits, got {}",
            out.len() * 2,
            hex.len()
        ));
    }
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16)?;
    }
    Ok(())
}

// ── model ────────────────────────────────────────────────────────────────

const NOMIC_TEXT_MODEL: &str = "nomic-ai/nomic-embed-text-v1.5";

type Keymap = HashMap<String, (Vec<f32>, Vec<usize>)>;

fn load_parts(
    model_pile: &Path,
    quantization: &str,
) -> Result<(Keymap, tokenizers::Tokenizer, Vec<Id>)> {
    use mary::selection::{ModelSelector, TokenizerSelector};
    let snapshot = mary::model_collection::load_model_collection_local_latest(model_pile)
        .with_context(|| format!("open model pile {}", model_pile.display()))?;
    let selector = ModelSelector::Source {
        source: NOMIC_TEXT_MODEL,
        quantization,
    };
    let roots = mary::selection::select_model_roots(snapshot.facts(), snapshot.store(), selector)
        .context("select nomic text source roots")?;
    let keymap =
        mary::selection::load_keymap_from_graph(snapshot.facts(), snapshot.store(), selector)
            .context("select native nomic text weights")?;
    let tokenizer = mary::selection::load_tokenizer_from_graph(
        snapshot.facts(),
        snapshot.store(),
        TokenizerSelector::Name(NOMIC_TEXT_MODEL),
    )
    .context("select nomic tokenizer")?;
    Ok((keymap, tokenizer, roots))
}

/// Fake-quantize every two-dimensional weight that is not an embedding table
/// or a norm: rows are output features, blocks of sixteen run along the input
/// features, one f32 scale per row. Returns how many tensors were touched.
fn fake_nvfp4_weights(keymap: &mut Keymap, only: &[String], scale_search: bool) -> (usize, usize) {
    let mut tensors = 0usize;
    let mut elements = 0usize;
    for (name, (data, shape)) in keymap.iter_mut() {
        let lower = name.to_ascii_lowercase();
        if shape.len() != 2 || !lower.contains("weight") {
            continue;
        }
        if !only.is_empty()
            && !only
                .iter()
                .any(|pattern| lower.contains(&pattern.to_ascii_lowercase()))
        {
            continue;
        }
        if lower.contains("embed") || lower.contains("norm") || lower.contains("ln") {
            continue;
        }
        let cols = shape[1];
        if cols % BLOCK != 0 {
            continue;
        }
        for row in data.chunks_mut(cols) {
            let (q, _) = nvfp4_quantize_with(row, scale_search);
            row.copy_from_slice(&q);
        }
        tensors += 1;
        elements += data.len();
    }
    (tensors, elements)
}

/// Candidate exponents for the activation-aware scale `s = mean|x|^alpha`.
const AWQ_ALPHAS: [f32; 10] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

/// Output error of a quantised weight on real inputs: sum over reservoir
/// rows of |(W - Wq) x|^2, the quantity AWQ minimises.
fn output_error(w: &[f32], wq: &[f32], cols: usize, rows: &[Vec<f32>]) -> f64 {
    let diff: Vec<f32> = w.iter().zip(wq).map(|(a, b)| a - b).collect();
    let threads = std::thread::available_parallelism()
        .map_or(8, |n| n.get())
        .min(32)
        .max(1);
    let chunk = rows.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = rows
            .chunks(chunk)
            .map(|part| {
                let diff = &diff;
                scope.spawn(move || {
                    let mut err = 0f64;
                    for x in part {
                        for row in diff.chunks(cols) {
                            let acc: f32 = row.iter().zip(x).map(|(d, xi)| d * xi).sum();
                            err += (acc as f64) * (acc as f64);
                        }
                    }
                    err
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("error worker"))
            .sum()
    })
}

/// Cholesky factor L (lower, row-major n x n) of a symmetric positive
/// definite matrix; panics on a non-positive pivot, which the dampening
/// below prevents.
fn cholesky(a: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                assert!(sum > 0.0, "cholesky: non-positive pivot at {i}: {sum}");
                l[i * n + i] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    l
}

/// Inverse of a symmetric positive definite matrix from its Cholesky factor,
/// one column of the identity per solve, columns spread over threads.
fn spd_inverse(l: &[f64], n: usize) -> Vec<f64> {
    let threads = std::thread::available_parallelism()
        .map_or(8, |v| v.get())
        .min(32)
        .max(1);
    let chunk = n.div_ceil(threads).max(1);
    let mut inv = vec![0f64; n * n];
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..n)
            .step_by(chunk)
            .map(|start| {
                let end = (start + chunk).min(n);
                scope.spawn(move || {
                    let mut cols = Vec::with_capacity((end - start) * n);
                    let mut y = vec![0f64; n];
                    for c in start..end {
                        // L y = e_c
                        for i in 0..n {
                            let mut sum = if i == c { 1.0 } else { 0.0 };
                            for k in 0..i {
                                sum -= l[i * n + k] * y[k];
                            }
                            y[i] = sum / l[i * n + i];
                        }
                        // L^T x = y
                        let mut x = vec![0f64; n];
                        for i in (0..n).rev() {
                            let mut sum = y[i];
                            for k in i + 1..n {
                                sum -= l[k * n + i] * x[k];
                            }
                            x[i] = sum / l[i * n + i];
                        }
                        cols.extend_from_slice(&x);
                    }
                    (start, cols)
                })
            })
            .collect();
        for h in handles {
            let (start, cols) = h.join().expect("inverse worker");
            for (c, col) in cols.chunks(n).enumerate() {
                for i in 0..n {
                    inv[i * n + start + c] = col[i];
                }
            }
        }
    });
    inv
}

/// Error-feedback rounding (GPTQ) of one weight `w` [rows x cols] to NVFP4,
/// against input rows `x` [.. x cols]: columns are rounded left to right and
/// each column's rounding error is pushed onto the columns still to come
/// through the inverse Hessian of the inputs, so the layer's output error is
/// what gets minimised rather than the weight error. Block scales are fixed
/// per row when a block of sixteen columns is reached, from the values the
/// feedback has left there; the per-row f32 scale comes from the row's
/// original maximum. Returns the dequantised weight.
fn gptq_quantize(w: &[f32], cols: usize, x: &[Vec<f32>]) -> Vec<f32> {
    let rows = w.len() / cols;
    let n = cols;
    // H = X^T X / R, dampened by one percent of its mean diagonal.
    let mut h = vec![0f64; n * n];
    for r in x {
        for i in 0..n {
            let xi = r[i] as f64;
            if xi == 0.0 {
                continue;
            }
            let row = &mut h[i * n..(i + 1) * n];
            for (hij, &xj) in row.iter_mut().zip(r.iter()) {
                *hij += xi * xj as f64;
            }
        }
    }
    let scale = 1.0 / x.len().max(1) as f64;
    for v in h.iter_mut() {
        *v *= scale;
    }
    let mean_diag = (0..n).map(|i| h[i * n + i]).sum::<f64>() / n as f64;
    let damp = 0.01 * mean_diag.max(1e-12);
    for i in 0..n {
        h[i * n + i] += damp;
    }
    let l = cholesky(&h, n);
    let hinv = spd_inverse(&l, n);
    // U = chol(Hinv)^T, upper; the feedback uses its rows.
    let lu = cholesky(&hinv, n);
    let mut u = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            u[j * n + i] = lu[i * n + j];
        }
    }
    let ts: Vec<f32> = w
        .chunks(cols)
        .map(|row| {
            let absmax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
            if absmax > 0.0 {
                absmax / (6.0 * E4M3_MAX)
            } else {
                1.0
            }
        })
        .collect();
    let threads = std::thread::available_parallelism()
        .map_or(8, |v| v.get())
        .min(32)
        .max(1);
    let chunk = rows.div_ceil(threads).max(1);
    let mut out = vec![0f32; w.len()];
    std::thread::scope(|scope| {
        let handles: Vec<_> = w
            .chunks(chunk * cols)
            .enumerate()
            .map(|(ci, part)| {
                let u = &u;
                let ts = &ts[ci * chunk..];
                scope.spawn(move || {
                    let mut work: Vec<f32> = part.to_vec();
                    let mut q = vec![0f32; part.len()];
                    let nrows = part.len() / cols;
                    let mut units = vec![0f32; nrows];
                    for j in 0..cols {
                        if j % BLOCK == 0 {
                            for r in 0..nrows {
                                let block = &work[r * cols + j..r * cols + j + BLOCK];
                                let bmax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
                                units[r] = e4m3(bmax / (6.0 * ts[r])) * ts[r];
                            }
                        }
                        let ujj = u[j * n + j];
                        for r in 0..nrows {
                            let v = work[r * cols + j];
                            let unit = units[r];
                            let qv = if unit == 0.0 {
                                0.0
                            } else {
                                let m = (v.abs() / unit).min(6.0);
                                E2M1[e2m1_code(m)] * unit * v.signum()
                            };
                            q[r * cols + j] = qv;
                            let err = ((v - qv) as f64 / ujj) as f32;
                            if err != 0.0 {
                                let urow = &u[j * n + j + 1..(j + 1) * n];
                                let wrow = &mut work[r * cols + j + 1..(r + 1) * cols];
                                for (wk, &ujk) in wrow.iter_mut().zip(urow) {
                                    *wk -= err * ujk as f32;
                                }
                            }
                        }
                    }
                    (ci, q)
                })
            })
            .collect();
        for h in handles {
            let (ci, q) = h.join().expect("gptq worker");
            out[ci * chunk * cols..ci * chunk * cols + q.len()].copy_from_slice(&q);
        }
    });
    out
}

/// One tensor's activation-aware NVFP4: scale each input channel by
/// `mean|x|^alpha` (geometric mean one) before rounding and divide it back
/// after, so the rounding grid follows the channels the inputs actually
/// exercise; the alpha with the least output error on the captured rows wins.
/// Returns the chosen alpha and the error ratio against plain rounding.
fn awq_quantize_tensor(
    data: &mut [f32],
    cols: usize,
    stats: &mary::embed::NomicActStats,
    scale_search: bool,
    gptq: bool,
) -> (f32, f64) {
    let mean_abs = stats.mean_abs();
    let original = data.to_vec();
    let plain: Vec<f32> = {
        let mut w = original.clone();
        for row in w.chunks_mut(cols) {
            let (q, _) = nvfp4_quantize_with(row, scale_search);
            row.copy_from_slice(&q);
        }
        w
    };
    let plain_err = output_error(&original, &plain, cols, &stats.rows);
    let mut best = (0f32, plain_err, plain);
    for &alpha in AWQ_ALPHAS.iter().skip(1) {
        let mut s: Vec<f32> = mean_abs.iter().map(|m| (m + 1e-8).powf(alpha)).collect();
        let log_mean = s.iter().map(|v| v.ln() as f64).sum::<f64>() / s.len() as f64;
        let norm = log_mean.exp() as f32;
        for v in &mut s {
            *v /= norm;
        }
        let mut w = original.clone();
        for row in w.chunks_mut(cols) {
            for (v, si) in row.iter_mut().zip(&s) {
                *v *= si;
            }
            let (q, _) = nvfp4_quantize_with(row, scale_search);
            for ((v, qi), si) in row.iter_mut().zip(q).zip(&s) {
                *v = qi / si;
            }
        }
        let err = output_error(&original, &w, cols, &stats.rows);
        if err < best.1 {
            best = (alpha, err, w);
        }
    }
    if gptq {
        // The chosen alpha's scales, then error-feedback rounding in the
        // scaled space against the correspondingly scaled inputs.
        let alpha = best.0;
        let mut s: Vec<f32> = mean_abs.iter().map(|m| (m + 1e-8).powf(alpha)).collect();
        let log_mean = s.iter().map(|v| v.ln() as f64).sum::<f64>() / s.len() as f64;
        let norm = log_mean.exp() as f32;
        for v in &mut s {
            *v /= norm;
        }
        let mut scaled = original.clone();
        for row in scaled.chunks_mut(cols) {
            for (v, si) in row.iter_mut().zip(&s) {
                *v *= si;
            }
        }
        let xs: Vec<Vec<f32>> = stats
            .rows
            .iter()
            .map(|r| r.iter().zip(&s).map(|(xi, si)| xi / si).collect())
            .collect();
        let mut q = gptq_quantize(&scaled, cols, &xs);
        for row in q.chunks_mut(cols) {
            for (v, si) in row.iter_mut().zip(&s) {
                *v /= si;
            }
        }
        let err = output_error(&original, &q, cols, &stats.rows);
        if err < best.1 {
            best = (alpha, err, q);
        } else {
            eprintln!(
                "    (feedback rounding did not improve on nearest: x{:.3} vs x{:.3})",
                err / plain_err.max(1e-30),
                best.1 / plain_err.max(1e-30)
            );
        }
    }
    data.copy_from_slice(&best.2);
    (
        best.0,
        if plain_err > 0.0 {
            best.1 / plain_err
        } else {
            1.0
        },
    )
}

/// Activation-aware fake NVFP4 over the keymap: every linear that has
/// captured input statistics is quantised with its best alpha, the rest
/// plainly. Prints one line per tensor.
fn fake_nvfp4_weights_awq(
    keymap: &mut Keymap,
    only: &[String],
    scale_search: bool,
    gptq: bool,
    embeddings: bool,
    stats: &std::collections::HashMap<String, mary::embed::NomicActStats>,
) -> (usize, usize, HashMap<String, mary::calibrate::Calibrated>) {
    if scale_search {
        eprintln!(
            "  (block scale search is ignored under calibration: it lost to nearest rounding there)"
        );
    }
    let opts = mary::calibrate::Options {
        alphas: &mary::calibrate::AWQ_ALPHAS,
        feedback: gptq,
        embeddings,
    };
    let inputs: HashMap<String, mary::calibrate::InputStats> = stats
        .iter()
        .map(|(k, s)| {
            (
                k.clone(),
                mary::calibrate::InputStats {
                    rows: s.rows.clone(),
                    mean_abs: s.mean_abs(),
                },
            )
        })
        .collect();
    let stats_for = |name: &str| inputs.get(&mary::calibrate::nomic_capture_key(name));
    let report = mary::calibrate::pack_keymap(keymap, only, &stats_for, &opts, &mut |line| {
        eprintln!("  {line}")
    })
    .unwrap_or_else(|e| panic!("pack: {e}"));
    (report.tensors, report.elements, report.packed)
}

// ── vision: the packed vision model against the f32 one, image to image ────

const NOMIC_VISION_MODEL: &str = "nomic-ai/nomic-embed-vision-v1.5";

fn vision_keymap(pile: &Path, quantization: &str) -> Result<Keymap> {
    let snapshot = mary::model_collection::load_model_collection_local_latest(pile)
        .with_context(|| format!("open model pile {}", pile.display()))?;
    mary::selection::load_keymap_from_graph(
        snapshot.facts(),
        snapshot.store(),
        mary::selection::ModelSelector::Source {
            source: NOMIC_VISION_MODEL,
            quantization,
        },
    )
    .with_context(|| {
        format!(
            "select {quantization} nomic vision weights from {}",
            pile.display()
        )
    })
}

fn image_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .with_context(|| format!("read {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .collect();
    files.sort();
    Ok(files)
}

/// Embed every image in `dir` with one vision model; unreadable images are
/// skipped and named. Returns the vectors in `files` order with the index of
/// each file that embedded.
fn embed_images(
    embedder: &mary::embed::NomicVisionEmbedder<mary::nn::backend::B>,
    files: &[PathBuf],
) -> Result<Vec<(usize, Vec<f32>)>> {
    use mary::embed::LocalEmbedder;
    let mut out = Vec::with_capacity(files.len());
    for (i, f) in files.iter().enumerate() {
        let bytes = fs::read(f).with_context(|| format!("read {}", f.display()))?;
        match embedder.embed_image(&bytes) {
            Ok(mut v) => {
                l2_normalize(&mut v);
                out.push((i, v));
            }
            Err(e) => eprintln!("  skip {}: {e}", f.display()),
        }
    }
    Ok(out)
}

/// Score a packed vision pile against the f32 one over a directory of images:
/// mean cosine between the two models' vectors of the same image, and
/// image-to-image recall@k, each image a query against the rest, the f32
/// model's top k as the baseline.
fn vision(
    model: &Path,
    quantization: &str,
    packed: &Path,
    packed_quantization: &str,
    images: &Path,
) -> Result<()> {
    let files = image_files(images)?;
    anyhow::ensure!(!files.is_empty(), "no images under {}", images.display());
    let device = mary::embed::default_device();
    let started = Instant::now();
    let base = {
        let emb = mary::embed::load_nomic_vision_from_keymap(
            vision_keymap(model, quantization)?,
            device.clone(),
        )?;
        embed_images(&emb, &files)?
    };
    eprintln!(
        "f32 model embedded {} of {} images in {:.1} s",
        base.len(),
        files.len(),
        started.elapsed().as_secs_f64()
    );
    let started = Instant::now();
    let cand = {
        let emb = mary::embed::load_nomic_vision_from_keymap(
            vision_keymap(packed, packed_quantization)?,
            device,
        )?;
        embed_images(&emb, &files)?
    };
    eprintln!(
        "packed model embedded {} of {} images in {:.1} s",
        cand.len(),
        files.len(),
        started.elapsed().as_secs_f64()
    );
    anyhow::ensure!(
        base.iter()
            .map(|(i, _)| *i)
            .eq(cand.iter().map(|(i, _)| *i)),
        "the two models did not embed the same images"
    );
    let a: Vec<Vec<f32>> = base.into_iter().map(|(_, v)| v).collect();
    let b: Vec<Vec<f32>> = cand.into_iter().map(|(_, v)| v).collect();
    let n = a.len();
    let mean_cos: f64 = a.iter().zip(&b).map(|(x, y)| dot(x, y) as f64).sum::<f64>() / n as f64;
    println!(
        "nomic-embed-vision-v1.5 over {n} images: packed ({packed_quantization}) against f32 ({quantization})"
    );
    println!("mean cosine between the two models' vectors of the same image: {mean_cos:.5}");
    for k in [1usize, 5, 10] {
        if n <= k {
            continue;
        }
        let mut recall = 0f64;
        let mut top1_same = 0usize;
        for i in 0..n {
            let base_top = top_k(&a[i], &a, Some(i), k);
            let cand_top = top_k(&b[i], &b, Some(i), k);
            recall += recall_at_k(&base_top, &cand_top) as f64;
            if base_top.first() == cand_top.first() {
                top1_same += 1;
            }
        }
        println!(
            "image-to-image recall@{k}: {:.1}%   (same nearest image: {}/{n})",
            100.0 * recall / n as f64,
            top1_same
        );
    }
    Ok(())
}

/// Print mary's vectors for a few texts (query side) and image files as JSON,
/// to set beside a reference implementation's vectors of the same inputs.
fn dump(
    text_pile: &Path,
    text_quantization: &str,
    vision_pile: &Path,
    vision_quantization: &str,
    texts: &[String],
    images: &[PathBuf],
) -> Result<()> {
    use mary::embed::LocalEmbedder;
    let device = mary::embed::default_device();
    let mut out = serde_json::Map::new();
    if !texts.is_empty() {
        let (keymap, tokenizer, _) = load_parts(text_pile, text_quantization)?;
        let emb = mary::embed::nomic_text_from_parts(keymap, tokenizer, device.clone())?;
        let mut arr = Vec::new();
        for t in texts {
            let mut v = emb.embed_query(t)?;
            l2_normalize(&mut v);
            arr.push(serde_json::json!({ "text": t, "vector": v }));
        }
        out.insert("texts".into(), serde_json::Value::Array(arr));
    }
    if !images.is_empty() {
        let emb = mary::embed::load_nomic_vision_from_keymap(
            vision_keymap(vision_pile, vision_quantization)?,
            device,
        )?;
        let mut arr = Vec::new();
        for p in images {
            let bytes = fs::read(p).with_context(|| format!("read {}", p.display()))?;
            let mut v = emb.embed_image(&bytes)?;
            l2_normalize(&mut v);
            arr.push(serde_json::json!({ "image": p.display().to_string(), "vector": v }));
        }
        out.insert("images".into(), serde_json::Value::Array(arr));
    }
    println!("{}", serde_json::Value::Object(out));
    Ok(())
}

fn recall_at_k(baseline: &[usize], candidate: &[usize]) -> f32 {
    let hits = candidate.iter().filter(|c| baseline.contains(c)).count();
    hits as f32 / baseline.len().max(1) as f32
}

fn top_k(query: &[f32], docs: &[Vec<f32>], exclude: Option<usize>, k: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = docs
        .iter()
        .enumerate()
        .filter(|(i, _)| Some(*i) != exclude)
        .map(|(i, d)| (i, dot(query, d)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

struct ProbeOptions {
    queries: usize,
    cache: Option<PathBuf>,
    only: Vec<String>,
    list_tensors: bool,
    skip_output_variants: bool,
    /// Per-block E4M3 scale by least squared error instead of block-max-to-6.
    scale_search: bool,
    /// Leave every weight in f32, so the run measures only what the model's
    /// own activation quantisation (`NOMIC_ACT_QUANT`) costs.
    keep_weights: bool,
    /// Activation-aware rounding: capture every linear's inputs over this many
    /// calibration texts, then choose each tensor's channel scale exponent by
    /// least output error before rounding. Zero means plain rounding.
    calibrate: usize,
    /// With `calibrate`: error-feedback rounding against the captured rows
    /// after the scale search (GPTQ).
    gptq: bool,
    /// Which model root to read: the `quantization` label on the root.
    quantization: String,
    /// With `calibrate`: write the calibrated model as packed NVFP4 leaves
    /// into this NEW pile (`--key` signs it).
    pack: Option<PathBuf>,
    key: Option<PathBuf>,
    /// With `calibrate`: pack the embedding tables as well (nearest rounding).
    pack_embeddings: bool,
}

/// f32 document and query vectors, saved after the first run so a sweep over
/// weight groups does not re-embed the corpus with the unquantized model.
fn save_vectors(path: &Path, docs: &[Vec<f32>], queries: &[Vec<f32>]) -> Result<()> {
    let mut bytes = Vec::new();
    for group in [docs, queries] {
        bytes.extend_from_slice(&(group.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(group.first().map_or(0, Vec::len) as u64).to_le_bytes());
        for v in group {
            for x in v {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
    fs::write(path, bytes)?;
    Ok(())
}

fn load_vectors(path: &Path) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let bytes = fs::read(path)?;
    let mut at = 0usize;
    let mut read_group = |at: &mut usize| -> Result<Vec<Vec<f32>>> {
        let n = u64::from_le_bytes(bytes[*at..*at + 8].try_into()?) as usize;
        let d = u64::from_le_bytes(bytes[*at + 8..*at + 16].try_into()?) as usize;
        *at += 16;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let mut v = Vec::with_capacity(d);
            for _ in 0..d {
                v.push(f32::from_le_bytes(bytes[*at..*at + 4].try_into()?));
                *at += 4;
            }
            out.push(v);
        }
        Ok(out)
    };
    let docs = read_group(&mut at)?;
    let queries = read_group(&mut at)?;
    Ok((docs, queries))
}

fn probe(model_pile: &Path, corpus: &Path, options: ProbeOptions) -> Result<()> {
    let rows = read_corpus(corpus)?;
    if rows.len() < 20 {
        return Err(anyhow!("corpus has only {} texts", rows.len()));
    }
    eprintln!("corpus: {} texts", rows.len());

    let (keymap, tokenizer, parents) = load_parts(model_pile, &options.quantization)?;
    if options.list_tensors {
        let mut names: Vec<_> = keymap
            .iter()
            .map(|(n, (_, s))| (n.clone(), s.clone()))
            .collect();
        names.sort();
        for (name, shape) in names {
            println!("{name} {shape:?}");
        }
        return Ok(());
    }
    let device = mary::embed::default_device();
    let queries = options.queries.min(rows.len());
    let step = rows.len() / queries;
    let query_ids: Vec<usize> = (0..queries).map(|i| i * step).collect();

    let cached = options
        .cache
        .as_ref()
        .filter(|p| p.exists())
        .map(|p| load_vectors(p))
        .transpose()?;
    let (docs, qvecs): (Vec<Vec<f32>>, Vec<Vec<f32>>) = if let Some((docs, qvecs)) = cached {
        if docs.len() != rows.len() || qvecs.len() != queries {
            return Err(anyhow!(
                "cached vectors do not match the corpus and query count"
            ));
        }
        eprintln!("f32 vectors loaded from cache");
        (docs, qvecs)
    } else {
        let started = Instant::now();
        let f32_model =
            mary::embed::nomic_text_from_parts(keymap.clone(), tokenizer.clone(), device.clone())?;
        eprintln!(
            "f32 model built in {:.1} s",
            started.elapsed().as_secs_f64()
        );
        let started = Instant::now();
        let mut docs: Vec<Vec<f32>> = Vec::with_capacity(rows.len());
        for (_, _, text) in &rows {
            let mut v = f32_model.embed_document(text)?;
            l2_normalize(&mut v);
            docs.push(v);
        }
        eprintln!(
            "embedded {} documents in {:.1} s",
            docs.len(),
            started.elapsed().as_secs_f64()
        );
        let mut qvecs: Vec<Vec<f32>> = Vec::with_capacity(queries);
        for &i in &query_ids {
            let mut v = f32_model.embed_query(&rows[i].2)?;
            l2_normalize(&mut v);
            qvecs.push(v);
        }
        if let Some(path) = &options.cache {
            save_vectors(path, &docs, &qvecs)?;
            eprintln!("f32 vectors cached at {}", path.display());
        }
        (docs, qvecs)
    };
    let dim = docs[0].len();
    let baseline: Vec<Vec<usize>> = query_ids
        .iter()
        .zip(&qvecs)
        .map(|(&i, q)| top_k(q, &docs, Some(i), 10))
        .collect();

    println!(
        "nomic-embed-text-v1.5 over {} texts, {} queries, {dim}-d, recall@10 against f32 model and f32 vectors",
        rows.len(),
        queries
    );
    println!(
        "{:<34} {:>10} {:>12} {:>14}",
        "variant", "bits/dim", "recall@10", "mean |dcos|"
    );

    let report = |name: &str, bits: f32, stored: &[Vec<f32>], qs: &[Vec<f32>]| {
        let mut recall = 0f32;
        let mut cos_err = 0f64;
        let mut pairs = 0usize;
        for ((&i, q), base) in query_ids.iter().zip(qs).zip(&baseline) {
            let got = top_k(q, stored, Some(i), 10);
            recall += recall_at_k(base, &got);
            for &j in base {
                let exact = dot(
                    &qvecs[query_ids.iter().position(|&x| x == i).unwrap()],
                    &docs[j],
                );
                let approx = dot(q, &stored[j]);
                cos_err += (exact - approx).abs() as f64;
                pairs += 1;
            }
        }
        println!(
            "{:<34} {:>10} {:>11.1}% {:>14.5}",
            name,
            format!("{bits:.1}"),
            recall / queries as f32 * 100.0,
            cos_err / pairs.max(1) as f64
        );
    };

    report("f32 vectors", 32.0, &docs, &qvecs);
    if !options.skip_output_variants {
        let v: Vec<Vec<f32>> = docs
            .iter()
            .map(|d| {
                let mut q = nvfp4_quantize(d).0;
                l2_normalize(&mut q);
                q
            })
            .collect();
        report("NVFP4, one stage", 4.5, &v, &qvecs);
        let v: Vec<Vec<f32>> = docs
            .iter()
            .map(|d| {
                let mut q = nvfp4_two_stage(d);
                l2_normalize(&mut q);
                q
            })
            .collect();
        report("NVFP4, two-stage residual", 9.0, &v, &qvecs);
        let v: Vec<Vec<f32>> = docs
            .iter()
            .map(|d| {
                let mut q = fp4_lanes16(d);
                l2_normalize(&mut q);
                q
            })
            .collect();
        report("FP4, sixteen lanes (fractional)", 64.5, &v, &qvecs);
        let v: Vec<Vec<f32>> = docs
            .iter()
            .map(|d| {
                let mut q = int8(d);
                l2_normalize(&mut q);
                q
            })
            .collect();
        report("int8, per-vector scale", 8.0, &v, &qvecs);
        let v: Vec<Vec<f32>> = docs
            .iter()
            .map(|d| {
                let mut q = binary(d);
                l2_normalize(&mut q);
                q
            })
            .collect();
        let qb: Vec<Vec<f32>> = qvecs
            .iter()
            .map(|q| {
                let mut b = binary(q);
                l2_normalize(&mut b);
                b
            })
            .collect();
        report("binary, both sides", 1.0, &v, &qb);
        // Query side quantized too, for the symmetric NVFP4 case an index would use.
        let v: Vec<Vec<f32>> = docs
            .iter()
            .map(|d| {
                let mut q = nvfp4_two_stage(d);
                l2_normalize(&mut q);
                q
            })
            .collect();
        let q2: Vec<Vec<f32>> = qvecs
            .iter()
            .map(|q| {
                let mut b = nvfp4_two_stage(q);
                l2_normalize(&mut b);
                b
            })
            .collect();
        report("NVFP4 two-stage, both sides", 9.0, &v, &q2);
    }

    // The model itself with NVFP4 weights, all of them or the named group.
    let mut quantized = keymap;
    let mut calibrated: HashMap<String, mary::calibrate::Calibrated> = HashMap::new();
    let (tensors, elements) = if options.keep_weights {
        (0, 0)
    } else if options.calibrate > 0 {
        // Calibration texts spread over the corpus, embedded through the f32
        // model with capture on; the captured inputs drive the per-tensor
        // scale search below.
        let n = options.calibrate.min(rows.len());
        let stride = rows.len() / n;
        let started = Instant::now();
        let f32_model = mary::embed::nomic_text_from_parts(
            quantized.clone(),
            tokenizer.clone(),
            device.clone(),
        )?;
        mary::embed::nomic_activation_capture_start(8);
        for i in 0..n {
            let _ = f32_model.embed_document(&rows[i * stride].2)?;
        }
        let stats = mary::embed::nomic_activation_capture_take();
        drop(f32_model);
        let rows_captured: usize = stats.values().map(|s| s.rows.len()).sum();
        eprintln!(
            "captured inputs of {} linears over {n} texts, {rows_captured} reservoir rows, in {:.1} s",
            stats.len(),
            started.elapsed().as_secs_f64()
        );
        let started = Instant::now();
        let (t, e, packed) = fake_nvfp4_weights_awq(
            &mut quantized,
            &options.only,
            options.scale_search,
            options.gptq,
            options.pack_embeddings,
            &stats,
        );
        eprintln!(
            "activation-aware scale search took {:.1} s",
            started.elapsed().as_secs_f64()
        );
        calibrated = packed;
        (t, e)
    } else {
        fake_nvfp4_weights(&mut quantized, &options.only, options.scale_search)
    };
    let rounding = if options.keep_weights {
        "weights left in f32"
    } else if options.calibrate > 0 && options.gptq {
        "activation-aware scales + error-feedback rounding"
    } else if options.calibrate > 0 {
        "activation-aware scales"
    } else if options.scale_search {
        "block scale search"
    } else {
        "round to nearest"
    };
    eprintln!(
        "fake-quantized {tensors} weight tensors, {elements} elements, to NVFP4 (group: {}; {rounding})",
        if options.only.is_empty() {
            "all".to_string()
        } else {
            options.only.join(",")
        }
    );
    if let Some(out) = &options.pack {
        let key = options
            .key
            .as_ref()
            .ok_or_else(|| anyhow!("--pack needs --key <signing key>"))?;
        anyhow::ensure!(
            options.calibrate > 0 && !options.keep_weights,
            "--pack writes the calibrated model; give --calibrate N"
        );
        let started = Instant::now();
        let key = triblespace::core::signing_key_file::load_existing(key)
            .with_context(|| format!("load signing key {}", key.display()))?;
        let json = tokenizer
            .to_string(false)
            .map_err(|e| anyhow!("serialise tokenizer: {e}"))?;
        let root = mary::calibrate::write_packed_pile(
            out,
            &key,
            &quantized,
            &calibrated,
            NOMIC_TEXT_MODEL,
            "nvfp4-calibrated",
            &parents,
            Some(json.as_bytes()),
            false,
        )?;
        eprintln!(
            "packed model written to {} (root {root}, {} packed tensors) in {:.1} s",
            out.display(),
            calibrated.len(),
            started.elapsed().as_secs_f64()
        );
    }
    let started = Instant::now();
    let q_model = mary::embed::nomic_text_from_parts(quantized, tokenizer, device)?;
    let mut qdocs: Vec<Vec<f32>> = Vec::with_capacity(rows.len());
    for (_, _, text) in &rows {
        let mut v = q_model.embed_document(text)?;
        l2_normalize(&mut v);
        qdocs.push(v);
    }
    let mut qqueries: Vec<Vec<f32>> = Vec::with_capacity(queries);
    for &i in &query_ids {
        let mut v = q_model.embed_query(&rows[i].2)?;
        l2_normalize(&mut v);
        qqueries.push(v);
    }
    eprintln!(
        "NVFP4-weight model embedded everything in {:.1} s",
        started.elapsed().as_secs_f64()
    );
    let same_text_cos: f64 = docs
        .iter()
        .zip(&qdocs)
        .map(|(a, b)| dot(a, b) as f64)
        .sum::<f64>()
        / docs.len() as f64;
    println!();
    println!(
        "model with NVFP4 linear weights, group {} ({tensors} tensors, {elements} elements, {rounding}): mean cosine to the f32 model's vector of the same text {same_text_cos:.5}",
        if options.only.is_empty() {
            "all".to_string()
        } else {
            options.only.join(",")
        }
    );
    report("NVFP4 weights, f32 vectors", 32.0, &qdocs, &qqueries);
    let v: Vec<Vec<f32>> = qdocs
        .iter()
        .map(|d| {
            let mut q = nvfp4_two_stage(d);
            l2_normalize(&mut q);
            q
        })
        .collect();
    let q2: Vec<Vec<f32>> = qqueries
        .iter()
        .map(|q| {
            let mut b = nvfp4_two_stage(q);
            l2_normalize(&mut b);
            b
        })
        .collect();
    report("NVFP4 weights + two-stage vectors", 9.0, &v, &q2);
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    match args.first().map(String::as_str) {
        Some("extract") => {
            let pile = PathBuf::from(flag("--pile").ok_or_else(|| anyhow!("--pile"))?);
            let out = PathBuf::from(flag("--out").unwrap_or_else(|| "corpus.jsonl".into()));
            let max_chars: usize = flag("--max-chars")
                .map(|s| s.parse())
                .transpose()?
                .unwrap_or(4000);
            let mut sources = Vec::new();
            let mut attr = [0u8; 16];
            if let Some(h) = flag("--wiki") {
                hex_decode("6DBBE746B7DD7A4793CA098AB882F553", &mut attr)?;
                sources.push(("wiki".to_string(), attr, h));
            }
            if let Some(h) = flag("--journal") {
                hex_decode("3292CF0B3B6077991D8ECE6E2973D4B6", &mut attr)?;
                sources.push(("journal".to_string(), attr, h));
            }
            extract(&pile, &sources, &out, max_chars)
        }
        Some("dump") => {
            let text_pile =
                PathBuf::from(flag("--text-model").ok_or_else(|| anyhow!("--text-model"))?);
            let vision_pile =
                PathBuf::from(flag("--vision-model").ok_or_else(|| anyhow!("--vision-model"))?);
            let tq = flag("--text-quantization")
                .unwrap_or_else(|| mary::persist::QUANTIZATION_NATIVE.to_string());
            let vq = flag("--vision-quantization")
                .unwrap_or_else(|| mary::persist::QUANTIZATION_NATIVE.to_string());
            let texts: Vec<String> = flag("--texts")
                .map(|s| s.split('|').map(str::to_string).collect())
                .unwrap_or_default();
            let images: Vec<PathBuf> = flag("--images")
                .map(|s| s.split(',').map(PathBuf::from).collect())
                .unwrap_or_default();
            dump(&text_pile, &tq, &vision_pile, &vq, &texts, &images)
        }
        Some("vision") => {
            let model = PathBuf::from(flag("--model").ok_or_else(|| anyhow!("--model"))?);
            let packed = PathBuf::from(flag("--packed").ok_or_else(|| anyhow!("--packed"))?);
            let images = PathBuf::from(flag("--images").ok_or_else(|| anyhow!("--images"))?);
            let quantization = flag("--quantization")
                .unwrap_or_else(|| mary::persist::QUANTIZATION_NATIVE.to_string());
            let packed_quantization =
                flag("--packed-quantization").unwrap_or_else(|| "nvfp4-calibrated".to_string());
            vision(
                &model,
                &quantization,
                &packed,
                &packed_quantization,
                &images,
            )
        }
        Some("probe") => {
            let model = PathBuf::from(flag("--model").ok_or_else(|| anyhow!("--model"))?);
            let corpus = PathBuf::from(flag("--corpus").unwrap_or_else(|| "corpus.jsonl".into()));
            let queries: usize = flag("--queries")
                .map(|s| s.parse())
                .transpose()?
                .unwrap_or(200);
            let only: Vec<String> = flag("--only")
                .map(|s| {
                    s.split(',')
                        .map(|p| p.trim().to_string())
                        .filter(|p| !p.is_empty())
                        .collect()
                })
                .unwrap_or_default();
            probe(
                &model,
                &corpus,
                ProbeOptions {
                    queries,
                    cache: flag("--cache").map(PathBuf::from),
                    only,
                    list_tensors: args.iter().any(|a| a == "--list-tensors"),
                    skip_output_variants: args.iter().any(|a| a == "--weights-only"),
                    scale_search: args.iter().any(|a| a == "--scale-search"),
                    keep_weights: args.iter().any(|a| a == "--keep-weights"),
                    calibrate: flag("--calibrate")
                        .map(|s| s.parse())
                        .transpose()?
                        .unwrap_or(0),
                    gptq: args.iter().any(|a| a == "--gptq"),
                    quantization: flag("--quantization")
                        .unwrap_or_else(|| mary::persist::QUANTIZATION_NATIVE.to_string()),
                    pack: flag("--pack").map(PathBuf::from),
                    key: flag("--key").map(PathBuf::from),
                    pack_embeddings: args.iter().any(|a| a == "--pack-embeddings"),
                },
            )
        }
        _ => Err(anyhow!(
            "usage: nomic_fp4_probe extract --pile P --wiki H --journal H --out F | probe --model P --corpus F [--queries N] [--cache F] [--only a,b] [--weights-only] [--scale-search] [--keep-weights] [--calibrate N] [--gptq] [--list-tensors] [--quantization TAG] [--pack OUT.pile --key KEY] [--pack-embeddings] | vision --model P --packed Q --images DIR [--quantization TAG] [--packed-quantization TAG]"
        )),
    }
}
