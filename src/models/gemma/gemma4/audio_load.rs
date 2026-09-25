//! Load an audio file (wav/mp3/m4a/flac/ogg) into 16kHz mono f32 samples
//! suitable for `AudioFeatureExtractor::extract`. Uses symphonia for
//! decoding and rubato for sample-rate conversion.

use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::path::Path;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Load any common audio file, downmix to mono, resample to 16 kHz.
/// Returns the f32 waveform in [-1, 1] range at 16 kHz.
pub fn load_audio_16k_mono(path: &Path) -> Result<Vec<f32>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("open: {e}"))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    decode_audio_16k_mono(mss, path.extension().and_then(|s| s.to_str()))
}

/// Decode immutable resident audio with the same decoder/downmix/resampler as
/// the file entrypoint. The optional extension is only a format hint: no path,
/// device, network or temporary file is accessed by this function.
pub fn load_audio_16k_mono_bytes(
    bytes: Vec<u8>,
    extension: Option<&str>,
) -> Result<Vec<f32>, String> {
    let mss = MediaSourceStream::new(Box::new(std::io::Cursor::new(bytes)), Default::default());
    decode_audio_16k_mono(mss, extension)
}

fn decode_audio_16k_mono(
    mss: MediaSourceStream,
    extension: Option<&str>,
) -> Result<Vec<f32>, String> {
    let mut hint = Hint::new();
    if let Some(ext) = extension {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| format!("probe: {e}"))?;
    let mut reader = probed.format;

    let track = reader.default_track().ok_or("no default track")?;
    let codec_params = track.codec_params.clone();
    let track_id = track.id;
    let src_rate = codec_params.sample_rate.ok_or("unknown sample rate")? as usize;

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| format!("decoder: {e}"))?;

    // Container-level channel count may be unknown (e.g. AAC in MP4). Pull
    // it from the first successfully-decoded packet instead.
    let mut channels: Option<usize> = codec_params.channels.map(|c| c.count());
    let mut per_ch: Vec<Vec<f32>> = Vec::new();
    loop {
        let packet = match reader.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(format!("next_packet: {e}")),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(SymError::IoError(_)) | Err(SymError::DecodeError(_)) => continue,
            Err(e) => return Err(format!("decode: {e}")),
        };
        // Lazily determine channel count from the first decoded frame.
        if channels.is_none() {
            channels = Some(decoded.spec().channels.count());
        }
        let ch = channels.unwrap();
        if per_ch.is_empty() {
            per_ch = (0..ch).map(|_| Vec::new()).collect();
        }

        match decoded {
            AudioBufferRef::F32(buf) => {
                for c in 0..ch {
                    per_ch[c].extend_from_slice(buf.chan(c));
                }
            }
            AudioBufferRef::S16(buf) => {
                for c in 0..ch {
                    per_ch[c].extend(buf.chan(c).iter().map(|&s| s as f32 / 32768.0));
                }
            }
            AudioBufferRef::S32(buf) => {
                for c in 0..ch {
                    per_ch[c].extend(buf.chan(c).iter().map(|&s| s as f32 / 2_147_483_648.0));
                }
            }
            other => {
                let spec = *other.spec();
                let duration = other.capacity() as u64;
                let mut fbuf = symphonia::core::audio::AudioBuffer::<f32>::new(duration, spec);
                other.convert(&mut fbuf);
                for c in 0..ch {
                    per_ch[c].extend_from_slice(fbuf.chan(c));
                }
            }
        }
    }
    let channels = channels.ok_or("no audio frames decoded")?;
    // A valid container can declare channels yet contain no audio packets.
    // Resident uploads must fail normally instead of indexing an empty vector.
    if channels == 0 || per_ch.is_empty() {
        return Err("no audio frames decoded".to_owned());
    }

    // Downmix to mono by averaging.
    let n = per_ch[0].len();
    let mut mono = vec![0.0f32; n];
    for c in 0..channels {
        let src = &per_ch[c];
        // Guard against channels of different lengths.
        let m = src.len().min(n);
        for i in 0..m {
            mono[i] += src[i];
        }
    }
    let inv = 1.0 / channels as f32;
    for v in &mut mono {
        *v *= inv;
    }

    resample_to_16k(mono, src_rate)
}

/// Resample a mono f32 buffer from `src_rate` to 16 kHz (rubato sinc).
/// Pass-through when already 16 kHz. Shared by the file loader and the live
/// capture path (`gemma_listen` resamples each finished utterance segment).
pub fn resample_to_16k(mono: Vec<f32>, src_rate: usize) -> Result<Vec<f32>, String> {
    let dst_rate = 16_000usize;
    if src_rate == 0 {
        return Err("sample rate must be positive".to_owned());
    }
    if mono.is_empty() {
        return Ok(mono);
    }
    if src_rate == dst_rate {
        return Ok(mono);
    }
    let ratio = dst_rate as f64 / src_rate as f64;
    // rubato SincFixedIn wants fixed input chunks; use a reasonable size.
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let chunk = 4096usize;
    let mut resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, chunk, 1)
        .map_err(|e| format!("rubato init: {e}"))?;
    let mut out = Vec::with_capacity((mono.len() as f64 * ratio) as usize + 1024);
    let mut i = 0;
    while i + chunk <= mono.len() {
        let waves_in = vec![mono[i..i + chunk].to_vec()];
        let waves_out = resampler
            .process(&waves_in, None)
            .map_err(|e| format!("rubato process: {e}"))?;
        out.extend_from_slice(&waves_out[0]);
        i += chunk;
    }
    // Final partial chunk — zero-pad to size `chunk` so SincFixedIn accepts it.
    if i < mono.len() {
        let mut tail = vec![0.0f32; chunk];
        tail[..mono.len() - i].copy_from_slice(&mono[i..]);
        let waves_in = vec![tail];
        let waves_out = resampler
            .process(&waves_in, None)
            .map_err(|e| format!("rubato process tail: {e}"))?;
        // This also contains delayed real samples. Trim only after draining,
        // not at the input chunk boundary.
        out.extend_from_slice(&waves_out[0]);
    }

    let expected = (mono.len() as f64 * ratio).round() as usize;
    // A clip ending on a full input chunk still has audio in the filter.
    // Zero input drains that state; no output beyond the source duration is
    // returned. This matters for Complete and forced VAD boundaries in speech.
    while out.len() < expected {
        let waves_out = resampler
            .process(&[vec![0.0f32; chunk]], None)
            .map_err(|e| format!("rubato drain: {e}"))?;
        if waves_out[0].is_empty() {
            return Err("resampler made no progress while draining".to_owned());
        }
        out.extend_from_slice(&waves_out[0]);
    }
    // In the pinned SincFixedIn 0.16, the initial negative half-kernel index
    // already aligns the sinc center with the source. Skipping output_delay()
    // here shifts real signal earlier (85 samples at 24 kHz -> 16 kHz).
    // The impulse-position and trailing-signal controls below pin this, rather
    // than assuming buffering latency means leading silent output samples.
    Ok(out.into_iter().take(expected).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    #[test]
    fn resampling_preserves_impulse_positions_in_the_source_clock() {
        for rate in [24_000, 48_000] {
            for position in [rate / 10, rate / 2, rate * 9 / 10] {
                let mut wave = vec![0.0; rate];
                wave[position] = 1.0;
                let out = resample_to_16k(wave, rate).unwrap();
                let peak = out
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                    .unwrap()
                    .0;
                let expected = position * 16_000 / rate;
                assert!(
                    peak.abs_diff(expected) <= 1,
                    "rate={rate}, peak={peak}, expected={expected}"
                );
            }
        }
    }

    #[test]
    fn resampling_drains_real_tail_at_partial_and_exact_chunk_boundaries() {
        for rate in [24_000, 48_000] {
            for len in [1, 4096, 4097, 24_000, 48_000] {
                let mut wave = vec![0.0; len];
                // Signal confined to the tail catches zero-padding in lieu
                // of draining; the one-sample case checks bounded length only.
                let tail = len.min(32);
                wave[len - tail..].fill(0.5);
                let out = resample_to_16k(wave, rate).unwrap();
                let expected = (len as f64 * 16_000.0 / rate as f64).round() as usize;
                assert_eq!(out.len(), expected, "rate={rate}, len={len}");
                assert!(out.iter().all(|sample| sample.is_finite()));
                if len > 1 {
                    let peak = out
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                        .unwrap();
                    assert!(
                        out.last().unwrap().abs() > 0.1,
                        "tail lost: rate={rate}, len={len}, peak={peak:?}, last={:?}, expected={expected}", out.last()
                    );
                }
            }
        }
        assert_eq!(
            resample_to_16k(vec![0.25, -0.5], 16_000).unwrap(),
            [0.25, -0.5]
        );
        assert!(resample_to_16k(vec![0.0], 0).is_err());
        assert!(resample_to_16k(Vec::new(), 48_000).unwrap().is_empty());
    }

    fn stereo_wav() -> Vec<u8> {
        let samples = [0_i16, 0, 8192, 24576, -8192, -24576, 16384, -16384];
        let data_bytes = (samples.len() * 2) as u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&16000_u32.to_le_bytes());
        bytes.extend_from_slice(&64000_u32.to_le_bytes());
        bytes.extend_from_slice(&4_u16.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_bytes.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn resident_and_file_audio_are_identical_after_downmix() {
        let bytes = stereo_wav();
        let resident = load_audio_16k_mono_bytes(bytes.clone(), Some("wav")).unwrap();
        assert_eq!(resident, [0.0, 0.5, -0.5, 0.0]);
        assert_eq!(
            resident,
            load_audio_16k_mono_bytes(bytes.clone(), None).unwrap()
        );
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "mary-resident-audio-{}-{nonce}.wav",
            std::process::id()
        ));
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.write_all(&bytes).unwrap();
        drop(file);
        let decoded = load_audio_16k_mono(&path);
        std::fs::remove_file(path).unwrap();
        assert_eq!(decoded.unwrap(), resident);
    }

    #[test]
    fn malformed_resident_audio_is_a_decode_error() {
        assert!(
            load_audio_16k_mono_bytes(b"not an audio container".to_vec(), Some("wav")).is_err()
        );
    }

    #[test]
    fn empty_wav_returns_an_error_instead_of_panicking() {
        let mut bytes = stereo_wav();
        bytes.truncate(44);
        bytes[4..8].copy_from_slice(&36_u32.to_le_bytes());
        bytes[40..44].copy_from_slice(&0_u32.to_le_bytes());
        assert!(load_audio_16k_mono_bytes(bytes, Some("wav")).is_err());
    }
}
