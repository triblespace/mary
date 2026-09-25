# Changelog

## Unreleased

- Hearing can decode existing audio embeddings without a second encoder pass.
  Add a Gemma hearing-specific CUDA backend on Linux, leaving other backend
  choices unchanged; move the streaming weight loader's progress to stderr.
  The transcript token limit now includes the first generated token.
  The whole-clip `gemma_hear` control uses that same backend and accepts pinned
  config/tokenizer paths for offline comparisons with segmented hearing.
  Resampling drains delayed trailing samples before trimming to source duration;
  embedding-based transcription reports exhaustion of its token budget.

- Pin the build root to AnyBytes `066c32a7`, matching TribleSpace, Faculties,
  and Drive. Temporary section freezes no longer perform durability flushes;
  explicit `ByteArea::persist` owns that barrier. Model identities, tensor
  encodings, numerical kernels, and feature selections are unchanged.
