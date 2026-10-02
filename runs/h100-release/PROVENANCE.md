# Release check: public 231 on one H100

Author-run, not an official JevBench score or rank.

- **What ran:** the README commands verbatim from this repository's `serving/`, with the model pulled by name from
  Hugging Face at the published revision `49107b5589bf2dc4ec3f5acd54316971dd5978b1`, then the JevBench CLI (`typesafe` adapter, serial, one request
  at a time) against `torchcast_shim.py` on port 8011. Started 2026-10-02T18:15:00Z, finished 2026-10-02T18:15:16Z.
- **Result:** 203/231 correct (easy 48/48, original 70/72, hard 85/111); 231/231 valid; 0 failed requests;
  served model id `torchcast-decision-12b`; readout temperatures choice 1.0, score 1.0, noul 0.2 loaded from
  `readout_config.json`. Easy+original serial client latency 35 ms p50, 43 ms p95. Mean 704 input tokens and 1
  output token per decision.
- **Earlier run:** an earlier release check on a pre-release revision with the same weights (SHA-256
  `f8553c9e625fa24853d57e938a1b1475975d9bce4bb79fccff2e8bfaaf6caa4e`) and temperatures gave the same 203/231 and
  the same per-item correctness split, with 30 ms p50 and 43 ms p95.
- **Files:** `manifest.json` (SHA-256 `3b17a8e2a4073d937764e9701a996fb0d8d8e9f268b12c5b7e50468df596530c`),
  `results.jsonl` (SHA-256 `fa2e0ca8e745c02c9add971f8f22cf90881829a88b578985db98161e9a78b0f9`), `env.txt`.
