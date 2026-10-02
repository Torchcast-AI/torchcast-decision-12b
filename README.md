# Torchcast Decision 12B

A text-based decision model developed by **Torchcast AI**, based on Gemma-4-12B-it. It supports yes/no probabilities (`noul`), choice distributions (`choice`), and ordinal score distributions (`score`) through a TypeSafe-compatible API.

[Model card and weights](https://huggingface.co/torchcast-ai/torchcast-decision-12b)

## Model and readout

A LoRA fine-tune of Gemma-4-12B-it, trained on a 50/50 mix of gold labels and a larger instruction model's option distributions (not TypeSafe/Jev outputs), over procedurally generated decisions, the Open-Jev train split and public dataset train splits; sources and their terms are listed in the [model repository's LICENSE](https://huggingface.co/torchcast-ai/torchcast-decision-12b/blob/main/LICENSE). Each decision is one forward pass that reads the option letters at the answer position. Per-type temperatures in `serving/readout_config.json` were fitted on validation or test splits of public non-JevBench datasets: Choice and Score use the negative-log-likelihood optimum (1.0); the yes/no (`noul`) temperature (0.2) is the value on a fixed grid (0.2–3.0) with the best chance-corrected fit-set accuracy among those whose fit-set calibration error stays within the unmodified base model's, and is the lowest value on that grid.

## Run for evaluation

Use Linux with an NVIDIA GPU. The release check used one H100 80 GB, BF16 weights and vLLM 0.30.0; the same public-item result was recorded on an L40S 48 GB during development. From a checkout of tag `v1.0.0` (or the commit named in the evaluation request), start the model and then the supplied server:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install vllm==0.30.0
vllm serve torchcast-ai/torchcast-decision-12b \
  --revision 49107b5589bf2dc4ec3f5acd54316971dd5978b1 \
  --tokenizer-revision 49107b5589bf2dc4ec3f5acd54316971dd5978b1 \
  --served-model-name torchcast-decision-12b \
  --host 127.0.0.1 --port 8890 \
  --max-model-len 16384 --gpu-memory-utilization 0.90
```

Alternatively, with the pinned vLLM container (not used for the recorded runs):

```bash
docker run --gpus all -p 127.0.0.1:8890:8000 \
  vllm/vllm-openai:v0.30.0@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90 \
  --model torchcast-ai/torchcast-decision-12b \
  --revision 49107b5589bf2dc4ec3f5acd54316971dd5978b1 \
  --tokenizer-revision 49107b5589bf2dc4ec3f5acd54316971dd5978b1 \
  --served-model-name torchcast-decision-12b \
  --max-model-len 16384 --gpu-memory-utilization 0.90
```

In a second terminal, from the same checkout, activate the same environment:

```bash
source .venv/bin/activate
SHIM_VLLM=http://127.0.0.1:8890/v1/chat/completions \
SHIM_MODEL=torchcast-decision-12b SHIM_PORT=8011 \
python serving/torchcast_shim.py
```

Keep the virtual environment activated when starting vLLM and the supplied server. This places installed compilation tools such as `ninja` on PATH; invoking the vLLM binary by its full path alone does not activate the environment.

Use the bundled configuration without `SHIM_*` overrides other than those shown. The pinned artifact is `torchcast-ai/torchcast-decision-12b@49107b5589bf2dc4ec3f5acd54316971dd5978b1`; weight SHA-256: `f8553c9e625fa24853d57e938a1b1475975d9bce4bb79fccff2e8bfaaf6caa4e`.

Check the served model name, then send a warm-up request before timing:

```bash
curl -s http://127.0.0.1:8890/v1/models   # vLLM: "id": "torchcast-decision-12b"
curl --fail http://127.0.0.1:8011/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"A customer reports two charges for one order.","questions":{"decision":{"type":"choice","instructions":"Choose the support team.","criteria":{"billing":"Payments and refunds","technical":"Product malfunctions","other":"Other requests"}}}}'
```

For JevBench, use the `typesafe` adapter against `http://127.0.0.1:8011` with model name `torchcast-decision-12b`. Results should be produced under the benchmark's current evaluation protocol. This checkpoint has no published provider tariff; apply the benchmark's documented estimated-cost method.

The public-231 result below was produced with this command, from a JevBench checkout at commit `bb05a335bc809e61b20c0f745d25499a82b326fc`, with the server above running:

```bash
git clone https://github.com/fstandhartinger/jevbench && cd jevbench
git checkout bb05a335bc809e61b20c0f745d25499a82b326fc && pip install -e .
J=datasets/public
python -m jevbench.cli run \
  --tasks $J/easy.jsonl,$J/original.jsonl,$J/hard.jsonl \
  --adapter typesafe --endpoint http://127.0.0.1:8011 --key-env "" \
  --model torchcast-decision-12b --cost-basis self_hosted_gpu --reserve-usd 0 \
  --results runs/results.jsonl --raw-dir runs/raw --ledger runs/ledger.jsonl \
  --manifest runs/manifest.json --delay-s 0
```

Regression checks for the serving code (standard library only, no GPU):

```bash
python3 serving/test_shim.py
python3 serving/test_decision_server.py
python3 serving/test_readout_config.py
```

## Supported interface and limits

The evaluation server accepts one question named `decision`. It listens on 127.0.0.1 by default; if the benchmark client runs on another machine, add `SHIM_HOST=0.0.0.0` (and restrict access at the network level). Its context limit is 16,384 tokens including the rendered request; overlength inputs receive HTTP 422. For multiple questions or more than 20 options, use `serving/decision_server.py` (settings in `serving/NOTE.md`); the measurements below do not establish that path's performance.

The evaluated use is English text and structured-state decision research. Probabilities can be unreliable on new domains. Multilingual, image/audio and autonomous-action quality are not established.

## Public-231 result (JevBench v1.4 public set)

**Author-run, not an official JevBench score or rank.** On the 231 public items: **203/231 correct (87.88%)**, split easy 48/48, original 70/72, hard 85/111; all answers valid, zero failed requests. On two H100 release-check runs, serial client latency for easy+original items was 30–35 ms p50 and 43 ms p95, with a mean of 704 input tokens and 1 output token per decision. This is not the benchmark's official Speed measurement.

The run used JevBench CLI `bb05a335bc809e61b20c0f745d25499a82b326fc`; its manifest and per-item results are in [`runs/h100-release/`](runs/h100-release/). It does not cover the current full suite or sealed set.

## Benchmark exposure

Public JevBench results influenced checkpoint and serving-configuration selection. An 8-gram screen of an earlier training set found procedurally generated rows whose template wording overlapped 14 public items. Those rows were removed, and this checkpoint's training data shares no 8-gram with the 231 public items. No item fact patterns or answers were copied, and no sealed or unpublished items were accessed. The public results are development measurements, not an untouched holdout or a zero-contamination claim.

## Licence and attribution

Model weights are labelled **CC BY-NC 4.0**, for non-commercial use. This label does not establish clearance of every training-source right. Applicable upstream model conditions and source-specific terms remain relevant; see the model repository's licence files. Use of the model is also subject to Google's Gemma Prohibited Use Policy (https://ai.google.dev/gemma/prohibited_use_policy).

Serving code is MIT-licensed. Preserve `LICENSE`, `NOTICE.md` and `serving/LICENSE-serving-MIT`. Credit Torchcast AI for this checkpoint, Google DeepMind for Gemma, and Cygnet/blockbrain-ai and NInfer for the upstream serving/readout foundation. This project is independent of TypeSafe AI and the JevBench maintainers.
