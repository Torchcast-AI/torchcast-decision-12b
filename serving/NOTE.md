# Serving checks

`torchcast_shim.py` serves the evaluation interface. `decision_server.py` serves applications.
`reference_shim.py` is a frozen upstream reference with renamed module and default model identifiers.
See `../NOTICE.md` and `LICENSE-serving-MIT` for attribution.

```bash
python3 serving/test_shim.py
python3 serving/test_decision_server.py
python3 serving/test_readout_config.py
```

## Evaluation server settings (`torchcast_shim.py`)

| Variable | Default | Meaning |
|---|---|---|
| `SHIM_VLLM` | `http://127.0.0.1:8890/v1/chat/completions` | vLLM chat-completions URL |
| `SHIM_MODEL` | `torchcast-decision-12b` | must equal vLLM `--served-model-name` |
| `SHIM_HOST` | `127.0.0.1` | listen address; set `0.0.0.0` when the benchmark client runs on another machine |
| `SHIM_PORT` | `8011` | listen port |
| `SHIM_TOP_LOGPROBS` | `20` | top logprobs requested per pass |
| `SHIM_TIMEOUT` | `180` | upstream timeout, seconds |
| `SHIM_MIN_CONTEXT` | `4096` | refuse to serve (503) if vLLM's `max_model_len` is lower |
| `SHIM_READOUT_CONFIG` | bundled `readout_config.json` | readout configuration file |
| `SHIM_T_CHOICE`, `SHIM_T_NOUL`, `SHIM_T_SCORE`, `SHIM_TEMPERATURE` | unset | overrides; leave unset for evaluation |

## Application server settings (`decision_server.py`)

`decision_server.py` accepts several questions per request and more than 20 options (grouped passes). It reads the
`SHIM_*` settings above for the upstream connection and readout, plus:

| Variable | Default | Meaning |
|---|---|---|
| `TORCHCAST_HOST` | `127.0.0.1` | listen address; a non-loopback address requires `TORCHCAST_API_KEY` |
| `TORCHCAST_PORT` | `8010` | listen port |
| `TORCHCAST_API_KEY` | empty | if set, requests need `Authorization: Bearer <key>` (401 otherwise) |
| `TORCHCAST_ALLOW_NO_KEY` | empty | `1` allows a non-loopback address without a key, when a proxy in front checks access |
| `TORCHCAST_MAX_PARALLEL` | `8` | cap on vLLM requests in flight |
| `TORCHCAST_GROUP_SIZE` | `min(20, SHIM_TOP_LOGPROBS)` | options per grouped pass |
| `TORCHCAST_MAX_BODY` | 16 MiB | request body limit (413 above it) |
| `TORCHCAST_MODEL_NAME`, `TORCHCAST_MODEL_DESCRIPTION`, `TORCHCAST_MODEL_RELEASE_DATE` | model metadata | returned by `GET /v1/models` |

Status codes: 400 malformed body, 401 missing or invalid key, 404 unknown path, 413 body too large, 422 input this
system cannot answer (over context, too many options, unknown type), 502 upstream failure (vLLM 401/403/429 pass
through), 503 vLLM model id missing or context below `SHIM_MIN_CONTEXT`.

Noul criteria: `{"true": ..., "false": ...}` in either order. Missing criteria or a missing side defaults to "No"/"Yes"; `yes`/`no` keys are read as `true`/`false`. Any other key is a 422.
