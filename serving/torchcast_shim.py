#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Nood Co and contributors; modifications by Torchcast AI.
"""Torchcast typed-decision HTTP shim. Uses the bundled readout_config.json; serves POST /v1/systemone. Derived from Cygnet (MIT); see LICENSE-serving-MIT."""
from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VLLM = os.environ.get("SHIM_VLLM", "http://127.0.0.1:8890/v1/chat/completions")
MODEL = os.environ.get("SHIM_MODEL", "torchcast-decision-12b")   # must match vLLM --served-model-name
HOST = os.environ.get("SHIM_HOST", "127.0.0.1")   # set 0.0.0.0 to accept a client on another machine
PORT = int(os.environ.get("SHIM_PORT", "8011"))
TOP_LOGPROBS = int(os.environ.get("SHIM_TOP_LOGPROBS", "20"))
TIMEOUT = float(os.environ.get("SHIM_TIMEOUT", "180"))
MIN_CONTEXT = int(os.environ.get("SHIM_MIN_CONTEXT", "4096"))
# Readout temperatures, one per question type: p_i^(1/T) renormalised; 1.0 leaves a type unchanged.
QTYPES = ("choice", "noul", "score")
TYPE_ENV = {"choice": "SHIM_T_CHOICE", "noul": "SHIM_T_NOUL", "score": "SHIM_T_SCORE"}
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "readout_config.json")


class ConfigError(ValueError):
    """The readout configuration is missing or malformed. The shim does not start without one."""


def _positive(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} must be a number, not {value!r}")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ConfigError(f"{where} must be a positive finite number, not {value!r}")
    return value


def load_temperatures(path=None, environ=None):
    """({qtype: T}, {qtype: source}) from the readout config and the environment.

    The config file is always read and checked, even when the environment sets every type, so a broken or missing
    file never goes unnoticed. Precedence per type: SHIM_T_<TYPE>, then SHIM_TEMPERATURE, then the file.
    Raises ConfigError on a missing or unreadable file, invalid JSON, a missing "temperature" object or type, or a
    value that is not a positive finite number.
    """
    environ = os.environ if environ is None else environ
    path = path or environ.get("SHIM_READOUT_CONFIG") or DEFAULT_CONFIG
    try:
        with open(path, encoding="utf-8") as f:
            config = json.load(f)
    except OSError as e:
        raise ConfigError(f"cannot read the readout config {path}: {e}") from e
    except ValueError as e:
        raise ConfigError(f"the readout config {path} is not valid JSON: {e}") from e
    table = config.get("temperature") if isinstance(config, dict) else None
    if not isinstance(table, dict):
        raise ConfigError(f"the readout config {path} has no \"temperature\" object")
    temps, sources = {}, {}
    for qtype in QTYPES:
        if qtype not in table:
            raise ConfigError(f"the readout config {path} has no temperature for {qtype!r}")
        temps[qtype] = _positive(table[qtype], f"{path}: temperature.{qtype}")
        sources[qtype] = os.path.basename(path)
    for qtype in QTYPES:
        for key in (TYPE_ENV[qtype], "SHIM_TEMPERATURE"):
            raw = environ.get(key)
            if raw:
                try:
                    value = float(raw)
                except ValueError:
                    raise ConfigError(f"{key}={raw!r} is not a number") from None
                temps[qtype], sources[qtype] = _positive(value, key), key
                break
    return temps, sources


def describe_temperatures():
    return ", ".join(f"{t} T={TEMPS[t]:g} ({TEMP_SOURCES[t]})" for t in QTYPES)


try:
    TEMPS, TEMP_SOURCES = load_temperatures()
except ConfigError as _e:
    raise SystemExit(f"shim: {_e}") from None
# Thinking stays off: this is a readout, and a reasoning trace before an answer slot is not one.
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}

# One letter per option, so up to 26 options. vLLM returns at most TOP_LOGPROBS (20) candidates, so on a
# decision with 21-26 options the 20 most probable letters carry the distribution and the rest get 0.
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# vLLM's wording when a prompt does not fit --max-model-len (it has changed between releases).
_CONTEXT_ERROR = re.compile(r"maximum context length|maximum model length|max_model_len|too long and exceeds", re.I)


class Unprocessable(ValueError):
    """An input this system cannot answer: over the server's context or size limit, more than 26 options, a
    question type it does not know, or options in the wrong form for their type.

    THE STATUS CODE IS THE CONTRACT. The benchmark's runner stops a run after three consecutive failed items
    unless the failure is an HTTP 422, which it scores as one wrong answer and moves on
    (`jevbench/runner.py`, "a 422 is the system refusing this input (e.g. over its context limit), not an
    outage"). Answering these with a 500 instead would let three long items in a row end the run.
    """


class UpstreamError(RuntimeError):
    """vLLM failed or could not be reached: a 502, so a dead server still stops the run. vLLM's own 401, 403
    and 429 pass through unchanged, because the runner stops at once on those."""

    def __init__(self, message, status=502):
        super().__init__(message)
        self.status = status


# A SERVER STARTED WITH A SMALL --max-model-len WOULD MAKE EVERY LONGER ITEM A 422, and a run of 422s
# completes quietly with all of them wrong. So before answering, the shim reads the served model's limit, and
# answers 503 if it is below SHIM_MIN_CONTEXT, or if the server does not list SHIM_MODEL or report a limit.
# A 503 counts toward the runner's stop rule. 4096 is the smallest cap this package was validated at, and the
# longest public prompt is 3,946 tokens.
_server_context = None
_context_lock = threading.Lock()


def server_context():
    """The served model's max_model_len, read once from vLLM's /v1/models.

    Raises UpstreamError: 502 if the server cannot be read (vLLM's own 401, 403 and 429 pass through), 503 if it
    does not list SHIM_MODEL or reports no integer limit. A failure is not cached, so a server that comes up
    later is read on the next request.
    """
    global _server_context
    with _context_lock:
        if _server_context is None:
            url = VLLM.split("/v1/", 1)[0] + "/v1/models"
            try:
                with urllib.request.urlopen(url, timeout=10) as r:
                    models = json.loads(r.read()).get("data") or []
                entry = next((m for m in models if isinstance(m, dict) and m.get("id") == MODEL), None)
            except urllib.error.HTTPError as e:
                raise UpstreamError(f"{url}: HTTP {e.code}", e.code if e.code in (401, 403, 429) else 502) from e
            except (OSError, ValueError, AttributeError, TypeError) as e:
                raise UpstreamError(f"cannot read {url}: {type(e).__name__}: {e}") from e
            if entry is None:
                raise UpstreamError(f"{url} does not list {MODEL!r}; set SHIM_MODEL to the served model name", 503)
            if not isinstance(entry.get("max_model_len"), int) or isinstance(entry.get("max_model_len"), bool):
                raise UpstreamError(f"{url} reports no max_model_len for {MODEL!r}", 503)
            _server_context = entry["max_model_len"]
            sys.stderr.write(f"shim: the server's max_model_len is {_server_context}\n")
    return _server_context


# The system prompt mirrors the scaffolding of JevBench's openai_compat adapter
# (jevbench/adapters/openai_compat.py:20-24); without it the model tends to answer the "Answer:" slot in
# prose (upstream Cygnet finding).
SYSTEM = (
    "You are a calibration engine. You never answer in prose. You are given a state, a question and "
    "a numbered set of options, and you choose exactly one option. You reply with that option's "
    "LETTER and nothing else — a single character, no words, no punctuation, no explanation."
)


def options_from(criteria, qtype):
    """Ordered [(letter, response_label, description)].

    choice: criteria is {label: description}; the LABEL NAME is what the response must be keyed by.
    score : criteria is a list of level descriptions; label names are "0".."n-1" per the task's
            `labels` field, which is always the stringified index.
    noul  : criteria is {"true": ..., "false": ...} and the adapter wants P(yes). We read the
            true/false letters and hand back P(true), which the adapter maps onto {"yes","no"}.
    """
    if qtype not in ("choice", "score", "noul"):
        raise Unprocessable(f"unsupported question type {qtype!r}")
    if qtype == "score":
        if not isinstance(criteria, (list, tuple)):
            raise Unprocessable("score without a criteria list")
        items = [(str(i), v) for i, v in enumerate(criteria)]
    elif qtype == "noul":
        items = noul_items(criteria)
    else:
        if not isinstance(criteria, dict):
            raise Unprocessable(f"{qtype} without a criteria mapping")
        items = list(criteria.items())
    if not items:
        raise Unprocessable("no options")
    if len(items) > len(LETTERS):
        raise Unprocessable(f"{len(items)} options exceeds the {len(LETTERS)}-letter alphabet")
    return [(LETTERS[i], k, v) for i, (k, v) in enumerate(items)]


def noul_items(criteria):
    """[(label, description)] for a noul question, labels "true"/"false".

    A record's {"true": ..., "false": ...} keeps its own order and descriptions; answer_for finds
    which letter means "true". Missing criteria, or a missing side, falls back to "No"/"Yes"
    (false first), as decision_server.py does. "yes"/"no" keys are read as "true"/"false".
    """
    if criteria is None:
        criteria = {}
    if not isinstance(criteria, dict):
        raise Unprocessable("noul criteria must be an object with 'true' and 'false' descriptions")
    alias = {"true": "true", "yes": "true", "false": "false", "no": "false"}
    sides = {}
    for key, desc in criteria.items():
        side = alias.get(str(key).strip().lower())
        if side is None or side in sides:
            raise Unprocessable("noul criteria takes only 'true' and 'false', once each")
        sides[side] = desc
    items = [(side, desc) for side, desc in sides.items()]
    for side, default in (("false", "No"), ("true", "Yes")):
        if side not in sides:
            items.insert(0 if side == "false" else len(items), (side, default))
    return [(side, default_text(desc, side)) for side, desc in items]


def default_text(desc, side):
    if desc is None or (isinstance(desc, str) and not desc.strip()):
        return "Yes" if side == "true" else "No"
    return desc


def build_prompt(state, instructions, opts):
    # STATE CAN BE A DICT. Some hard tasks ship structured state (alias_directory, amendment,
    # archive, ...). The board's own `openai_compat` adapter serialises it --
    # `state_text = task.state if isinstance(task.state, str) else json.dumps(task.state,
    # ensure_ascii=False)` (jevbench/adapters/openai_compat.py:64-67), compactly. THIS SHIM DOES NOT: it
    # renders structured state with indent=1, which is the rendering every published upstream reference figure was
    # measured with, so it is kept. Calling .rstrip() on a dict raised an error, hence the explicit dump.
    state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions, ensure_ascii=False)
    lines = [state_text.rstrip(), "", instructions.rstrip(), "", "Options:"]
    for letter, _lab, desc in opts:
        lines.append(f"{letter}. {desc}")
    lines += ["", "Answer with the letter of exactly one option, and nothing else:"]
    return "\n".join(lines)


_TOK_LETTER = re.compile(r"^[\s(\[{'\"]*([A-Za-z])[\s.,:)\]}'\"]*$")


def letter_probs(top_logprobs, n_letters):
    """Sum probability mass per option letter from vLLM's top_logprobs for the first token.

    A letter can appear as several surface tokens ("A", " A", "A."), so mass is SUMMED per letter
    rather than taking only the exact match.

    Returns None only if NO letter appeared at all. Under the guided-choice constraint a missing
    letter means the model gave it ~0 mass, which is a legitimate value and not a failure — the
    earlier `len(acc) < 2` guard turned ordinary low-probability options into a run-killing 502.
    """
    acc = {}
    for tok, lp in (top_logprobs.items() if isinstance(top_logprobs, dict) else top_logprobs):
        m = _TOK_LETTER.match(tok)
        if not m:
            continue
        idx = LETTERS.index(m.group(1).upper())
        if idx >= n_letters:
            continue
        acc[idx] = acc.get(idx, 0.0) + math.exp(lp)
    if not acc:
        return None
    total = sum(acc.values())
    if total <= 0:
        return None
    return {i: acc.get(i, 0.0) / total for i in range(n_letters)}


def call_vllm(prompt_text, allowed_letters):
    """One forward pass, one token, constrained to the option letters.

    THE CONSTRAINT IS THE POINT. Without it the model can answer the "Answer:" slot in prose, so the
    top logprobs may contain few or none of the option letters and no distribution over the options can
    be recovered (upstream Cygnet finding). `structured_outputs: {"choice": [...]}` masks the logits to
    the allowed letters, so every option receives a real logprob and disallowed tokens are excluded.

    This is a readout of P(option | state) restricted to the options — which is exactly what a
    decision model returns natively, and what NInfer reads. It is not a
    repair of a malformed distribution: nothing is invented, and the relative mass between options
    is the model's own.
    """
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt_text}],
        "max_tokens": 1,
        "temperature": 1.0,
        "logprobs": True,
        "top_logprobs": TOP_LOGPROBS,
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
        "structured_outputs": {"choice": allowed_letters},
    }).encode()
    req = urllib.request.Request(VLLM, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        if e.code == 400 and _CONTEXT_ERROR.search(detail):
            raise Unprocessable(f"over the server's context limit: {detail}") from e
        if e.code == 413:
            raise Unprocessable(f"over the server's request size limit: {detail}") from e
        raise UpstreamError(f"vLLM HTTP {e.code}: {detail}", e.code if e.code in (401, 403, 429) else 502) from e
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise UpstreamError(f"vLLM unreachable or unreadable: {type(e).__name__}: {e}") from e


def answer_for(task_state, decision):
    qtype = decision.get("type")
    instructions = decision.get("instructions") or ""
    criteria = decision.get("criteria")
    opts = options_from(criteria, qtype)

    resp = call_vllm(build_prompt(task_state, instructions, opts),
                     [letter for letter, _l, _d in opts])
    usage = resp.get("usage") or {}
    try:
        lp = resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        # a list of pairs, not a dict keyed by text: some vocabularies (Gemma-4) have two tokens that decode to the
        # same letter, and a dict keeps only the last one; letter_probs sums the mass per letter
        top = [(d["token"], d["logprob"]) for d in lp]
    except (KeyError, IndexError, TypeError):
        return None, usage, "vLLM returned no top_logprobs at the answer slot"

    probs = letter_probs(top, len(opts))
    if probs is None:
        return None, usage, "could not recover an option-letter distribution from the answer slot"
    temp = TEMPS[qtype]
    if temp != 1.0:
        z = {i: max(p, 1e-12) ** (1.0 / temp) for i, p in probs.items()}
        zs = sum(z.values())
        probs = {i: v / zs for i, v in z.items()}

    if qtype == "noul":
        # criteria order is the record's; find which letter carried "true"
        true_letter = next(i for i, (_l, lab, _d) in enumerate(opts) if str(lab).lower() == "true")
        p_yes = probs.get(true_letter, 0.0)
        return {"type": "noul", "noul": p_yes}, usage, None

    dist = {lab: probs.get(i, 0.0) for i, (_l, lab, _d) in enumerate(opts)}
    total = sum(dist.values())
    if total <= 0:
        return None, usage, "empty distribution"
    dist = {k: v / total for k, v in dist.items()}
    best = max(dist, key=dist.get)
    return {"type": qtype, "choice": best, "probabilities": dist}, usage, None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):          # keep the log to one line per request
        sys.stderr.write("shim %s\n" % (fmt % args))

    def _send(self, code, obj):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._send(200, {"models": [{"id": MODEL,
                                         "readout": "native option-logit over vLLM"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/v1/systemone"):
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            questions = body.get("questions") if isinstance(body, dict) else None
            decision = questions.get("decision") if isinstance(questions, dict) else None
            if not isinstance(decision, dict):
                raise ValueError("expected a JSON object with questions.decision")
        except Exception as e:
            self._send(400, {"error": f"bad request body: {e}"})
            return
        try:
            context = server_context()
            if context < MIN_CONTEXT:
                raise UpstreamError(f"the server's max_model_len is {context}, below SHIM_MIN_CONTEXT={MIN_CONTEXT};"
                                    f" restart vLLM with a larger --max-model-len", 503)
            ans, usage, err = answer_for(body.get("state") or "", decision)
        except Unprocessable as e:
            sys.stderr.write(f"shim 422 {e}\n")
            self._send(422, {"error": str(e)})
            return
        except UpstreamError as e:
            self._send(e.status, {"error": str(e)})
            return
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})
            return
        if err:
            self._send(502, {"error": err})
            return
        self._send(200, {
            "model": MODEL,
            "answers": {"decision": ans},
            # KEY NAMES MATTER: the harness reads usage.prompt_tokens / usage.completion_tokens.
            # Returning input_tokens/output_tokens instead would silently lose the
            # token accounting the board prices from.
            "usage": {"prompt_tokens": usage.get("prompt_tokens"),
                      "completion_tokens": usage.get("completion_tokens"),
                      "total_tokens": usage.get("total_tokens")},
        })


if __name__ == "__main__":
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"native-readout shim on {HOST}:{PORT} -> {VLLM} (model {MODEL})", flush=True)
    print(f"readout temperatures: {describe_temperatures()}", flush=True)
    srv.serve_forever()
