#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Nood Co and contributors; modifications by Torchcast AI.
"""Torchcast application decision server. Imports torchcast_shim; use TORCHCAST_* server settings. Derived from Cygnet (MIT); see LICENSE-serving-MIT and ../NOTICE.md."""
from __future__ import annotations

import hmac
import importlib.util
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_spec = importlib.util.spec_from_file_location("torchcast_shim", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                          "torchcast_shim.py"))
shim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shim)

HOST = os.environ.get("TORCHCAST_HOST", "127.0.0.1")
PORT = int(os.environ.get("TORCHCAST_PORT", "8010"))
API_KEY = os.environ.get("TORCHCAST_API_KEY", "")
MAX_PARALLEL = int(os.environ.get("TORCHCAST_MAX_PARALLEL", "8"))
GROUP_SIZE = int(os.environ.get("TORCHCAST_GROUP_SIZE", str(min(20, shim.TOP_LOGPROBS))))
MAX_BODY = int(os.environ.get("TORCHCAST_MAX_BODY", str(16 * 1024 * 1024)))
MODEL_NAME = os.environ.get("TORCHCAST_MODEL_NAME", shim.MODEL)
MODEL_DESCRIPTION = os.environ.get("TORCHCAST_MODEL_DESCRIPTION",
                                   "Torchcast Decision 12B")
MODEL_RELEASE_DATE = os.environ.get("TORCHCAST_MODEL_RELEASE_DATE", "2026-10-02")
ALLOW_NO_KEY = os.environ.get("TORCHCAST_ALLOW_NO_KEY", "") == "1"
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
_letters_read = min(len(shim.LETTERS), shim.TOP_LOGPROBS)   # options one pass can return a probability for
if not 2 <= GROUP_SIZE <= _letters_read or -(-MAX_CHOICE_OPTIONS // GROUP_SIZE) > _letters_read:
    raise SystemExit(f"TORCHCAST_GROUP_SIZE must be between {-(-MAX_CHOICE_OPTIONS // _letters_read)} and {_letters_read}, "
                     f"so that {MAX_CHOICE_OPTIONS} options fit in one final pass over the group winners")

_passes = ThreadPoolExecutor(max_workers=MAX_PARALLEL)    # group passes within a question
_in_flight = threading.BoundedSemaphore(MAX_PARALLEL)      # every vLLM request, across all requests


def _text(value):
    """A description as the model sees it: text as it is, anything else as compact JSON."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _blank(value):
    return value is None or (isinstance(value, str) and not value.strip())


def parse_question(name, q):
    """(type, instructions, [(label, shown text)], legend or None), or Unprocessable naming what is wrong."""
    where = f"questions.{name}"
    if not isinstance(q, dict):
        raise shim.Unprocessable(f"{where} must be an object")
    qtype = q.get("type")
    instructions = q.get("instructions") or ""          # as reference_shim.answer_for
    criteria = q.get("criteria")
    if qtype == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise shim.Unprocessable(f"{where}.criteria must be a non-empty object of options")
        if len(criteria) > MAX_CHOICE_OPTIONS:
            raise shim.Unprocessable(f"{where} has {len(criteria)} options; a Choice takes at most {MAX_CHOICE_OPTIONS}")
        items = []
        for label, desc in criteria.items():
            if not isinstance(desc, (str, dict, list)) and desc is not None:
                raise shim.Unprocessable(f"{where}.criteria.{label} must be text, JSON or null")
            items.append((label, label if _blank(desc) else desc if isinstance(desc, str) else f"{label}: {_text(desc)}"))
        return qtype, instructions, items, None
    if qtype == "score":
        if not isinstance(criteria, list) or not criteria:
            raise shim.Unprocessable(f"{where}.criteria must be a non-empty list of levels")
        if len(criteria) > MAX_SCORE_LEVELS:
            raise shim.Unprocessable(f"{where} has {len(criteria)} levels; a Score takes at most {MAX_SCORE_LEVELS}")
        items, legend = [], {}
        for i, desc in enumerate(criteria):
            if not isinstance(desc, (str, dict, list)) and desc is not None:
                raise shim.Unprocessable(f"{where}.criteria[{i}] must be text, JSON or null")
            items.append((str(i), f"Level {i}" if _blank(desc) else _text(desc)))
            legend[str(i)] = "" if desc is None else desc
        return qtype, instructions, items, legend
    if qtype == "noul":
        if criteria is None:
            criteria = {}
        if not isinstance(criteria, dict):
            raise shim.Unprocessable(f"{where}.criteria must be an object with 'true' and 'false' descriptions")
        sides = {}
        for key, desc in criteria.items():
            side = str(key).lower()
            if side not in ("true", "false") or side in sides:
                raise shim.Unprocessable(f"{where}.criteria takes only 'true' and 'false', once each")
            if not isinstance(desc, (str, dict, list)) and desc is not None:
                raise shim.Unprocessable(f"{where}.criteria.{key} must be text, JSON or null")
            sides[side] = desc
        items = [("false", "No" if _blank(sides.get("false")) else _text(sides["false"])),
                 ("true", "Yes" if _blank(sides.get("true")) else _text(sides["true"]))]
        return qtype, instructions, items, None
    raise shim.Unprocessable(f"{where}.type must be 'choice', 'score' or 'noul', not {qtype!r}")


def read_pass(state, instructions, items):
    """One benchmark-path pass over at most GROUP_SIZE options: raw probabilities (before the temperature) and usage.
    A single option needs no pass."""
    if len(items) == 1:
        return [1.0], {}
    opts = [(shim.LETTERS[i], label, text) for i, (label, text) in enumerate(items)]
    with _in_flight:
        resp = shim.call_vllm(shim.build_prompt(state, instructions, opts), [letter for letter, _l, _t in opts])
    usage = resp.get("usage") or {}
    try:
        # a list of (token, logprob) pairs: Gemma-4 has two tokens that decode to each letter (reference_shim.answer_for)
        top = [(d["token"], d["logprob"]) for d in resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]]
    except (KeyError, IndexError, TypeError):
        raise shim.UpstreamError("vLLM returned no top_logprobs at the answer slot")
    probs = shim.letter_probs(top, len(opts))
    if probs is None:
        raise shim.UpstreamError("could not recover an option-letter distribution from the answer slot")
    return [probs.get(i, 0.0) for i in range(len(opts))], usage


def temper(probs, qtype):
    """reference_shim's calibration: p^(1/T) renormalised, with the same floor; T = 1 leaves it unchanged.
    Torchcast: T is the question type's readout temperature (shim.TEMPS, from readout_config.json or the env)."""
    t = shim.TEMPS[qtype]
    if t == 1.0:
        return list(probs)
    z = [max(p, 1e-12) ** (1.0 / t) for p in probs]
    s = sum(z)
    return [v / s for v in z]


def distribution(state, instructions, items, qtype):
    """Calibrated probabilities over `items` (list of (label, text)), and summed usage."""
    tokens = {"input": 0, "output": 0}

    def count(usage):
        tokens["input"] += usage.get("prompt_tokens") or 0
        tokens["output"] += usage.get("completion_tokens") or 0

    if len(items) <= GROUP_SIZE:
        raw, usage = read_pass(state, instructions, items)
        count(usage)
        return temper(raw, qtype), tokens
    # near-equal groups, in the given order, so no winner reaches the final pass through a small group
    m = -(-len(items) // GROUP_SIZE)
    bounds = [round(i * len(items) / m) for i in range(m + 1)]
    groups = [items[bounds[i]:bounds[i + 1]] for i in range(m)]
    reads = list(_passes.map(lambda g: read_pass(state, instructions, g), groups))
    within = []
    for raw, usage in reads:
        count(usage)
        within.append(raw)
    # each group's winner is shown with its own description, so the final pass compares their content
    winners = [group[max(range(len(group)), key=lambda i: p[i])] for group, p in zip(groups, within)]
    between, usage = read_pass(state, instructions, winners)
    count(usage)
    composed = [between[g] * p for g, p_group in enumerate(within) for p in p_group]
    s = sum(composed)
    if s <= 0:
        raise shim.UpstreamError("the grouped readout gave every option zero probability")
    return temper([p / s for p in composed], qtype), tokens


def confidence(probs):
    k = len(probs)
    return 1.0 if k == 1 else max(0.0, min(1.0, (k * max(probs) - 1.0) / (k - 1.0)))


def answer(state, qtype, instructions, items, legend):
    probs, tokens = distribution(state, instructions, items, qtype)
    if qtype == "noul":
        return {"type": "noul", "noul": probs[1]}, tokens          # items are (false, true)
    s = sum(probs)                     # reference_shim renormalises Choice and Score once more; so do we, for identity
    probs = [p / s for p in probs]
    if qtype == "score":
        return {"type": "score",
                "score": sum(i * p for i, p in enumerate(probs)),
                "legend": legend,
                "probabilities": {str(i): p for i, p in enumerate(probs)},
                "confidence": confidence(probs)}, tokens
    dist = {label: p for (label, _t), p in zip(items, probs)}
    return {"type": "choice", "choice": max(dist, key=dist.get), "probabilities": dist,
            "confidence": confidence(probs)}, tokens


def evaluate(body):
    """(status, response object) for one parsed request body."""
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise shim.Unprocessable("questions must be a non-empty object of named questions")
    parsed = {name: parse_question(name, q) for name, q in questions.items()}
    context = shim.server_context()
    if context < shim.MIN_CONTEXT:
        raise shim.UpstreamError(f"the server's max_model_len is {context}, below SHIM_MIN_CONTEXT={shim.MIN_CONTEXT};"
                                 f" restart vLLM with a larger --max-model-len", 503)
    state = body.get("state") or ""                    # as reference_shim

    def one(name):
        qtype, instructions, items, legend = parsed[name]
        try:
            return name, answer(state, qtype, instructions, items, legend), None
        except (shim.Unprocessable, shim.UpstreamError) as e:
            return name, None, e

    with ThreadPoolExecutor(max_workers=max(1, min(MAX_PARALLEL, len(parsed)))) as pool:
        results = list(pool.map(one, parsed))
    errors = [e for _n, _r, e in results if e is not None]
    # vLLM's own 401, 403 and 429 pass through first (the server's auth or load, which the caller must see); then a 422,
    # which a retry cannot fix; then any other upstream failure
    passed = [e for e in errors if isinstance(e, shim.UpstreamError) and e.status in (401, 403, 429)]
    unprocessable = [e for e in errors if isinstance(e, shim.Unprocessable)]
    if errors:
        raise (passed or unprocessable or errors)[0]
    answers, input_tokens, output_tokens = {}, 0, 0
    for name, (ans, tokens), _e in results:
        answers[name] = ans
        input_tokens += tokens["input"]
        output_tokens += tokens["output"]
    return {"model": MODEL_NAME, "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("decision-server %s\n" % (fmt % args))

    def _send(self, code, obj):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _authorised(self):
        if not API_KEY:
            return True
        given = self.headers.get("Authorization") or ""
        return given.startswith("Bearer ") and hmac.compare_digest(given[7:].strip().encode(), API_KEY.encode())

    def do_GET(self):
        if not self.path.startswith("/v1/models"):
            return self._send(404, {"error": "not found"})
        if not self._authorised():
            return self._send(401, {"error": "missing or invalid API key"})
        self._send(200, {"models": [{"name": MODEL_NAME, "description": MODEL_DESCRIPTION,
                                     "release_date": MODEL_RELEASE_DATE}]})

    def do_POST(self):
        if not self.path.startswith("/v1/systemone"):
            return self._send(404, {"error": "not found"})
        length = self.headers.get("Content-Length")
        refuse = None
        if not self._authorised():
            refuse = (401, "missing or invalid API key")
        elif length is None:
            refuse = (411, "a Content-Length header is required")
        elif not length.strip().isdecimal():
            refuse = (400, "Content-Length is not a non-negative integer")
        elif int(length) > MAX_BODY:
            refuse = (413, f"request body over {MAX_BODY} bytes")
        if refuse:
            self.close_connection = True           # the unread body must not be taken for the next request
            return self._send(refuse[0], {"error": refuse[1]})
        raw = self.rfile.read(int(length))
        try:
            body = json.loads(raw or b"{}")
        except ValueError as e:
            return self._send(400, {"error": f"request body is not JSON: {e}"})
        if not isinstance(body, dict):
            return self._send(422, {"error": "request body must be a JSON object"})
        try:
            self._send(200, evaluate(body))
        except shim.Unprocessable as e:
            sys.stderr.write(f"decision-server 422 {e}\n")
            self._send(422, {"error": str(e)})
        except shim.UpstreamError as e:
            self._send(e.status, {"error": str(e)})
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    if HOST not in ("127.0.0.1", "localhost", "::1") and not API_KEY and not ALLOW_NO_KEY:
        raise SystemExit(f"decision-server: refusing to listen on {HOST} without TORCHCAST_API_KEY "
                         f"(set TORCHCAST_ALLOW_NO_KEY=1 when something in front of it checks access)")
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"decision server on {HOST}:{PORT} -> {shim.VLLM} (model {shim.MODEL}, groups of {GROUP_SIZE})", flush=True)
    print(f"readout temperatures: {shim.describe_temperatures()}", flush=True)
    srv.serve_forever()
