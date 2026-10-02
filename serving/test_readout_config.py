#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Torchcast AI.
"""Checks torchcast_shim's readout configuration and per-type temperatures, and that the served
code applies no band remap to Noul probabilities. Standard library only, no GPU.

    python3 serving/test_readout_config.py

Four kinds of check:
  1. Loading: readout_config.json gives Choice 1.0, Score 1.0, Noul 0.2; SHIM_T_<TYPE> and SHIM_TEMPERATURE override
     it in that order; a missing or malformed file, or a bad value, raises ConfigError, and the shim and the decision
     server refuse to start (non-zero exit, message on stderr). A good start prints the effective temperatures.
  2. Per-type temperatures end to end: through the shim and the decision server, each question type is tempered with
     its own T and only its own T.
  3. No band remap: the served files contain no snap code or switch, and a Noul P(yes) inside 0.2-0.8 is returned as the
     model's tempered probability, and no environment switch changes that.
  4. The shipped readout_config.json carries exactly the temperatures the README and the model card state.
"""
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "readout_config.json")
SHIPPED = {"choice": 1.0, "noul": 0.2, "score": 1.0}

failures = 0


def check(name, good, detail=""):
    global failures
    failures += not good
    print(f"{'PASS' if good else 'FAIL'}  {name}" + ("" if good else f"  {str(detail)[:300]}"))


class FakeVLLM(BaseHTTPRequestHandler):
    """vLLM under a structured-output mask. MOCK_PICK=<letter> puts most mass on that letter (logprob -0.1, the others
    -3.0 - 0.1 * position); MOCK_EVEN gives every allowed letter the same logprob."""

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"object": "list", "data": [{"id": "mock", "max_model_len": 262144}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = req["messages"][-1]["content"]
        allowed = req["structured_outputs"]["choice"]
        if "MOCK_EVEN" in text:
            top = [{"token": a, "logprob": math.log(1.0 / len(allowed))} for a in allowed]
        else:
            m = re.search(r"MOCK_PICK=([A-Z])", text)
            pick = m.group(1) if m else allowed[0]
            top = [{"token": a, "logprob": -0.1 if a == pick else -3.0 - 0.1 * i} for i, a in enumerate(allowed)]
        self._send(200, {"choices": [{"logprobs": {"content": [{"token": top[0]["token"], "logprob": top[0]["logprob"],
                                                                 "top_logprobs": top[:20]}]}}],
                         "usage": {"prompt_tokens": len(text) // 4, "completion_tokens": 1}})


def serve(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv.server_address[1]


# The module-level configuration must come from readout_config.json alone, whatever the caller's shell exports.
# A switch name a band remap would plausibly use is set, to show that nothing reads it.
for key in ("SHIM_T_CHOICE", "SHIM_T_NOUL", "SHIM_T_SCORE", "SHIM_TEMPERATURE", "SHIM_READOUT_CONFIG"):
    os.environ.pop(key, None)
os.environ["SHIM_NOUL_SNAP"] = "1"
vllm_port = serve(FakeVLLM)
os.environ.update({"SHIM_VLLM": f"http://127.0.0.1:{vllm_port}/v1/chat/completions", "SHIM_MODEL": "mock"})


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


shim = load("torchcast_shim_under_test", os.path.join(HERE, "torchcast_shim.py"))
server = load("decision_server_under_test", os.path.join(HERE, "decision_server.py"))
shim_port, server_port = serve(shim.Handler), serve(server.Handler)

# ---- 1. loading
check("the shipped readout_config.json gives Choice 1.0, Score 1.0, Noul 0.2",
      shim.load_temperatures(CONFIG, {})[0] == SHIPPED, shim.load_temperatures(CONFIG, {}))
check("by default (no SHIM_T_* / SHIM_TEMPERATURE / SHIM_READOUT_CONFIG) the shim uses readout_config.json",
      shim.TEMPS == SHIPPED and set(shim.TEMP_SOURCES.values()) == {"readout_config.json"},
      (shim.TEMPS, shim.TEMP_SOURCES))
check("the decision server reads the same temperatures through the shim", server.shim.TEMPS == SHIPPED, server.shim.TEMPS)
t, src = shim.load_temperatures(CONFIG, {"SHIM_T_NOUL": "0.5"})
check("SHIM_T_NOUL overrides Noul only", t == {"choice": 1.0, "noul": 0.5, "score": 1.0} and src["noul"] == "SHIM_T_NOUL"
      and src["choice"] == "readout_config.json", (t, src))
t, src = shim.load_temperatures(CONFIG, {"SHIM_TEMPERATURE": "2", "SHIM_T_SCORE": "1.5"})
check("SHIM_T_<TYPE> wins over SHIM_TEMPERATURE, which covers the other types",
      t == {"choice": 2.0, "noul": 2.0, "score": 1.5} and src == {"choice": "SHIM_TEMPERATURE", "noul": "SHIM_TEMPERATURE",
                                                                    "score": "SHIM_T_SCORE"}, (t, src))
t, _ = shim.load_temperatures(None, {"SHIM_T_CHOICE": "", "SHIM_TEMPERATURE": ""})
check("empty environment values are ignored", t == SHIPPED, t)
t, _ = shim.load_temperatures(CONFIG, {"SHIM_T_CHOICE": "1.0", "SHIM_T_SCORE": "1.0", "SHIM_T_NOUL": "0.2"})
check("the explicit values SHIM_T_CHOICE=1.0 SHIM_T_SCORE=1.0 SHIM_T_NOUL=0.2 equal the file", t == SHIPPED, t)

tmp = tempfile.mkdtemp()


def config_file(name, text):
    path = os.path.join(tmp, name)
    with open(path, "w") as f:
        f.write(text)
    return path


custom = config_file("custom.json", json.dumps({"temperature": {"choice": 1.2, "noul": 0.7, "score": 0.9}}))
t, src = shim.load_temperatures(None, {"SHIM_READOUT_CONFIG": custom})
check("SHIM_READOUT_CONFIG points the shim at another file", t == {"choice": 1.2, "noul": 0.7, "score": 0.9}
      and src["noul"] == "custom.json", (t, src))

bad = {
    "a missing file": (os.path.join(tmp, "missing.json"), {}),
    "invalid JSON": (config_file("broken.json", "{temperature: "), {}),
    "a JSON list": (config_file("list.json", "[1, 2]"), {}),
    "no temperature object": (config_file("none.json", json.dumps({"readout": "x"})), {}),
    "a missing type": (config_file("two.json", json.dumps({"temperature": {"choice": 1, "score": 1}})), {}),
    "a string value": (config_file("str.json", json.dumps({"temperature": {"choice": 1, "noul": "0.3", "score": 1}})), {}),
    "a boolean value": (config_file("bool.json", json.dumps({"temperature": {"choice": True, "noul": 0.3, "score": 1}})), {}),
    "a zero value": (config_file("zero.json", json.dumps({"temperature": {"choice": 1, "noul": 0, "score": 1}})), {}),
    "a negative value": (config_file("neg.json", json.dumps({"temperature": {"choice": 1, "noul": -0.3, "score": 1}})), {}),
    "an infinite value": (config_file("inf.json", '{"temperature": {"choice": 1, "noul": Infinity, "score": 1}}'), {}),
    "a NaN value": (config_file("nan.json", '{"temperature": {"choice": 1, "noul": NaN, "score": 1}}'), {}),
    "a non-numeric SHIM_T_NOUL": (CONFIG, {"SHIM_T_NOUL": "sharp"}),
    "SHIM_T_NOUL=0": (CONFIG, {"SHIM_T_NOUL": "0"}),
    "SHIM_TEMPERATURE=-1": (CONFIG, {"SHIM_TEMPERATURE": "-1"}),
    "a broken file even when the environment sets every type": (
        config_file("broken2.json", "not json"), {"SHIM_T_CHOICE": "1", "SHIM_T_NOUL": "0.3", "SHIM_T_SCORE": "1"}),
}
for what, (path, env) in bad.items():
    try:
        shim.load_temperatures(path, env)
        check(f"{what} is refused (ConfigError)", False, "loaded without an error")
    except shim.ConfigError:
        check(f"{what} is refused (ConfigError)", True)


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("SHIM_T") and k not in ("SHIM_READOUT_CONFIG",)}
    env.update(extra)
    return env


def start_and_fail(script, **extra):
    """Run a server script that must stop at once; (return code, stderr)."""
    try:
        r = subprocess.run([sys.executable, os.path.join(HERE, script)], env=clean_env(**extra),
                           capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return None, "it started and kept running"
    return r.returncode, r.stderr


for script in ("torchcast_shim.py", "decision_server.py"):
    rc, err = start_and_fail(script, SHIM_READOUT_CONFIG=os.path.join(tmp, "missing.json"), SHIM_PORT="0", TORCHCAST_PORT="0")
    check(f"{script} refuses to start without its readout config", rc not in (0, None)
          and "cannot read the readout config" in err, (rc, err[-300:]))
    rc, err = start_and_fail(script, SHIM_READOUT_CONFIG=bad["a missing type"][0], SHIM_PORT="0", TORCHCAST_PORT="0")
    check(f"{script} refuses to start with a malformed readout config", rc not in (0, None)
          and "has no temperature for 'noul'" in err, (rc, err[-300:]))


def first_lines(script, n, **extra):
    """Start a server script, return its first n stdout lines, stop it."""
    p = subprocess.Popen([sys.executable, os.path.join(HERE, script)], env=clean_env(**extra),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    lines, deadline = [], time.time() + 20
    try:
        while len(lines) < n and time.time() < deadline and p.poll() is None:
            line = p.stdout.readline()
            if line:
                lines.append(line.rstrip("\n"))
    finally:
        p.kill()
        p.wait()
    return lines


want = "readout temperatures: choice T=1 (readout_config.json), noul T=0.2 (readout_config.json), score T=1 (readout_config.json)"
out = first_lines("torchcast_shim.py", 2, SHIM_PORT="0")
check("the shim prints its effective temperatures at startup", want in out, out)
out = first_lines("decision_server.py", 2, TORCHCAST_PORT="0")
check("the decision server prints its effective temperatures at startup", want in out, out)
out = first_lines("torchcast_shim.py", 2, SHIM_PORT="0", SHIM_T_NOUL="1.0")
check("an environment override is printed with its source", any("noul T=1 (SHIM_T_NOUL)" in line for line in out), out)


# ---- 2. per-type temperatures end to end
def tempered(logprobs, T):
    z = [math.exp(v) for v in logprobs]
    s = sum(z)
    p = [(v / s) ** (1 / T) for v in z]
    s = sum(p)
    return [v / s for v in p]


def mock_logprobs(n, pick=0):
    return [-0.1 if i == pick else -3.0 - 0.1 * i for i in range(n)]


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def one(port, qtype, criteria, marker):
    return post(port, {"state": "s", "questions": {"decision": {"type": qtype, "instructions": f"Decide. {marker}",
                                                                "criteria": criteria}}})


CHOICE = {"a": "first", "b": "second", "c": "third"}
SCORE = ["low", "mid", "high"]
NOUL = {"false": "No.", "true": "Yes."}


def close(a, b):
    return all(abs(x - y) < 1e-9 for x, y in zip(a, b)) and len(a) == len(b)


def run_types(port, temps):
    """Ask one question of each type and compare with an independent computation at `temps`."""
    st, b = one(port, "choice", CHOICE, "MOCK_PICK=A")
    got = list(b["answers"]["decision"]["probabilities"].values()) if st == 200 else []
    ok_c = close(got, tempered(mock_logprobs(3), temps["choice"]))
    st, b = one(port, "score", SCORE, "MOCK_PICK=C")
    got = [b["answers"]["decision"]["probabilities"][str(i)] for i in range(3)] if st == 200 else []
    ok_s = close(got, tempered(mock_logprobs(3, 2), temps["score"]))
    st, b = one(port, "noul", NOUL, "MOCK_PICK=B")      # B is "true": P(yes) is the second option's probability
    got = b["answers"]["decision"]["noul"] if st == 200 else -1
    ok_n = abs(got - tempered(mock_logprobs(2, 1), temps["noul"])[1]) < 1e-9
    return ok_c, ok_s, ok_n


for label, port in (("shim", shim_port), ("decision server", server_port)):
    ok = run_types(port, SHIPPED)
    check(f"{label}: with the shipped config Choice is read at T 1.0, Score at T 1.0, Noul at T 0.2", all(ok), ok)
    saved = dict(shim.TEMPS)
    shim.TEMPS = server.shim.TEMPS = {"choice": 1.0, "noul": 2.0, "score": 1.0}
    ok = run_types(port, shim.TEMPS)
    check(f"{label}: changing the Noul temperature changes Noul only", all(ok), ok)
    shim.TEMPS = server.shim.TEMPS = {"choice": 0.5, "noul": 1.0, "score": 3.0}
    ok = run_types(port, shim.TEMPS)
    check(f"{label}: Choice and Score each use their own temperature", all(ok), ok)
    shim.TEMPS = server.shim.TEMPS = saved

st, b = post(server_port, {"state": "s", "questions": {
    "c": {"type": "choice", "instructions": "MOCK_PICK=A", "criteria": CHOICE},
    "n": {"type": "noul", "instructions": "MOCK_PICK=B", "criteria": NOUL},
    "s": {"type": "score", "instructions": "MOCK_PICK=C", "criteria": SCORE}}})
check("decision server: three types in one request are each tempered with their own T",
      st == 200 and close(list(b["answers"]["c"]["probabilities"].values()), tempered(mock_logprobs(3), 1.0))
      and abs(b["answers"]["n"]["noul"] - tempered(mock_logprobs(2, 1), 0.2)[1]) < 1e-9
      and close([b["answers"]["s"]["probabilities"][str(i)] for i in range(3)], tempered(mock_logprobs(3, 2), 1.0)), b)

# ---- 3. no band remap
served = ["torchcast_shim.py", "decision_server.py", "readout_config.json"]
hits = {f: [i + 1 for i, line in enumerate(open(os.path.join(HERE, f), encoding="utf-8")) if "snap" in line.lower()]
        for f in served}
check("no 'snap' anywhere in the served files (code, switches, comments, config)", not any(hits.values()), hits)
check("the shim has no snap attributes", not [a for a in dir(shim) if "snap" in a.lower()], dir(shim))
check("the decision server has no snap attributes", not [a for a in dir(server) if "snap" in a.lower()])
for label, port in (("shim", shim_port), ("decision server", server_port)):
    st, b = one(port, "noul", NOUL, "MOCK_EVEN")
    got = b["answers"]["decision"]["noul"] if st == 200 else None
    check(f"{label}: a Noul P(yes) of 0.5 stays 0.5 (not moved to a band edge)",
          got is not None and abs(got - 0.5) < 1e-12, (st, b))
# ---- 4. the shipped file
with open(CONFIG, encoding="utf-8") as f:
    shipped = json.load(f)
check("readout_config.json: temperature is exactly {choice 1.0, noul 0.2, score 1.0} and thinking stays off",
      shipped["temperature"] == SHIPPED and shipped.get("enable_thinking") is False, shipped)

print(f"\n{'all passed' if not failures else f'{failures} failed'}")
sys.exit(1 if failures else 0)
