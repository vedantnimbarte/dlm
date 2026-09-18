#!/usr/bin/env python3
"""Check dlm against llama.cpp on the same .gguf file.

A GGUF file is quantized, and the quantization is the part most easily decoded
wrongly: a misread scale or a swapped nibble half produces weights that are
close enough to generate fluent text and wrong enough to be a different model.
Nothing in dlm's own test suite can catch that, because dlm would be consistent
with itself either way. llama.cpp wrote the format, so it is the reference.

Two things are compared, both on the file itself rather than on a conversion of
it:

* **Tokenization** — the ids each side turns text into.
* **Greedy generation** — the ids each side produces from the same prompt ids,
  which fails on any decoding error that moves the argmax.

Usage:

    # start the reference server (from a llama.cpp release build)
    llama-server -m model.gguf --port 8099 -c 512

    python tools/llamacpp_parity.py --model model.gguf --server http://127.0.0.1:8099

Needs a dlm binary (default `target/release/dlm`) and a running `llama-server`
for the same file. No Python dependencies beyond the standard library.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

# Text chosen to move through the parts a quantized model gets wrong quietly:
# ordinary prose, code with indentation, digits, and non-Latin scripts.
DEFAULT_TEXTS = [
    "The capital of France is",
    "def add(a, b):\n    # returns the sum\n    return a + b\n",
    "In 1969, 12 people walked on the Moon.",
    "Le café coûte 4,50 € — c'est cher.",
    "東京は日本の首都です。",
]


def post(server: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{server}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def run_dlm(dlm: Path, args: list[str]) -> subprocess.CompletedProcess:
    """Run the dlm binary, decoding its output as UTF-8.

    Windows resolves a bare name to `name.exe` only sometimes, and decodes a
    child's output with the console code page unless told otherwise — which
    mangles any non-ASCII piece a tokenizer prints back.
    """
    exe = dlm
    if not exe.exists() and exe.with_suffix(".exe").exists():
        exe = exe.with_suffix(".exe")
    return subprocess.run(
        [str(exe), *args],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )


def dlm_tokens(dlm: Path, model: Path, text: str) -> list[int]:
    """`dlm tokenize`'s ids for `text`, read out of its report."""
    run = run_dlm(dlm, ["tokenize", "--tokenizer", str(model), "--text", text])
    if run.returncode != 0:
        sys.exit(f"dlm tokenize failed:\n{run.stdout}\n{run.stderr}")
    match = re.search(r"^ids\s*:\s*\[(.*)\]", run.stdout, re.M)
    if not match:
        sys.exit(f"could not find ids in dlm's output:\n{run.stdout}")
    return [int(x) for x in match.group(1).split(",") if x.strip()]


def dlm_generate(dlm: Path, model: Path, prompt: list[int], n: int, device: str) -> list[int]:
    """The ids dlm generates greedily from `prompt`."""
    run = run_dlm(
        dlm,
        [
            "generate",
            "--model-path", str(model),
            "--prompt", ",".join(str(t) for t in prompt),
            "--max-new-tokens", str(n),
            "--device", device,
        ],
    )
    if run.returncode != 0:
        sys.exit(f"dlm generate failed:\n{run.stdout}\n{run.stderr}")
    match = re.search(r"^generated ids:\s*\[(.*)\]", run.stdout, re.M)
    if not match:
        sys.exit(f"could not find generated ids in dlm's output:\n{run.stdout}")
    return [int(x) for x in match.group(1).split(",") if x.strip()]


def dlm_next_logprobs(dlm: Path, model: Path, ctx: list[int], top: int) -> dict[int, float]:
    """dlm's top-`top` next-token log-probabilities after `ctx`."""
    run = run_dlm(
        dlm,
        [
            "score",
            "--model-path", str(model),
            "--prompt", ",".join(str(t) for t in ctx),
            "--top", str(top),
        ],
    )
    if run.returncode != 0:
        sys.exit(f"dlm score failed:\n{run.stdout}\n{run.stderr}")
    tail = run.stdout.split("next token, most likely first:")[-1]
    out = {}
    for line in tail.splitlines():
        m = re.match(r"\s+(\d+)\s+(-?\d+\.\d+)", line)
        if m:
            out[int(m.group(1))] = float(m.group(2))
    return out


def explain_divergence(
    dlm: Path, model: Path, server: str, ctx: list[int], theirs: int, top: int
) -> tuple[bool, str]:
    """Why greedy decoding split at this point.

    A tie, not a bug, is the common answer: llama.cpp quantizes activations to
    int8 for its dot products where dlm dequantizes the weights and computes in
    f32. The two therefore disagree slightly about every logit, and wherever the
    top two candidates are closer together than that disagreement, either can
    come out on top.

    So the test is not a fixed tolerance. Both distributions are read at this
    position, the disagreement between them is measured on the tokens they share,
    and the split counts as a tie when the gap between the two candidates is
    smaller than that measured noise. A real decoding bug moves one distribution
    away from the other, which shows up as noise far larger than the gap -- or as
    llama.cpp's choice missing from dlm's candidates entirely.
    """
    ours = dlm_next_logprobs(dlm, model, ctx, top)
    if not ours:
        return False, "dlm reported no distribution"
    best_id, best = max(ours.items(), key=lambda kv: kv[1])
    if theirs not in ours:
        return False, f"llama.cpp's choice {theirs} is not in dlm's top {top}"
    gap = best - ours[theirs]

    reply = post(
        server,
        "/completion",
        {"prompt": ctx, "n_predict": 1, "temperature": 1.0, "n_probs": top},
    )
    entries = reply.get("completion_probabilities", [{}])[0].get("top_logprobs", [])
    theirs_lp = {t["id"]: t["logprob"] for t in entries}
    shared = set(ours) & set(theirs_lp)
    noise = max((abs(ours[i] - theirs_lp[i]) for i in shared), default=float("inf"))
    return gap <= noise, (
        f"dlm {best_id} at {best:.4f}, llama.cpp {theirs} at {ours[theirs]:.4f} in dlm's "
        f"own distribution: {gap:.4f} apart, against {noise:.4f} of arithmetic difference "
        f"between the runtimes over {len(shared)} shared candidates"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True, help="the .gguf file both sides read")
    ap.add_argument("--dlm", type=Path, default=Path("target/release/dlm"))
    ap.add_argument("--server", default="http://127.0.0.1:8099", help="a running llama-server")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--predict", type=int, default=24, help="tokens to generate per prompt")
    ap.add_argument("--text", action="append", help="repeatable; defaults to a built-in set")
    ap.add_argument(
        "--top",
        type=int,
        default=8,
        help="candidates to compare when greedy decoding splits",
    )
    args = ap.parse_args()

    texts = args.text or DEFAULT_TEXTS
    print(f"comparing {args.dlm} ({args.device}) against {args.server} on {args.model}")
    ok = True

    for text in texts:
        ours = dlm_tokens(args.dlm, args.model, text)
        # `add_special: false`, so both sides tokenize exactly the text given.
        theirs = post(args.server, "/tokenize", {"content": text, "add_special": False})["tokens"]
        same = ours == theirs
        ok &= same
        print(f"  {'PASS' if same else 'FAIL'}  tokenize {text[:40]!r} -> {len(ours)} ids")
        if not same:
            print(f"        dlm       : {ours}")
            print(f"        llama.cpp : {theirs}")

    for text in texts:
        prompt = post(args.server, "/tokenize", {"content": text, "add_special": False})["tokens"]
        ours = dlm_generate(args.dlm, args.model, prompt, args.predict, args.device)
        reply = post(
            args.server,
            "/completion",
            {
                "prompt": prompt,
                "n_predict": args.predict,
                "temperature": 0,
                "top_k": 1,
                # `return_tokens`, not the per-token probability list: that list
                # drops a token whose bytes are half a character, which looks
                # exactly like a disagreement and is not one.
                "return_tokens": True,
            },
        )
        theirs = reply.get("tokens", [])
        # Either side may stop early on EOS; compare what they both produced.
        n = min(len(ours), len(theirs))
        agree = next((i for i in range(n) if ours[i] != theirs[i]), n)
        if agree == n and n > 0:
            print(f"  PASS  generate {text[:40]!r} -> {n}/{n} ids agree")
            continue
        # Diverged: say whether the two runtimes disagreed, or merely broke a tie
        # the other way.
        tie, why = explain_divergence(
            args.dlm, args.model, args.server, prompt + ours[:agree], theirs[agree], args.top
        )
        ok &= tie
        verdict = "TIE " if tie else "FAIL"
        print(f"  {verdict}  generate {text[:40]!r} -> split at {agree}/{n}: {why}")
        if not tie:
            print(f"        dlm       : {ours}")
            print(f"        llama.cpp : {theirs}")

    print("llama.cpp parity: PASS" if ok else "llama.cpp parity: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
