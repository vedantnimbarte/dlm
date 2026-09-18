#!/usr/bin/env python3
"""Check dlm's forward pass against Hugging Face transformers.

dlm's own tests check it against itself: CPU against GPU, streamed against
resident, one run against the next. All of those pass while the whole model is
wrong in the same way -- a norm in the wrong place, a RoPE base off by a factor,
a mis-sliced fused tensor. This script is the outside reference, and the gate a
family should pass before dlm claims to support it.

It compares log-probabilities, not generated text. Greedy output stays readable
long after the probabilities have drifted, so text agreeing proves much less
than it appears to.

Usage:

    python tools/hf_parity.py --model models/qwen2.5-0.5b
    python tools/hf_parity.py --model models/gemma-3-1b --device gpu

Needs `torch` and `transformers` (dev-only -- dlm itself has no Python
dependency), and a dlm binary, by default `target/release/dlm`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Text that exercises what dlm has been wrong about before: ordinary prose,
# indentation and newlines (the pre-tokenizer), digits, and a long-ish run of
# positions (RoPE). Every one is scored on both sides.
DEFAULT_TEXTS = [
    "The capital of France is Paris, and the capital of Japan is Tokyo.",
    "def add(a, b):\n    # returns the sum\n    return a + b\n",
    "In 1969, 12 people walked on the Moon; the last one left in 1972.",
]


def dlm_score(dlm: Path, model: Path, text: str, device: str, top: int) -> dict:
    """`dlm score` for one text, as the JSON it writes."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "score.json"
        run = subprocess.run(
            [
                str(dlm), "score",
                "--model-path", str(model),
                "--text", text,
                "--device", device,
                "--top", str(top),
                "--json", str(out),
            ],
            capture_output=True,
            text=True,
        )
        if run.returncode != 0:
            sys.exit(f"dlm score failed:\n{run.stdout}\n{run.stderr}")
        return json.loads(out.read_text(encoding="utf-8"))


def hf_score(model, tokenizer, ids: list[int], top: int) -> dict:
    """The same numbers from transformers, for the token ids dlm used.

    dlm's ids drive the model on both sides deliberately: a tokenizer
    disagreement would otherwise show up here as a forward-pass disagreement,
    which is a much harder thing to read. Tokenization is compared separately.
    """
    import torch

    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    scored = [logprobs[i, ids[i + 1]].item() for i in range(len(ids) - 1)]
    best = torch.topk(logprobs[-1], top)
    return {
        "logprobs": scored,
        "next": [
            {"id": int(i), "logprob": float(v)}
            for v, i in zip(best.values, best.indices)
        ],
        "tokenizer_ids": tokenizer(
            tokenizer.decode(ids), add_special_tokens=False
        )["input_ids"],
    }


def compare(text: str, ours: dict, theirs: dict, tol: float) -> bool:
    """Print one text's comparison; True when it is within tolerance."""
    deltas = [abs(a - b) for a, b in zip(ours["logprobs"], theirs["logprobs"])]
    worst = max(deltas) if deltas else 0.0
    mean = sum(deltas) / len(deltas) if deltas else 0.0

    # What the model would generate next, which is what decoding depends on.
    ours_next = {t["id"]: t["logprob"] for t in ours["next"]}
    theirs_next = {t["id"]: t["logprob"] for t in theirs["next"]}
    shared = set(ours_next) & set(theirs_next)
    next_worst = max((abs(ours_next[i] - theirs_next[i]) for i in shared), default=0.0)
    top1 = ours["next"][0]["id"] == theirs["next"][0]["id"]
    overlap = len(shared) / len(theirs_next) if theirs_next else 1.0

    ok = worst <= tol and next_worst <= tol and top1
    print(f"  {'PASS' if ok else 'FAIL'}  {text[:48]!r}")
    print(f"        tokens          : {len(ours['tokens'])}")
    print(f"        logprob delta   : max {worst:.4f}, mean {mean:.4f} (tol {tol})")
    print(f"        next-token delta: max {next_worst:.4f} over {len(shared)} shared ids")
    print(f"        top-1 agrees    : {top1}")
    print(f"        top-{len(theirs_next)} overlap    : {overlap:.0%}")
    if ours["tokens"] != theirs["tokenizer_ids"]:
        # Not a forward-pass failure: both sides ran dlm's ids. It does mean the
        # two tokenizers disagree, which changes what a user's prompt becomes.
        print("        tokenizer       : DIFFERS from transformers")
        print(f"          dlm : {ours['tokens']}")
        print(f"          hf  : {theirs['tokenizer_ids']}")
        ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True, help="model directory")
    ap.add_argument("--dlm", type=Path, default=Path("target/release/dlm"))
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--text", action="append", help="repeatable; defaults to a built-in set")
    ap.add_argument("--top", type=int, default=20, help="next-token ids to compare")
    ap.add_argument(
        "--dtype",
        default="float32",
        help="precision transformers runs the reference in. Keep float32: a "
        "bfloat16 reference is not a parity check. Qwen2.5-0.5B, which matches "
        "a float32 reference to 0.0001, differs from a bfloat16 one by up to "
        "0.49 -- all of it the reference's own rounding. If the model does not "
        "fit in float32, run this where it does.",
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=0.05,
        help="largest log-probability difference to accept. dlm reads fp16/bf16 "
        "weights into f32 and transformers is run in f32, so a few hundredths "
        "is arithmetic, not disagreement.",
    )
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # dlm runs first, and every one of its processes has exited before
    # transformers loads the model: both hold the weights in f32, and on a
    # 16 GB machine a 1.5B model in two copies is already an allocation failure.
    texts = args.text or DEFAULT_TEXTS
    print(f"scoring {len(texts)} texts with {args.dlm} on {args.device}")
    ours = [dlm_score(args.dlm, args.model, t, args.device, args.top) for t in texts]

    if args.dtype != "float32":
        print(
            f"warning: a {args.dtype} reference rounds more than dlm does, so a "
            "difference here says nothing about dlm. Measured: Qwen2.5-0.5B is "
            "within 0.0001 of a float32 reference and up to 0.49 from a "
            "bfloat16 one."
        )
    print(f"loading {args.model} in transformers ({args.dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=args.dtype)
    model.eval()

    passed = True
    for text, ours_one in zip(texts, ours):
        theirs = hf_score(model, tokenizer, ours_one["tokens"], args.top)
        passed &= compare(text, ours_one, theirs, args.tol)

    print("parity: PASS" if passed else "parity: FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
