"""vLLM side of dlm's bench/BASELINES.md comparison.

Prefill tok/s = batch * PROMPT / t(1 token); decode tok/s = batch * (GEN - 1)
/ (t(GEN) - t(1)). Random token ids per sequence, greedy, ignore_eos, prefix
caching off; median of RUNS after one warm-up per shape.

    python vllm_bench.py float32 [4,8,16]

The optional second argument picks the batch sizes. Each batch's work starts
with a `BEGIN batch N` line, which `bench/power.py` uses to time it; a
final `BEGIN shutdown` keeps teardown out of the last batch's window.
"""
import os
import random
import statistics
import sys
import time

PROMPT, GEN, RUNS = 256, 128, 3
BATCHES = [1, 4, 8, 16]


def main():
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    dtype = sys.argv[1]
    batches = [int(b) for b in sys.argv[2].split(",")] if len(sys.argv) > 2 else BATCHES
    llm = LLM(
        model=os.path.expanduser("~/models/Qwen2.5-0.5B-Instruct"),
        dtype=dtype,
        max_model_len=1024,
        max_num_seqs=16,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.78,
        enable_prefix_caching=False,
    )
    rng = random.Random(0)

    def timed(batch, n):
        prompts = [
            TokensPrompt(prompt_token_ids=[rng.randrange(100, 30000) for _ in range(PROMPT)])
            for _ in range(batch)
        ]
        sp = SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True)
        t = time.perf_counter()
        llm.generate(prompts, sp, use_tqdm=False)
        return time.perf_counter() - t

    for batch in batches:
        print(f"BEGIN batch {batch}", flush=True)
        timed(batch, 1)
        timed(batch, GEN)
        t1 = statistics.median(timed(batch, 1) for _ in range(RUNS))
        tg = statistics.median(timed(batch, GEN) for _ in range(RUNS))
        print(
            f"RESULT {dtype} batch {batch}: prefill {batch * PROMPT / t1:.0f} tok/s, "
            f"decode {batch * (GEN - 1) / (tg - t1):.1f} tok/s",
            flush=True,
        )
    print("BEGIN shutdown", flush=True)  # closes the last batch's window


if __name__ == "__main__":
    main()
