"""Run a benchmark and measure the GPU's power while it works.

    python bench/power.py MARK -- <command...>

Runs the command, echoing its output, while sampling `nvidia-smi` every
100 ms. Every output line containing MARK opens a window; the next such line,
or the end of the command, closes it. For each window it prints the mean
power and utilization of the samples inside it, so model loading and other
set-up stay out of the energy figures:

    WINDOW <label> seconds=<s> watts=<mean W> util=<mean %> samples=<n>

`<label>` is the text after MARK on the line that opened it. Sampler and
command share a clock because the sampler times the lines as they arrive, so
the command must flush each marker line (`dlm` does; run Python with `-u`).
Standard library only, so it runs under any Python, in WSL beside vLLM too.
"""

import shutil
import statistics
import subprocess
import sys
import threading
import time


def nvidia_smi():
    for name in ("nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi"):
        if shutil.which(name) or name.startswith("/"):
            return name
    sys.exit("nvidia-smi not found")


def main():
    if "--" not in sys.argv:
        sys.exit(__doc__)
    split = sys.argv.index("--")
    mark, cmd = sys.argv[1], sys.argv[split + 1 :]
    samples = []  # (time, watts, util)
    smi = subprocess.Popen(
        [nvidia_smi(), "--query-gpu=power.draw,utilization.gpu", "--format=csv,noheader,nounits", "-lms", "100"],
        stdout=subprocess.PIPE,
        text=True,
    )

    def sample():
        for line in smi.stdout:
            try:
                watts, util = (float(x) for x in line.split(","))
            except ValueError:
                continue
            samples.append((time.time(), watts, util))

    threading.Thread(target=sample, daemon=True).start()
    windows = []  # (label, start, end)
    child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    for line in child.stdout:
        print(line, end="", flush=True)
        if mark in line:
            now = time.time()
            if windows and windows[-1][2] is None:
                windows[-1] = (windows[-1][0], windows[-1][1], now)
            windows.append((line.split(mark, 1)[1].strip(), now, None))
    child.wait()
    end = time.time()
    time.sleep(0.3)
    smi.terminate()
    for label, start, stop in windows:
        stop = stop or end
        inside = [(w, u) for t, w, u in samples if start <= t <= stop]
        if not inside:
            continue
        print(
            f"WINDOW {label} seconds={stop - start:.1f} "
            f"watts={statistics.mean(w for w, _ in inside):.1f} "
            f"util={statistics.mean(u for _, u in inside):.0f} samples={len(inside)}",
            flush=True,
        )
    sys.exit(child.returncode)


if __name__ == "__main__":
    main()
