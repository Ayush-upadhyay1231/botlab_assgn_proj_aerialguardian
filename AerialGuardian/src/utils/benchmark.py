# src/utils/benchmark.py
"""FPS and memory benchmarking script."""

import time, torch, numpy as np
from collections import defaultdict

class Benchmarker:
    def __init__(self):
        self.timings = defaultdict(list)

    def tick(self, stage: str):
        self.timings[stage].append(time.perf_counter())

    def report(self):
        print("\n── Benchmark Report ──────────────────────")
        stages = list(self.timings.keys())
        for i in range(0, len(stages)-1, 2):
            name = stages[i]
            starts = self.timings[stages[i]]
            ends   = self.timings[stages[i+1]]
            dts = [e-s for s,e in zip(starts, ends)]
            if dts:
                mean_ms = 1000 * np.mean(dts)
                print(f"  {name:<25} {mean_ms:6.1f} ms/frame")
        if torch.cuda.is_available():
            vram = torch.cuda.max_memory_allocated() / 1e9
            print(f"\n  Peak VRAM used: {vram:.2f} GB")
        print("──────────────────────────────────────────")