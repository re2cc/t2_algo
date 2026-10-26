import multiprocessing as mp
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt


def estimate_pi(seed: int, samples: int = 1_000_000) -> float:
    """Monte Carlo: fraction of random points that fall inside the unit quarter circle."""
    rng = random.Random(seed)
    inside = sum(rng.random() ** 2 + rng.random() ** 2 <= 1.0 for _ in range(samples))
    return 4 * inside / samples


if __name__ == "__main__":
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    seeds = list(range(8))  # 8 independent runs, one seed each
    n = [1, 2, 4, 8]

    # Sequential Looping
    t0 = time.perf_counter()
    seq_results = [estimate_pi(s) for s in seeds]
    t_seq = time.perf_counter() - t0

    # Multiprocessing
    t_pro: list[float] = []
    for i in n:
        t0 = time.perf_counter()
        with mp.Pool(processes=i) as pool:
            par_results = pool.map(estimate_pi, seeds)
        t_pro.append(time.perf_counter() - t0)

    # Threading (For this to work fast you need a GIL disabled python build)
    t_thr: list[float] = []
    for i in n:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=i) as pool:
            par_results = list(pool.map(estimate_pi, seeds))
        t_thr.append(time.perf_counter() - t0)

    t_pro_spd = [t_seq / t for t in t_pro]
    t_thr_spd = [t_seq / t for t in t_thr]

    print(f"Sequential wall-clock: {t_seq:.3f} s")
    for i in range(len(n)):
        print(
            f"Parallel wall-clock (process={n[i]}):   {t_pro[i]:.3f} s (Speedup: {t_pro_spd[i]:.2f}x)"
        )
    for i in range(len(n)):
        print(
            f"Parallel wall-clock (threads={n[i]}):   {t_thr[i]:.3f} s (Speedup: {t_thr_spd[i]:.2f}x)"
        )

    print(
        f"pi estimate (mean of {len(seeds)} runs): {sum(par_results) / len(par_results):.5f}"
    )

    plt.plot(n, n, "o--", label="Ideal")
    plt.plot(n, t_pro_spd, "o-", label="Multiprocess")
    plt.plot(n, t_thr_spd, "o-", label="Threading")
    plt.scatter(1, 1, marker="*", color="yellow", s=10, zorder=5, label="Sequential")

    plt.yticks(range(9), [f"{i}x" for i in range(9)])
    plt.xticks(n)
    plt.xlabel("Number of processes/threads")
    plt.ylabel("Speedup")

    plt.grid(True)
    plt.legend()

    plt.savefig(output_dir / "performance.png", dpi=300, bbox_inches="tight")
