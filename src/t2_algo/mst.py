import itertools
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt

from t2_algo.sim_ann import simulated_annealing
from t2_algo.two_opt import two_opt


def mst_prim(points: list[tuple[float, float]]) -> dict[int, list[int]]:
    """Prim's algorithm from city 0; returns the tree as {node: children}."""
    best = {v: (math.dist(points[0], points[v]), 0) for v in range(1, len(points))}
    children = {v: [] for v in range(len(points))}
    while best:
        v = min(best, key=lambda u: best[u][0])
        children[best.pop(v)[1]].append(v)          # add the cheapest edge to the tree
        for u in best:
            d = math.dist(points[v], points[u])
            if d < best[u][0]:
                best[u] = (d, v)
    return children


def preorder(children: dict[int, list[int]], root: int = 0) -> list[int]:
    """Walk around the tree; visiting each city once = shortcutting repeated cities."""
    return [root] + [c for child in children[root] for c in preorder(children, child)]


def tour_length(points, tour) -> float:
    return sum(math.dist(points[tour[i]], points[tour[(i + 1) % len(tour)]]) for i in range(len(tour)))


def brute_force(points):
    optimal = min(((0,) + p for p in itertools.permutations(range(1, len(points)))),
                  key=lambda t: tour_length(points, t))   # brute force: 8! = 40,320 tours for 9 cities
    return optimal


def generate_instances(seed=7):
    rng = random.Random(seed)
    instances = []

    for instance in range(100):
        points = []
        while len(points) < 9:
            x = rng.uniform(0, 10)
            y = rng.uniform(0, 10)
            point = (x, y)
            if point not in points:
                points.append(point)
        instances.append(points)

    return instances


if __name__ == "__main__":
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    instances = generate_instances()
    ratios = []
    mst_two_opt_ratios = []
    two_opt_ratios = []
    annealing_ratios = []

    for instance in range(len(instances)):
        points = instances[instance]
        approx = preorder(mst_prim(points))
        optimal = brute_force(points)
        optimal_length = tour_length(points, optimal)
        ratio = tour_length(points, approx) / optimal_length
        ratios.append(ratio)

        cities = {}
        for i in range(len(points)):
            cities[i] = points[i]

        improved_mst, improvements = two_opt(approx, cities)
        mst_two_opt_ratios.append(tour_length(points, improved_mst) / optimal_length)

        initial_route = list(range(len(points)))
        route_rng = random.Random(42 + instance)
        route_rng.shuffle(initial_route)

        local_route, improvements = two_opt(initial_route, cities)
        two_opt_ratios.append(tour_length(points, local_route) / optimal_length)

        annealing_route, annealing_length = simulated_annealing(
            initial_route,
            cities,
            schedule="logarithmic",
            initial_temperature=10.0,
            iterations=15000,
            seed=42 + instance,
        )
        annealing_ratios.append(annealing_length / optimal_length)

    print(f"Instances: {len(instances)} | Cities per instance: 9")
    print(f"Mean approximation ratio: {sum(ratios) / len(ratios):.3f}")
    print(f"Minimum ratio: {min(ratios):.3f}")
    print(f"Maximum ratio: {max(ratios):.3f}")

    plt.scatter(range(1, len(ratios) + 1), ratios, label="MST approximation")
    plt.axhline(2, color="red", linestyle="--", label="Theoretical upper bound (2)")
    plt.xlabel("Instance")
    plt.ylabel("MST tour length / optimal tour length")
    plt.title("MST approximation ratio")
    plt.ylim(0.95, 2.1)
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "mst_approximation_ratio.png", dpi=300, bbox_inches="tight")
    plt.close()

    methods = ["MST", "MST + 2-opt", "2-opt (random start)", "SA (logarithmic)"]
    comparison = [
        ratios,
        mst_two_opt_ratios,
        two_opt_ratios,
        annealing_ratios,
    ]

    print("\nOne result per method per instance; lower ratios are better.")
    print("SA: logarithmic cooling, 15000 iterations, initial temperature 10.")
    print(f"{'Method':<22} {'Mean ratio':>12} {'Maximum ratio':>14}")
    print("-" * 50)
    for i in range(len(methods)):
        mean_ratio = sum(comparison[i]) / len(comparison[i])
        maximum_ratio = max(comparison[i])
        print(f"{methods[i]:<22} {mean_ratio:>12.3f} {maximum_ratio:>14.3f}")

    labels = ["MST", "MST +\n2-opt", "2-opt\n(random start)", "SA\n(logarithmic)"]
    for i in range(len(methods)):
        optimal_count = 0
        for ratio in comparison[i]:
            if math.isclose(ratio, 1.0, rel_tol=1e-9):
                optimal_count += 1
        labels[i] += f"\n{optimal_count}/{len(comparison[i])} optimal"

    plt.figure()
    plt.boxplot(
        comparison,
        tick_labels=labels,
    )
    plt.axhline(1, color="gray", linestyle="--", label="Optimal ratio (1)")
    plt.xlabel("Method")
    plt.ylabel("Tour length / optimal tour length")
    plt.title("TSP quality")
    plt.ylim(bottom=0.98)
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "mst_method_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
