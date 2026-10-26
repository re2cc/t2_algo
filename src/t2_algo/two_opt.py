import math
import random
from pathlib import Path

import matplotlib.pyplot as plt


def tour_length(route: list[str], cities: dict[str, tuple[float, float]]) -> float:
    return sum(
        math.dist(cities[route[i]], cities[route[(i + 1) % len(route)]])
        for i in range(len(route))
    )


def two_opt_move(route: list[str], i: int, k: int) -> list[str]:
    """Reverse route[i..k]: removes edges (i-1, i) and (k, k+1)."""
    return route[:i] + route[i : k + 1][::-1] + route[k + 1 :]


def random_route(cities, rng):
    route = list(cities)
    rng.shuffle(route)
    return route


def nearest_neighbor_route(cities, rng):
    remaining = list(cities)
    start = rng.choice(remaining)
    route = [start]
    remaining.remove(start)

    while remaining:
        current = route[-1]
        nearest = remaining[0]
        nearest_distance = math.dist(cities[current], cities[nearest])

        for city in remaining:
            distance = math.dist(cities[current], cities[city])
            if distance < nearest_distance:
                nearest = city
                nearest_distance = distance

        route.append(nearest)
        remaining.remove(nearest)

    return route


def two_opt(route, cities):
    current_route = route.copy()
    current_length = tour_length(current_route, cities)
    improvements = 0
    improvement_found = True

    while improvement_found:
        improvement_found = False

        for i in range(1, len(current_route) - 1):
            for k in range(i + 1, len(current_route)):
                if i == 1 and k == len(current_route) - 1:
                    continue

                candidate = two_opt_move(current_route, i, k)
                candidate_length = tour_length(candidate, cities)

                # Ignore rounding errors
                if candidate_length < current_length - 1e-10:
                    current_route = candidate
                    current_length = candidate_length
                    improvements += 1
                    improvement_found = True
                    break

            if improvement_found:
                break

    return current_route, improvements


def run_experiment(cities, trials=30, seed=42):
    rng = random.Random(seed)
    results = []

    for trial in range(1, trials + 1):
        initial_routes = [
            ("Random", random_route(cities, rng)),
            ("Nearest neighbor", nearest_neighbor_route(cities, rng)),
        ]

        for method, initial_route in initial_routes:
            final_route, improvements = two_opt(initial_route, cities)
            results.append(
                {
                    "trial": trial,
                    "method": method,
                    "initial_route": initial_route,
                    "initial_length": tour_length(initial_route, cities),
                    "final_route": final_route,
                    "final_length": tour_length(final_route, cities),
                    "improvements": improvements,
                }
            )

    return results


def save_diagrams(results):
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    methods = ["Random", "Nearest neighbor"]
    improvement_counts = []
    final_lengths = []

    for method in methods:
        method_improvements = []
        method_lengths = []

        for result in results:
            if result["method"] == method:
                method_improvements.append(result["improvements"])
                method_lengths.append(result["final_length"])

        improvement_counts.append(method_improvements)
        final_lengths.append(method_lengths)

    plt.figure()
    plt.boxplot(improvement_counts, tick_labels=methods)
    plt.xlabel("Initial solution")
    plt.ylabel("2-opt improvements")
    plt.title("Improvements needed")
    plt.grid(True)
    plt.savefig(output_dir / "two_opt_improvements.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.boxplot(final_lengths, tick_labels=methods)
    plt.xlabel("Initial solution")
    plt.ylabel("Final tour length (lower is better)")
    plt.title("Quality")
    plt.grid(True)
    plt.savefig(output_dir / "two_opt_quality.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    city_rng = random.Random(7)
    cities = {}
    for i in range(20):
        name = f"C{i + 1}"
        x = city_rng.uniform(0, 100)
        y = city_rng.uniform(0, 100)
        cities[name] = (x, y)

    results = run_experiment(cities, trials=30, seed=42)

    print(f"Cities: {len(cities)} | Trials per method: 30")
    print("Quality is the final tour length; lower is better.")
    print("Improvements counts accepted 2-opt moves.")
    print()
    print(
        f"{'Trial':>5} {'Initial solution':<18} {'Initial length':>14} "
        f"{'Final length':>14} {'Improvements':>12}"
    )
    print("-" * 67)

    for result in results:
        print(
            f"{result['trial']:>5} {result['method']:<18} "
            f"{result['initial_length']:>14.2f} "
            f"{result['final_length']:>14.2f} "
            f"{result['improvements']:>12}"
        )

    print()
    print(
        f"{'Initial solution':<18} {'Mean improvements':>18} {'Mean final length':>18}"
    )
    print("-" * 56)

    for method in ["Random", "Nearest neighbor"]:
        total_improvements = 0
        total_length = 0
        count = 0

        for result in results:
            if result["method"] == method:
                total_improvements += result["improvements"]
                total_length += result["final_length"]
                count += 1

        mean_improvements = total_improvements / count
        mean_length = total_length / count
        print(f"{method:<18} {mean_improvements:>18.2f} {mean_length:>18.2f}")

    save_diagrams(results)
