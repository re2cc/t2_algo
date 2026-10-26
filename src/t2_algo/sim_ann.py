import math
import random
import time
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


def linear_cooling(step, initial_temperature, iterations):
    return initial_temperature * (1 - step / iterations)


def exponential_cooling(step, initial_temperature, cooling_rate):
    return initial_temperature * cooling_rate**step


def logarithmic_cooling(step, initial_temperature):
    return initial_temperature / (1 + math.log(1 + step))


def random_cooling(temperature, cooling_rate, rng):
    # A random reduction, not a random reset of the temperature.
    factor = rng.uniform(cooling_rate, 1.0)
    return temperature * factor


def simulated_annealing(
    route,
    cities,
    schedule="exponential",
    initial_temperature=100.0,
    iterations=10000,
    cooling_rate=0.995,
    seed=None,
    history=None,
):
    """Return the best tour found and its length, not necessarily a local optimum.

    Schedules: linear, exponential, logarithmic, or random.
    cooling_rate is used only by exponential and random cooling.
    All schedules start at initial_temperature and use the iteration limit.
    If history is a list, append the initial state and each iteration's results.
    """
    if schedule not in ("linear", "exponential", "logarithmic", "random"):
        raise ValueError("Unknown cooling schedule")
    if initial_temperature <= 0:
        raise ValueError("initial_temperature must be positive")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if schedule in ("exponential", "random") and not 0 < cooling_rate < 1:
        raise ValueError("cooling_rate must be between 0 and 1")

    rng = random.Random(seed)
    current_route = route.copy()
    current_length = tour_length(current_route, cities)
    best_route = current_route.copy()
    best_length = current_length
    temperature = initial_temperature

    if history is not None:
        history.append(
            {
                "iteration": 0,
                "temperature": temperature,
                "current_length": current_length,
                "best_length": best_length,
            }
        )

    # Fewer than four cities have no nontrivial 2-opt moves.
    if len(current_route) < 4:
        return best_route, best_length

    for step in range(iterations):
        if schedule == "linear":
            temperature = linear_cooling(step, initial_temperature, iterations)
        elif schedule == "exponential":
            temperature = exponential_cooling(step, initial_temperature, cooling_rate)
        elif schedule == "logarithmic":
            temperature = logarithmic_cooling(step, initial_temperature)
        elif step > 0:
            temperature = random_cooling(temperature, cooling_rate, rng)

        if temperature <= 0:
            break

        # Choose a random valid 2-opt move, keeping the starting city fixed.
        while True:
            i, k = rng.sample(range(1, len(current_route)), 2)
            if i > k:
                temp = i
                i = k
                k = temp

            if i == 1 and k == len(current_route) - 1:
                continue
            break

        candidate = two_opt_move(current_route, i, k)
        candidate_length = tour_length(candidate, cities)
        delta = candidate_length - current_length

        if delta < 0:
            current_route = candidate
            current_length = candidate_length
        else:
            probability = math.exp(-delta / temperature)
            if rng.random() < probability:
                current_route = candidate
                current_length = candidate_length

        if current_length < best_length:
            best_route = current_route.copy()
            best_length = current_length

        if history is not None:
            history.append(
                {
                    "iteration": step + 1,
                    "temperature": temperature,
                    "current_length": current_length,
                    "best_length": best_length,
                }
            )

    return best_route, best_length


if __name__ == "__main__":
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    # Same city instance as two_opt.py.
    city_count = 20
    city_rng = random.Random(7)
    cities = {}
    for i in range(city_count):
        name = f"C{i + 1}"
        x = city_rng.uniform(0, 100)
        y = city_rng.uniform(0, 100)
        cities[name] = (x, y)

    trials = 30
    iterations = 15000
    initial_temperature = 100.0
    cooling_rate = 0.995
    schedules = ["linear", "exponential", "logarithmic", "random"]

    print(f"Cities: {len(cities)} | Trials per schedule: {trials}")
    print(
        f"Iterations: {iterations} | Initial temperature: {initial_temperature} "
        f"| Cooling rate: {cooling_rate}"
    )
    print("Quality is the best tour length found; lower is better.")
    print()
    print(f"{'Schedule':<14} {'Mean time (s)':>16} {'Mean tour length':>18}")
    print("-" * 50)

    for schedule in schedules:
        total_time = 0
        total_length = 0
        total_costs = [0.0] * (iterations + 1)
        total_temperatures = [0.0] * (iterations + 1)

        for trial in range(trials):
            # Recreate the same initial tour for every schedule in this trial.
            route_rng = random.Random(42 + trial)
            initial_route = list(cities)
            route_rng.shuffle(initial_route)
            history = []

            start_time = time.perf_counter()
            best_route, best_length = simulated_annealing(
                initial_route,
                cities,
                schedule=schedule,
                initial_temperature=initial_temperature,
                iterations=iterations,
                cooling_rate=cooling_rate,
                seed=42 + trial,
                history=history,
            )
            elapsed_time = time.perf_counter() - start_time

            total_time += elapsed_time
            total_length += best_length

            for i in range(len(history)):
                total_costs[i] += history[i]["best_length"]
                total_temperatures[i] += history[i]["temperature"]

        mean_costs = []
        mean_temperatures = []
        for i in range(iterations + 1):
            mean_costs.append(total_costs[i] / trials)
            mean_temperatures.append(total_temperatures[i] / trials)

        mean_time = total_time / trials
        mean_length = total_length / trials
        print(f"{schedule:<14} {mean_time:>16.4f} {mean_length:>18.2f}")

        plt.figure(1)
        plt.plot(range(iterations + 1), mean_costs, label=schedule.capitalize())

        plt.figure(2)
        plt.plot(mean_temperatures, mean_costs, label=schedule.capitalize())

        plt.figure(3)
        plt.scatter(mean_time, mean_length, label=schedule.capitalize())

    plt.figure(1)
    plt.xlabel("Iterations")
    plt.ylabel("Mean best tour length")
    plt.title("Convergence by iteration")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "sim_ann_iterations.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(2)
    plt.xlabel("Mean temperature (cooling from left to right)")
    plt.ylabel("Mean best tour length")
    plt.title("Convergence by temperature")
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "sim_ann_temperature.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(3)
    plt.xlabel("Mean execution time (s)")
    plt.ylabel("Mean best tour length (lower is better)")
    plt.title("Execution time vs solution quality")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "sim_ann_time_quality.png", dpi=300, bbox_inches="tight")
    plt.close()
