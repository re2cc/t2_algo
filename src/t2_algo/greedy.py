def knapsack(values: list[int], weights: list[int], W: int):
    n = len(values)
    K = [
        [0] * (W + 1) for _ in range(n + 1)
    ]  # K[i][c]: best value with items 1..i and capacity c
    for i in range(1, n + 1):
        v, w = values[i - 1], weights[i - 1]
        for c in range(W + 1):
            K[i][c] = K[i - 1][c]  # skip item i
            if w <= c:
                K[i][c] = max(K[i][c], v + K[i - 1][c - w])  # take item i
    chosen, c = [], W  # walk back through the table
    for i in range(n, 0, -1):
        if K[i][c] != K[i - 1][c]:
            chosen.append(i - 1)
            c -= weights[i - 1]
    return K, K[n][W], chosen[::-1]


def greedy_knapsack_lightest_first(values: list[int], weights: list[int], W: int):
    order = list(range(len(values)))
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            if weights[order[j]] < weights[order[i]]:
                temp = order[i]
                order[i] = order[j]
                order[j] = temp

    chosen = []
    total_value = 0
    total_weight = 0

    for i in order:
        if total_weight + weights[i] <= W:
            chosen.append(i)
            total_value += values[i]
            total_weight += weights[i]
    return total_value, chosen


def greedy_knapsack_value_weight(values: list[int], weights: list[int], W: int):
    order = list(range(len(values)))
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            current_ratio = values[order[i]] / weights[order[i]]
            other_ratio = values[order[j]] / weights[order[j]]
            if other_ratio > current_ratio:
                temp = order[i]
                order[i] = order[j]
                order[j] = temp

    chosen = []
    total_value = 0
    total_weight = 0

    for i in order:
        if total_weight + weights[i] <= W:
            chosen.append(i)
            total_value += values[i]
            total_weight += weights[i]
    return total_value, chosen


if __name__ == "__main__":
    items = ["A", "B", "C", "D"]
    values = [1, 12, 9, 9]
    weights = [1, 6, 5, 5]
    capacity = 10

    print(f"Knapsack capacity: {capacity}")
    print(f"{'Object':<8} {'Weight':>8} {'Value':>8}")
    for i in range(len(items)):
        print(f"{items[i]:<8} {weights[i]:>8} {values[i]:>8}")
    print()

    K, dp_value, dp_chosen = knapsack(values, weights, capacity)
    ratio_value, ratio_chosen = greedy_knapsack_value_weight(values, weights, capacity)
    lightest_value, lightest_chosen = greedy_knapsack_lightest_first(
        values, weights, capacity
    )

    results = [
        ("Provided code", dp_chosen, dp_value),
        ("Value/weight greedy", ratio_chosen, ratio_value),
        ("Lightest-first greedy", lightest_chosen, lightest_value),
    ]

    print(f"{'Method':<24} {'Objects':<12} {'Capacity used':>14} {'Value':>8}")
    print("-" * 61)

    for method, chosen, value in results:
        selected_objects = []
        used_weight = 0
        for i in chosen:
            selected_objects.append(items[i])
            used_weight += weights[i]

        objects = ", ".join(selected_objects)
        capacity_used = f"{used_weight}/{capacity}"
        print(f"{method:<24} {objects:<12} {capacity_used:>14} {value:>8}")
