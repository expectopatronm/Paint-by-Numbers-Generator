from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pbn import mixing
from pbn.color import deltaE_lab, rgb_to_hsv
from pbn.config import BASE_PALETTE, DARKEN_PER_PIGMENT, DEFAULT_CONFIG
from pbn.mixing import recipe_text, stochastic_integer_mix_best


def recipe_to_parts(entries, names):
    parts = np.zeros(len(names), dtype=int)
    index = {name: i for i, name in enumerate(names)}
    for name, amount in entries:
        parts[index[name]] += int(amount)
    return parts


def parse_recipe(text: str):
    entries = []
    if not text:
        return entries
    for chunk in text.split(","):
        name, amount = chunk.rsplit(":", 1)
        entries.append((name.strip(), int(amount.strip())))
    return entries


def entries_from_parts(parts, names):
    return [(names[i], int(p)) for i, p in enumerate(parts) if p > 0]


def random_recipe(rng, n_pigments, max_parts, max_components):
    k = int(rng.integers(1, max_components + 1))
    chosen = rng.choice(n_pigments, size=k, replace=False)
    total = int(rng.integers(k, max_parts + 1))
    parts = np.ones(k, dtype=int)
    for _ in range(total - k):
        parts[int(rng.integers(0, k))] += 1
    out = np.zeros(n_pigments, dtype=int)
    out[chosen] = parts
    return out


def normalize_constraints(parts, rng, max_parts, max_components):
    parts = np.maximum(np.array(parts, dtype=int), 0)

    while int(parts.sum()) > max_parts:
        nonzero = np.flatnonzero(parts > 0)
        parts[int(rng.choice(nonzero))] -= 1

    while int(np.count_nonzero(parts)) > max_components:
        nonzero = np.flatnonzero(parts > 0)
        smallest = min(nonzero, key=lambda i: parts[i])
        parts[int(smallest)] = 0

    if int(parts.sum()) <= 0:
        parts[int(rng.integers(0, len(parts)))] = 1

    return parts


def make_evaluator(target, names):
    base = np.array([BASE_PALETTE[n] for n in names], dtype=float)
    cache = {}

    def evaluate(parts):
        key = tuple(int(x) for x in parts)
        if key not in cache:
            rgb = mixing.mix_learned(np.array(parts, dtype=float), base, names)
            cache[key] = (deltaE_lab(rgb, target), rgb)
        return cache[key]

    return evaluate


def mutate_move(parts, rng, max_parts, max_components):
    candidate = parts.copy()
    move = rng.random()
    n = len(candidate)

    if move < 0.55 and candidate.sum() > 0:
        srcs = np.flatnonzero(candidate > 0)
        src = int(rng.choice(srcs))
        dst = int(rng.integers(0, n))
        if src != dst:
            candidate[src] -= 1
            candidate[dst] += 1
    elif move < 0.75 and candidate.sum() < max_parts:
        active = np.flatnonzero(candidate > 0)
        inactive = np.flatnonzero(candidate == 0)
        if len(active) < max_components and len(inactive) > 0 and rng.random() < 0.35:
            dst = int(rng.choice(inactive))
        else:
            dst = int(rng.integers(0, n))
        candidate[dst] += 1
    elif candidate.sum() > 1:
        srcs = np.flatnonzero(candidate > 0)
        src = int(rng.choice(srcs))
        candidate[src] -= 1
    else:
        candidate[int(rng.integers(0, n))] += 1

    return normalize_constraints(candidate, rng, max_parts, max_components)


def local_refine(parts, evaluate, rng, max_parts, max_components):
    best = parts.copy()
    best_err, best_rgb = evaluate(best)
    improved = True
    while improved:
        improved = False
        active_count = int(np.count_nonzero(best))
        for src in range(len(best)):
            if best[src] <= 0:
                continue
            for dst in range(len(best)):
                if src == dst:
                    continue
                if best[dst] == 0 and active_count >= max_components:
                    continue
                candidate = best.copy()
                candidate[src] -= 1
                candidate[dst] += 1
                candidate = normalize_constraints(candidate, rng, max_parts, max_components)
                err, rgb = evaluate(candidate)
                if err + 1e-9 < best_err:
                    best, best_err, best_rgb = candidate, err, rgb
                    improved = True
                    active_count = int(np.count_nonzero(best))
    return best, best_err, best_rgb


def simulated_annealing(target, names, seed_parts, max_parts, max_components, restarts, steps, seed):
    rng = np.random.default_rng(seed)
    evaluate = make_evaluator(target, names)
    best_parts = seed_parts.copy()
    best_err, best_rgb = evaluate(best_parts)

    starts = [seed_parts.copy()]
    starts.extend(random_recipe(rng, len(names), max_parts, max_components) for _ in range(max(0, restarts - 1)))

    for start in starts:
        current = normalize_constraints(start, rng, max_parts, max_components)
        current_err, _current_rgb = evaluate(current)
        temp0 = max(0.5, current_err * 0.75)
        temp1 = 0.02

        for step in range(max(1, steps)):
            t = step / max(1, steps - 1)
            temp = temp0 * ((temp1 / temp0) ** t)
            candidate = mutate_move(current, rng, max_parts, max_components)
            candidate_err, candidate_rgb = evaluate(candidate)
            delta = candidate_err - current_err
            if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
                current, current_err = candidate, candidate_err
                if candidate_err < best_err:
                    best_parts, best_err, best_rgb = candidate.copy(), candidate_err, candidate_rgb

    return local_refine(best_parts, evaluate, rng, max_parts, max_components)


def genetic_algorithm(target, names, seed_parts, max_parts, max_components, population, generations, seed):
    rng = np.random.default_rng(seed)
    evaluate = make_evaluator(target, names)
    pop = [normalize_constraints(seed_parts, rng, max_parts, max_components)]
    pop.extend(random_recipe(rng, len(names), max_parts, max_components) for _ in range(population - 1))

    best_parts = pop[0].copy()
    best_err, best_rgb = evaluate(best_parts)

    for _ in range(generations):
        scored = []
        for parts in pop:
            err, rgb = evaluate(parts)
            scored.append((err, parts, rgb))
            if err < best_err:
                best_parts, best_err, best_rgb = parts.copy(), err, rgb
        scored.sort(key=lambda x: x[0])
        elites = [parts.copy() for _err, parts, _rgb in scored[: max(2, population // 8)]]

        next_pop = elites[:]
        weights = np.array([1.0 / (0.05 + err) for err, _parts, _rgb in scored], dtype=float)
        weights = weights / weights.sum()

        while len(next_pop) < population:
            p1 = scored[int(rng.choice(len(scored), p=weights))][1]
            p2 = scored[int(rng.choice(len(scored), p=weights))][1]
            mask = rng.random(len(names)) < 0.5
            child = np.where(mask, p1, p2)
            if rng.random() < 0.85:
                child = mutate_move(child, rng, max_parts, max_components)
            child = normalize_constraints(child, rng, max_parts, max_components)
            next_pop.append(child)
        pop = next_pop

    return local_refine(best_parts, evaluate, rng, max_parts, max_components)


def print_result(label, parts, err, rgb, names, seconds):
    hsv = rgb_to_hsv(rgb)
    print(f"\n{label}")
    print("-" * len(label))
    print(f"Seconds: {seconds:.2f}")
    print(f"Recipe: {recipe_text(entries_from_parts(parts, names))}")
    print(f"Predicted RGB: {tuple(np.rint(rgb).astype(int))}")
    print(f"HSV: h={hsv[0]:.1f}, s={hsv[1]:.3f}, v={hsv[2]:.3f}")
    print(f"Delta E: {err:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Compare integer optimizers for one color target.")
    parser.add_argument("--target-rgb", nargs=3, type=int, default=(165, 161, 156))
    parser.add_argument(
        "--seed-recipe",
        default="burnt_sienna:1,cobalt_blue:2,titanium_white:5,yellow_ochre:1",
    )
    parser.add_argument("--max-parts", type=int, default=20)
    parser.add_argument("--components", type=int, default=10)
    parser.add_argument("--anneal-restarts", type=int, default=12)
    parser.add_argument("--anneal-steps", type=int, default=6000)
    parser.add_argument("--ga-population", type=int, default=140)
    parser.add_argument("--ga-generations", type=int, default=140)
    parser.add_argument("--stochastic-samples", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    names = list(DEFAULT_CONFIG["palette"])
    target = np.array(args.target_rgb, dtype=float)
    seed_entries = parse_recipe(args.seed_recipe)
    seed_parts = recipe_to_parts(seed_entries, names)

    mixing.BASE_PALETTE = BASE_PALETTE
    mixing.DARKEN_PER_PIGMENT = DARKEN_PER_PIGMENT
    evaluate = make_evaluator(target, names)

    seed_err, seed_rgb = evaluate(seed_parts)
    print_result("Seed recipe", seed_parts, seed_err, seed_rgb, names, 0.0)

    t0 = time.perf_counter()
    st_entries, st_rgb, st_err = stochastic_integer_mix_best(
        target,
        names,
        max_parts=args.max_parts,
        max_components=args.components,
        samples=args.stochastic_samples,
        seed_entries=seed_entries,
        seed=args.seed,
    )
    print_result(
        "Current stochastic retry",
        recipe_to_parts(st_entries, names),
        st_err,
        st_rgb,
        names,
        time.perf_counter() - t0,
    )

    t0 = time.perf_counter()
    sa_parts, sa_err, sa_rgb = simulated_annealing(
        target,
        names,
        seed_parts,
        args.max_parts,
        args.components,
        args.anneal_restarts,
        args.anneal_steps,
        args.seed + 1,
    )
    print_result("Simulated annealing", sa_parts, sa_err, sa_rgb, names, time.perf_counter() - t0)

    t0 = time.perf_counter()
    ga_parts, ga_err, ga_rgb = genetic_algorithm(
        target,
        names,
        seed_parts,
        args.max_parts,
        args.components,
        args.ga_population,
        args.ga_generations,
        args.seed + 2,
    )
    print_result("Genetic algorithm", ga_parts, ga_err, ga_rgb, names, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
