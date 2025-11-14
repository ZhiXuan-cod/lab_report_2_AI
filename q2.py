import math
import random
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# -------------------- Problem Definition for Lab Report 2 --------------------
@dataclass
class GAProblem:
    name: str
    dim: int
    fitness_fn: Callable[[np.ndarray], float]


def make_lab2_problem() -> GAProblem:
    """
    Required by Question 2:
    - Chromosome length = 80 bits
    - Fitness max = 80 when ones = 50
    - Penalize by deviation * 1.6
    """
    def fitness(x: np.ndarray) -> float:
        ones = int(np.sum(x))
        if ones == 50:
            return 80.0
        return max(0.0, 80.0 - abs(ones - 50) * 1.6)

    return GAProblem(
        name="Lab 2: 80-bit chromosome targeting exactly 50 ones",
        dim=80,
        fitness_fn=fitness,
    )


# -------------------- GA Operators --------------------
def init_population(dim: int, pop_size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=(pop_size, dim), dtype=np.int8)


def tournament_selection(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
    idxs = rng.integers(0, fitness.size, size=k)
    return int(idxs[np.argmax(fitness[idxs])])


def one_point_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator):
    point = int(rng.integers(1, a.size))
    c1 = np.concatenate([a[:point], b[point:]])
    c2 = np.concatenate([b[:point], a[point:]])
    return c1, c2


def bit_mutation(x: np.ndarray, mutation_rate: float, rng: np.random.Generator):
    mask = rng.random(x.shape) < mutation_rate
    y = x.copy()
    y[mask] = 1 - y[mask]
    return y


def evaluate(pop: np.ndarray, fitness_fn):
    return np.array([fitness_fn(ind) for ind in pop], dtype=float)


# -------------------- GA Execution (for Lab Report Question 2) --------------------
def run_ga_lab2(stream_live=True):
    # Required parameters (fixed by assignment)
    pop_size = 300
    dim = 80
    generations = 50
    mutation_rate = 0.01
    crossover_rate = 0.9
    tournament_k = 3

    rng = np.random.default_rng(42)
    problem = make_lab2_problem()

    pop = init_population(dim, pop_size, rng)
    fit = evaluate(pop, problem.fitness_fn)

    # Logging containers
    chart_area = st.empty()
    info_area = st.empty()

    history_best = []
    history_avg = []

    for gen in range(generations):
        best_idx = int(np.argmax(fit))
        best_fit = float(fit[best_idx])
        avg_fit = float(np.mean(fit))

        history_best.append(best_fit)
        history_avg.append(avg_fit)

        # Stream updates
        if stream_live:
            df = pd.DataFrame({"Best": history_best, "Average": history_avg})
            chart_area.line_chart(df)
            info_area.write(f"Generation {gen+1}: Best fitness = {best_fit:.2f}")

        # STOP EARLY if perfect chromosome found
        if best_fit == 80.0:
            break

        next_pop = []
        while len(next_pop) < pop_size:
            # Selection
            i1 = tournament_selection(fit, tournament_k, rng)
            i2 = tournament_selection(fit, tournament_k, rng)
            p1, p2 = pop[i1], pop[i2]

            # Crossover
            if rng.random() < crossover_rate:
                c1, c2 = one_point_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutation
            c1 = bit_mutation(c1, mutation_rate, rng)
            c2 = bit_mutation(c2, mutation_rate, rng)

            next_pop.append(c1)
            if len(next_pop) < pop_size:
                next_pop.append(c2)

        pop = np.array(next_pop)
        fit = evaluate(pop, problem.fitness_fn)

    # Final best solution
    best_idx = int(np.argmax(fit))
    best = pop[best_idx]
    best_fit = float(fit[best_idx])

    return {
        "best": best,
        "best_fitness": best_fit,
        "history_best": history_best,
        "history_avg": history_avg,
        "population": pop,
    }


# -------------------- Streamlit UI (Simple) --------------------
st.title("Genetic Algorithm – Lab Report 2 (BSD3513)")
st.write("Target: Find an 80-bit chromosome containing **exactly 50 ones**.")

if st.button("Run Genetic Algorithm"):
    result = run_ga_lab2(stream_live=True)

    st.subheader("Best Fitness Achieved")
    st.write(result["best_fitness"])

    best_bits = ''.join(map(str, result["best"].tolist()))
    st.code(best_bits)

    st.write(f"Number of ones: {np.sum(result['best'])} / 50")
