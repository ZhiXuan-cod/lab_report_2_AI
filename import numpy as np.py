import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import dataclass
from typing import Callable, Tuple

# -------------------- Problem Definition --------------------
@dataclass
class GAProblem:
    name: str
    chromosome_type: str
    dim: int
    bounds: Tuple[float, float] | None
    fitness_fn: Callable[[np.ndarray], float]


# Fitness: peak at exactly 50 ones → 80 points
def make_custom_peak(dim: int = 80, peak: int = 50, max_value: float = 80.0) -> GAProblem:
    def fitness(x: np.ndarray) -> float:
        ones = int(np.sum(x))
        return max_value - abs(ones - peak)

    return GAProblem(
        name=f"Bitstring Peak – max at {peak} ones",
        chromosome_type="bit",
        dim=dim,
        bounds=None,
        fitness_fn=fitness,
    )


# -------------------- GA Operators --------------------
def init_population(problem: GAProblem, pop_size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=(pop_size, problem.dim), dtype=np.int8)


def tournament_selection(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
    idxs = rng.integers(0, fitness.size, size=k)
    return int(idxs[np.argmax(fitness[idxs])])


def one_point_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator):
    point = int(rng.integers(1, a.size))
    return (
        np.concatenate([a[:point], b[point:]]),
        np.concatenate([b[:point], a[point:]]),
    )


def bit_mutation(x: np.ndarray, mut_rate: float, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(x.shape) < mut_rate
    y = x.copy()
    y[mask] = 1 - y[mask]
    return y


def evaluate(pop: np.ndarray, problem: GAProblem) -> np.ndarray:
    return np.array([problem.fitness_fn(ind) for ind in pop], dtype=float)


# -------------------- GA Execution --------------------
def run_ga(
    problem: GAProblem,
    pop_size: int,
    generations: int,
    crossover_rate: float,
    mutation_rate: float,
    tournament_k: int,
    elitism: int,
    seed: int,
    stream_live: bool = True,
):

    rng = np.random.default_rng(seed)

    pop = init_population(problem, pop_size, rng)
    fit = evaluate(pop, problem)

    # Live plotting containers
    chart_area = st.empty()
    stats_area = st.empty()

    history = {"Best": [], "Average": [], "Worst": []}

    for gen in range(generations):
        # Track stats
        best = float(np.max(fit))
        avg = float(np.mean(fit))
        worst = float(np.min(fit))

        history["Best"].append(best)
        history["Average"].append(avg)
        history["Worst"].append(worst)

        # Live update
        if stream_live:
            df_hist = pd.DataFrame(history)
            chart_area.line_chart(df_hist)
            stats_area.write(f"Generation {gen+1}/{generations} — Best fitness: **{best:.2f}**")

        # Elitism
        E = min(elitism, pop_size)
        elite_idx = np.argpartition(fit, -E)[-E:]
        elites = pop[elite_idx].copy()

        # New population
        next_pop = []

        while len(next_pop) < pop_size - E:
            p1 = pop[tournament_selection(fit, tournament_k, rng)]
            p2 = pop[tournament_selection(fit, tournament_k, rng)]

            # Crossover
            if rng.random() < crossover_rate:
                c1, c2 = one_point_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutation
            c1 = bit_mutation(c1, mutation_rate, rng)
            c2 = bit_mutation(c2, mutation_rate, rng)

            next_pop.append(c1)
            if len(next_pop) < pop_size - E:
                next_pop.append(c2)

        # Replace
        pop = np.vstack((np.array(next_pop), elites))
        fit = evaluate(pop, problem)

    # Final result
    best_idx = int(np.argmax(fit))

    return {
        "best": pop[best_idx],
        "best_fitness": float(fit[best_idx]),
        "history": pd.DataFrame(history),
        "population": pop,
        "fitness": fit,
    }


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Genetic Algorithm – Peak at 50 Ones", layout="wide")
st.title("Genetic Algorithm (80-bit chromosome, peak fitness at 50 ones)")

with st.sidebar:
    st.header("Problem Settings")
    problem = make_custom_peak()

    st.write("**Fixed Assignment Requirements**")
    st.write("- Population = 300\n- Chromosome length = 80 bits\n- Peak fitness = 80 at 50 ones\n- Generations = 50")

    pop_size = 300
    generations = 50

    st.header("GA Controls")
    crossover_rate = st.slider("Crossover rate", 0.0, 1.0, 0.8)
    mutation_rate = st.slider("Mutation rate", 0.0, 1.0, 0.01)
    tournament_k = st.slider("Tournament size", 2, 10, 3)
    elitism = st.slider("Elites per generation", 0, 20, 2)
    seed = st.number_input("Random seed", value=42)
    live = st.checkbox("Live updates", value=True)

# Layout
left, right = st.columns([1, 1])

with left:
    if st.button("Run GA", type="primary"):
        result = run_ga(
            problem=problem,
            pop_size=pop_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            tournament_k=tournament_k,
            elitism=elitism,
            seed=int(seed),
            stream_live=live,
        )

        st.subheader("Fitness Curve")
        st.line_chart(result["history"])

        st.subheader("Best Individual")
        st.write(f"Best fitness: **{result['best_fitness']:.2f}**")
        ones = int(np.sum(result["best"]))
        st.write(f"Number of ones: **{ones} / 80**")

        st.code("".join(map(str, result["best"].tolist())))

        st.session_state["population"] = result["population"]
        st.session_state["fitness"] = result["fitness"]

with right:
    st.subheader("Final Population (First 20)")
    if st.button("Show population"):
        pop = st.session_state.get("population")
        fit = st.session_state.get("fitness")

        if pop is None:
            st.info("Run the GA first.")
        else:
            df = pd.DataFrame(pop[:20])
            df["fitness"] = fit[:20]
            st.dataframe(df, use_container_width=True)

# Init
if "population" not in st.session_state:
    st.session_state["population"] = None
    st.session_state["fitness"] = None
