import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import dataclass
from typing import Callable, Tuple, List

# -------------------- Problem Definition --------------------
@dataclass
class GAProblem:
    name: str
    chromosome_type: str  # "bit"
    dim: int
    bounds: Tuple[float, float] | None
    fitness_fn: Callable[[np.ndarray], float]


# Custom fitness: peak at 50 ones → max value 80
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
    if a.size <= 1:
        return a.copy(), b.copy()
    point = int(rng.integers(1, a.size))
    c1 = np.concatenate([a[:point], b[point:]])
    c2 = np.concatenate([b[:point], a[point:]])
    return c1, c2


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

    # Live UI containers
    chart_area = st.empty()
    best_area = st.empty()

    history_best = []
    history_avg = []
    history_worst = []

    for gen in range(generations):
        # Statistics
        best_idx = int(np.argmax(fit))
        best_fit = float(fit[best_idx])
        avg_fit = float(np.mean(fit))
        worst_fit = float(np.min(fit))

        history_best.append(best_fit)
        history_avg.append(avg_fit)
        history_worst.append(worst_fit)

        # Live update
        if stream_live:
            df = pd.DataFrame({"Best": history_best, "Average": history_avg, "Worst": history_worst})
            chart_area.line_chart(df)
            best_area.markdown(
                f"Generation {gen+1}/{generations} — Best fitness: **{best_fit:.2f}**"
            )

        # Elitism
        E = max(0, min(elitism, pop_size))
        elite_idx = np.argpartition(fit, -E)[-E:] if E > 0 else np.array([], dtype=int)
        elites = pop[elite_idx].copy()

        # Create next population
        next_pop = []

        while len(next_pop) < pop_size - E:
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
            if len(next_pop) < pop_size - E:
                next_pop.append(c2)

        # Replace population
        pop = np.vstack([np.array(next_pop), elites])
        fit = evaluate(pop, problem)

    # Final results
    best_idx = int(np.argmax(fit))
    df_hist = pd.DataFrame({"Best": history_best, "Average": history_avg, "Worst": history_worst})

    return {
        "best": pop[best_idx],
        "best_fitness": float(fit[best_idx]),
        "history": df_hist,
        "final_population": pop,
        "final_fitness": fit,
    }


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Genetic Algorithm — 50 Ones Peak", layout="wide")
st.title("Genetic Algorithm (Bitstring Peak at 50 Ones)")
st.caption("GA evolves an 80-bit chromosome whose fitness peaks when it contains exactly 50 ones.")


# Sidebar — Fixed for assignment
with st.sidebar:
    st.header("GA Problem")
    st.write("Chromosome length fixed: **80 bits**\nFitness max at **50 ones → 80 points**")

    problem = make_custom_peak()

    st.header("GA Parameters (Fixed for Assignment)")
    pop_size = 300
    generations = 50
    crossover_rate = st.slider("Crossover rate", 0.0, 1.0, 0.8)
    mutation_rate = st.slider("Mutation rate", 0.0, 1.0, 0.01)
    tournament_k = st.slider("Tournament size", 2, 10, 3)
    elitism = st.slider("Elites per generation", 0, 20, 2)
    seed = st.number_input("Random seed", value=42)
    live = st.checkbox("Live chart updates", value=True)


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
            stream_live=bool(live),
        )

        st.subheader("Fitness Curve")
        st.line_chart(result["history"])

        st.subheader("Best Individual")
        st.write(f"Best fitness: **{result['best_fitness']:.2f}**")
        ones = int(np.sum(result["best"]))
        st.write(f"Number of ones: **{ones} / 80**")

        bitstring = "".join(map(str, result["best"].astype(int).tolist()))
        st.code(bitstring)

        # Save to session
        st.session_state["_final_pop"] = result["final_population"]
        st.session_state["_final_fit"] = result["final_fitness"]


with right:
    st.subheader("Final Population Snapshot")
    if st.button("Show first 20 individuals"):
        pop = st.session_state.get("_final_pop")
        fit = st.session_state.get("_final_fit")

        if pop is None:
            st.info("Run GA first.")
        else:
            df = pd.DataFrame(pop[:20])
            df["fitness"] = fit[:20]
            st.dataframe(df, use_container_width=True)


# Initialize session state on first load
if "_final_pop" not in st.session_state:
    st.session_state["_final_pop"] = None
    st.session_state["_final_fit"] = None
