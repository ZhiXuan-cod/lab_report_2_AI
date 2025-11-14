import numpy as np
import streamlit as st

st.title("Genetic Algorithm — Bitstring Peak at 50 Ones")

# =========================
# Parameters
# =========================
POP_SIZE = 300
CHROM_LEN = 80
GENERATIONS = 50
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.01
TOURNAMENT_K = 3
rng = np.random.default_rng(42)

# =========================
# Fitness: Peak when ones = 50 → fitness = 80
# =========================
def fitness(ind):
    ones = np.sum(ind)
    return 80 - abs(ones - 50)

# =========================
# Initialize population
# =========================
def init_population():
    return rng.integers(0, 2, size=(POP_SIZE, CHROM_LEN), dtype=np.int8)

# =========================
# Tournament selection
# =========================
def tournament_select(pop, fits):
    idx = rng.integers(0, POP_SIZE, size=TOURNAMENT_K)
    best = idx[np.argmax(fits[idx])]
    return pop[best]

# =========================
# One-point crossover
# =========================
def crossover(a, b):
    if rng.random() > CROSSOVER_RATE:
        return a.copy(), b.copy()
    point = rng.integers(1, CHROM_LEN)
    c1 = np.concatenate([a[:point], b[point:]])
    c2 = np.concatenate([b[:point], a[point:]])
    return c1, c2

# =========================
# Bit mutation
# =========================
def mutate(ind):
    mask = rng.random(CHROM_LEN) < MUTATION_RATE
    child = ind.copy()
    child[mask] = 1 - child[mask]
    return child

# =========================
# GA main loop
# =========================
if st.button("Run Genetic Algorithm"):
    pop = init_population()

    progress_area = st.empty()
    log_area = st.empty()

    for gen in range(GENERATIONS):
        fits = np.array([fitness(ind) for ind in pop])

        new_pop = []
        while len(new_pop) < POP_SIZE:
            p1 = tournament_select(pop, fits)
            p2 = tournament_select(pop, fits)

            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)

            new_pop.append(c1)
            if len(new_pop) < POP_SIZE:
                new_pop.append(c2)

        pop = np.array(new_pop)

        # Update Streamlit UI
        best_fit = np.max(fits)
        avg_fit = np.mean(fits)
        worst_fit = np.min(fits)

        progress_area.write(
            f"Generation {gen+1}/{GENERATIONS} — Best={best_fit:.2f}, "
            f"Avg={avg_fit:.2f}, Worst={worst_fit:.2f}"
        )

    # Final result
    fits = np.array([fitness(ind) for ind in pop])
    best = pop[np.argmax(fits)]
    ones = np.sum(best)

    st.subheader("Best Individual Found")
    st.write(f"Fitness: **{fitness(best)}**")
    st.write(f"Number of ones: **{ones} / 80**")
    st.code("".join(map(str, best)))
