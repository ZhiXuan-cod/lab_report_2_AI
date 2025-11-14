import numpy as np

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
pop = init_population()

for gen in range(GENERATIONS):
    fits = np.array([fitness(ind) for ind in pop])

    new_pop = []

    while len(new_pop) < POP_SIZE:
        # Select parents
        p1 = tournament_select(pop, fits)
        p2 = tournament_select(pop, fits)

        # Recombine
        c1, c2 = crossover(p1, p2)

        # Mutate
        c1 = mutate(c1)
        c2 = mutate(c2)

        new_pop.append(c1)
        if len(new_pop) < POP_SIZE:
            new_pop.append(c2)

    pop = np.array(new_pop)

    # Logging
    best_fit = np.max(fits)
    avg_fit = np.mean(fits)
    worst_fit = np.min(fits)
    print(f"Gen {gen+1:02d}: Best={best_fit:.2f} Avg={avg_fit:.2f} Worst={worst_fit:.2f}")

# =========================
# Final result
# =========================
fits = np.array([fitness(ind) for ind in pop])
best = pop[np.argmax(fits)]
ones = np.sum(best)

print("\nBest individual found:")
print("Bitstring:", "".join(map(str, best)))
print("Ones:", ones)
print("Fitness:", fitness(best))
