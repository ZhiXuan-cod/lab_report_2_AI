import random

# ============================
# Genetic Algorithm Parameters
# ============================
POPULATION_SIZE = 300
CHROMOSOME_LENGTH = 80
TARGET_ONES = 50
GENERATIONS = 50
MUTATION_RATE = 0.01

# ============================
# Fitness Function
# Returns 80 when number of ones = 50
# ============================
def fitness(chromosome):
    ones = sum(chromosome)
    if ones == TARGET_ONES:
        return 80
    else:
        return 80 - abs(TARGET_ONES - ones)

# ============================
# Create Individual
# ============================
def create_individual():
    return [random.randint(0, 1) for _ in range(CHROMOSOME_LENGTH)]

# ============================
# Selection (Tournament)
# ============================
def selection(population):
    c1 = random.choice(population)
    c2 = random.choice(population)
    return c1 if fitness(c1) > fitness(c2) else c2

# ============================
# Crossover (Single Point)
# ============================
def crossover(parent1, parent2):
    point = random.randint(1, CHROMOSOME_LENGTH - 1)
    return parent1[:point] + parent2[point:]

# ============================
# Mutation
# ============================
def mutate(chromosome):
    return [
        bit if random.random() > MUTATION_RATE else 1 - bit
        for bit in chromosome
    ]

# ============================
# Main GA Loop
# ============================
population = [create_individual() for _ in range(POPULATION_SIZE)]

for gen in range(GENERATIONS):
    new_population = []
    for _ in range(POPULATION_SIZE):
        p1 = selection(population)
        p2 = selection(population)
        child = crossover(p1, p2)
        child = mutate(child)
        new_population.append(child)

    population = new_population

    # Track the best fitness in each generation
    best_fit = max(fitness(ind) for ind in population)
    print(f"Generation {gen+1}: Best Fitness = {best_fit}")

# ============================
# Final Best Chromosome
# ============================
best = max(population, key=fitness)
print("\nBest Chromosome Found:")
print(best)
print("Number of 1s:", sum(best))
print("Final Fitness:", fitness(best))
