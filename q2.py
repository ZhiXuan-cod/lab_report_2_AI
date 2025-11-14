# ================================================================
# BSD3513 – LAB REPORT 2
# Genetic Algorithm: Generate Bit Pattern with Exactly 50 Ones
# Student ID: STUDENT_ID
# Name: YOUR_NAME
# ================================================================

import streamlit as st
import random
import matplotlib.pyplot as plt
import time

# ---------------------------------------------------------------
# Streamlit UI Setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Genetic Algorithm – 50 Ones", layout="centered")
st.title("🧬 Genetic Algorithm – Generate Bit Pattern (80 bits, 50 Ones)")
st.markdown("### BSD3513 – Lab Report 2  
**Student ID:** STUDENT_ID  
**Name:** YOUR_NAME  
")

# ---------------------------------------------------------------
# GA FUNCTIONS
# ---------------------------------------------------------------

def create_individual():
    """
    Create a chromosome of length 80 consisting of bits (0/1).
    Requirement: Individual length = 80.
    """
    return [random.randint(0, 1) for _ in range(80)]


def create_population():
    """
    Create a population of 300 individuals.
    Requirement: Population size = 300.
    """
    return [create_individual() for _ in range(300)]


def fitness(individual):
    """
    Fitness function required by Question 2.

    The formula must reach maximum value = 80
    when the number of ones equals 50.

    Penalization is linear based on distance from 50.
    """
    ones = sum(individual)
    if ones == 50:
        return 80
    else:
        return max(0, 80 - abs(ones - 50) * 1.6)


def select_parents(population):
    """
    Roulette wheel selection with fallback.
    If all fitness values are zero, randomly select two parents.
    """
    fitnesses = [fitness(ind) for ind in population]
    total = sum(fitnesses)

    if total == 0:
        return random.sample(population, 2)

    probs = [f / total for f in fitnesses]
    return random.choices(population, weights=probs, k=2)


def crossover(p1, p2):
    """
    Single-point crossover at random position.
    """
    point = random.randint(1, 79)
    c1 = p1[:point] + p2[point:]
    c2 = p2[:point] + p1[point:]
    return c1, c2


def mutate(individual, rate=0.01):
    """
    Bit-flip mutation with probability = 1%.
    """
    mutated = individual[:]
    for i in range(len(mutated)):
        if random.random() < rate:
            mutated[i] = 1 - mutated[i]
    return mutated


def run_ga():
    """
    Run genetic algorithm for 50 generations.
    Requirement: Number of generations = 50.
    """
    population = create_population()
    history = []

    for gen in range(50):
        fitnesses = [fitness(ind) for ind in population]
        best_idx = fitnesses.index(max(fitnesses))
        best = population[best_idx]
        best_ones = sum(best)

        # Store generation details
        history.append({
            'gen': gen + 1,
            'best_fitness': fitness(best),
            'avg_fitness': sum(fitnesses) / len(fitnesses),
            'best_ones': best_ones
        })

        # Early stopping if perfect solution found
        if best_ones == 50:
            return history, best, population

        # Elitism – keep best and reproduce rest
        new_pop = [best]

        while len(new_pop) < 300:
            p1, p2 = select_parents(population)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_pop.extend([c1, c2])

        population = new_pop[:300]

    return history, population[0], population


# ---------------------------------------------------------------
# STREAMLIT EXECUTION BLOCK
# ---------------------------------------------------------------
if st.button("▶ Run Genetic Algorithm (50 Generations)", type="primary"):
    with st.spinner("Running genetic algorithm... Please wait ~10 seconds"):
        start = time.time()
        history, best_solution, _ = run_ga()
        elapsed = time.time() - start

    st.success(f"GA completed in {elapsed:.2f} seconds!")

    ones_count = sum(best_solution)
    best_fit = fitness(best_solution)

    st.metric("Best Fitness Achieved", f"{best_fit}/80")
    st.metric("Total Ones in Best Chromosome", ones_count, delta=ones_count - 50)

    # Display Perfect Result
    if ones_count == 50:
        st.balloons()
        st.success("🎉 PERFECT RESULT FOUND: Exactly 50 ones!")
    else:
        st.warning(f"Best result: {ones_count} ones (target = 50)")

    # FULL CHROMOSOME OUTPUT (Required: Bit Pattern Generator)
    chromosome_str = ''.join(map(str, best_solution))
    st.text("Full Chromosome (80 bits):")
    st.code(chromosome_str, language="text")

    # ------------------- Evolution Plots ------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    gens = [h['gen'] for h in history]

    ax1.plot(gens, [h['best_fitness'] for h in history], marker="o", linewidth=2)
    ax1.axhline(80, color="red", linestyle="--", label="Target = 80")
    ax1.set_title("Best & Average Fitness per Generation")
    ax1.set_ylabel("Fitness")
    ax1.grid(True)

    ax2.plot(gens, [h['best_ones'] for h in history], marker="o", linewidth=2)
    ax2.axhline(50, color="red", linestyle="--", label="Target = 50 ones")
    ax2.set_title("Number of Ones in Best Chromosome")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Number of Ones")
    ax2.grid(True)

    plt.tight_layout()
    st.pyplot(fig)

# ---------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------
st.markdown("---")
st.markdown("**GitHub Repo:** <Paste your link>  
**Streamlit Deployment:** <Paste your app link>  
**Population = 300 | Length = 80 bits | Target = 50 ones | Generations = 50**")
