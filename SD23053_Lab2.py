# app.py - FIXED VERSION: Genetic Algorithm for Lab Report 2 - BSD3513
# Target: Evolve a chromosome with EXACTLY 50 ones → Fitness = 80

import streamlit as st
import random
import matplotlib.pyplot as plt
import time

# DEBUG: Show errors if any
st.set_page_config(page_title="GA - Exactly 50 Ones", layout="centered")
st.title("🧬 Genetic Algorithm: Find Chromosome with Exactly 50 Ones")
st.markdown("**BSD3513 - Lab Report 2 | Student ID: [Your ID] | Name: [Your Name]**")

# ========================== GA FUNCTIONS ==========================
def create_individual():
    return [random.randint(0, 1) for _ in range(80)]

def create_population():
    return [create_individual() for _ in range(300)]

def fitness(individual):
    ones = sum(individual)
    if ones == 50:
        return 80                    # Maximum fitness
    else:
        return max(0, 80 - abs(ones - 50) * 1.6)  # Penalty based on deviation

def select_parents(population):
    fitnesses = [fitness(ind) for ind in population]
    total = sum(fitnesses)
    if total == 0:
        return random.choices(population, k=2)
    probs = [f/total for f in fitnesses]
    return random.choices(population, weights=probs, k=2)

def crossover(p1, p2):
    point = random.randint(1, 79)
    c1 = p1[:point] + p2[point:]
    c2 = p2[:point] + p1[point:]
    return c1, c2

def mutate(individual, rate=0.01):
    mutated = individual[:]  # Copy to avoid mutating original
    for i in range(len(mutated)):
        if random.random() < rate:
            mutated[i] = 1 - mutated[i]
    return mutated

def run_ga():
    population = create_population()
    history = []
    
    for gen in range(50):
        fitnesses = [fitness(ind) for ind in population]
        best_idx = fitnesses.index(max(fitnesses))
        best = population[best_idx]
        best_ones = sum(best)
        
        history.append({
            'gen': gen + 1,
            'best_fitness': fitness(best),
            'avg_fitness': sum(fitnesses)/len(fitnesses),
            'best_ones': best_ones
        })
        
        if best_ones == 50:
            return history, best, population  # Early exit on success
        
        # Elitism + create new population
        new_pop = [best]
        while len(new_pop) < 300:
            p1, p2 = select_parents(population)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)  # Fixed: Pass copy
            c2 = mutate(c2)
            new_pop.extend([c1, c2])
        population = new_pop[:300]
    
    return history, population[0], population  # Return best from final pop

# ========================== STREAMLIT UI ==========================
if st.button("🧬 Run Genetic Algorithm (50 Generations)", type="primary"):
    with st.spinner("Evolving population... ~10-20 seconds"):
        start = time.time()
        history, best_solution, _ = run_ga()
        elapsed = time.time() - start
    
    st.success(f"✅ Evolution complete in {elapsed:.2f} seconds!")
    
    # Results
    ones_count = sum(best_solution)
    best_fit = fitness(best_solution)
    st.metric("Best Fitness Achieved", f"{best_fit:.2f}/80")
    st.metric("Number of Ones", ones_count, delta=ones_count - 50)
    
    if ones_count == 50:
        st.balloons()
        st.success("🎉 PERFECT SOLUTION: Exactly 50 ones achieved!")
    else:
        st.warning(f"🔍 Close! Best chromosome has {ones_count} ones (target: 50)")
    
    # Show best chromosome (first 40 + ... + last 40 for readability)
    chromosome_str = ''.join(map(str, best_solution))
    st.code(chromosome_str[:40] + "..." + chromosome_str[-40:], language="text")
    
    # Evolution Plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    gens = [h['gen'] for h in history]
    ax1.plot(gens, [h['best_fitness'] for h in history], label="Best Fitness", color="green", marker="o", linewidth=2)
    ax1.plot(gens, [h['avg_fitness'] for h in history], label="Average Fitness", color="blue", alpha=0.7)
    ax1.axhline(80, color="red", linestyle="--", label="Target = 80")
    ax1.set_ylabel("Fitness Score")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(gens, [h['best_ones'] for h in history], color="purple", marker="o", linewidth=2)
    ax2.axhline(50, color="red", linestyle="--", label="Target = 50 ones")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Number of 1s in Best Chromosome")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

st.info("📱 Click the button above to run the GA. Reload the page if needed.")

# Footer
st.markdown("---")
st.markdown("**GitHub Repo:** [Paste your GitHub link]  |  **Live Demo:** This page  |  **Parameters:** Pop=300, Length=80, Gens=50, Target=50 ones")