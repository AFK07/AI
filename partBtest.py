import random as rd
import numpy as np

# Problem Data: Items with weight (tonnes) and value (£ thousands), 
items = [
    {"type": 1, "weight": 3, "value": 126},
    {"type": 2, "weight": 8, "value": 154},
    {"type": 3, "weight": 2, "value": 256},
    {"type": 4, "weight": 9, "value": 526},
    {"type": 5, "weight": 7, "value": 388},
    {"type": 6, "weight": 1, "value": 245},
    {"type": 7, "weight": 8, "value": 210},
    {"type": 8, "weight": 13, "value": 442},
    {"type": 9, "weight": 10, "value": 671},
    {"type": 10, "weight": 9, "value": 348},
]

# Constants
MAX_WEIGHT = 35  # Maximum load capacity (tonnes)
POP_SIZE = 20    # Number of individuals in the population
CHROMOSOME_LENGTH = len(items)
GENERATIONS = 100
MUTATION_RATE = 0.1


# Fitness Function
def fitness_function(individual):
    """
    Calculates the fitness of an individual.
    Fitness is the total value of selected items, penalized if the weight exceeds the maximum.
    """
    total_weight = sum(individual[i] * items[i]["weight"] for i in range(len(items)))
    total_value = sum(individual[i] * items[i]["value"] for i in range(len(items)))

    # Penalize solutions that exceed the maximum weight
    if total_weight > MAX_WEIGHT:
        return 0  # Invalid solution
    return total_value


# Initialization Function
def initialize_population(pop_size, chromosome_length):
    """
    Generates an initial population with binary chromosomes.
    Each gene represents whether an item is selected (1) or not (0).
    """
    return [rd.choices([0, 1], k=chromosome_length) for _ in range(pop_size)]


# Selection using Roulette Wheel
def roulette_wheel_selection(population, fitness_values):
    """
    Selects individuals proportionally to their fitness values.
    """
    total_fitness = sum(fitness_values)
    if total_fitness == 0:
        return rd.choices(population, k=len(population))  # Random selection if all fitness are 0

    probabilities = [f / total_fitness for f in fitness_values]
    return [population[np.searchsorted(np.cumsum(probabilities), rd.random())] for _ in range(len(population))]


# Selection using Stochastic Universal Sampling (SUS)
def stochastic_universal_sampling(population, fitness_values):
    """
    Selects parents using Stochastic Universal Sampling (SUS).
    """
    total_fitness = sum(fitness_values)
    if total_fitness == 0:
        return rd.choices(population, k=len(population))  # Random selection if all fitness are 0

    probabilities = [f / total_fitness for f in fitness_values]
    cumulative_fitness = np.cumsum(probabilities)
    step = 1 / len(population)
    start_point = rd.uniform(0, step)
    points = [start_point + i * step for i in range(len(population))]

    selected_parents = []
    for p in points:
        for i, cf in enumerate(cumulative_fitness):
            if p <= cf:
                selected_parents.append(population[i])
                break
    return selected_parents


# Crossover Function
def crossover(parent_a, parent_b):
    """
    Performs single-point crossover between two parents.
    """
    cut_point = rd.randint(1, len(parent_a) - 1)
    offspring_a = parent_a[:cut_point] + parent_b[cut_point:]
    offspring_b = parent_b[:cut_point] + parent_a[cut_point:]
    return offspring_a, offspring_b


# Mutation Function
def mutate(chromosome, mutation_rate=0.1):
    """
    Mutates a chromosome based on the given mutation rate.
    """
    for i in range(len(chromosome)):
        if rd.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]  # Flip the bit
    return chromosome


# Main Genetic Algorithm
def genetic_algorithm():
    """
    Runs the Genetic Algorithm to solve the problem.
    """
    # Step 1: Initialize Population
    population = initialize_population(POP_SIZE, CHROMOSOME_LENGTH)

    for generation in range(GENERATIONS):
        # Step 2: Evaluate Fitness
        fitness_values = [fitness_function(ind) for ind in population]

        # Step 3: Selection (Using SUS here, replace with roulette_wheel_selection if needed)
        selected_parents = stochastic_universal_sampling(population, fitness_values)

        # Step 4: Crossover
        next_generation = []
        while len(next_generation) < POP_SIZE:
            parent_a, parent_b = rd.sample(selected_parents, 2)
            offspring_a, offspring_b = crossover(parent_a, parent_b)
            next_generation.extend([offspring_a, offspring_b])

        # Step 5: Mutation
        next_generation = [mutate(ind, MUTATION_RATE) for ind in next_generation]

        # Step 6: Replace Population
        population = next_generation[:POP_SIZE]

        # Step 7: Print Progress
        max_fitness = max([fitness_function(ind) for ind in population])
        print(f"Generation {generation + 1}: Max Fitness = {max_fitness}")

    # Step 8: Return Best Solution
    fitness_values = [fitness_function(ind) for ind in population]
    best_individual = population[np.argmax(fitness_values)]
    best_fitness = max(fitness_values)
    return best_individual, best_fitness


# Run the Genetic Algorithm
best_solution, best_profit = genetic_algorithm()

# Output Results
print("\nBest Solution (Item Selection):", best_solution)
print("Best Profit (£ thousands):", best_profit)

# List Selected Items
selected_items = [items[i] for i in range(len(items)) if best_solution[i] == 1]
print("\nSelected Items:")
for item in selected_items:
    print(f"Type: {item['type']}, Weight: {item['weight']} tonnes, Value: {item['value']}k") 
