import random as rd
import numpy as np

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

MAX_WEIGHT = 35 
POP_SIZE = 20    
CHROMOSOME_LENGTH = len(items)
GENERATIONS = 100
MUTATION_RATE = 0.1

def fitness_function(individual):
    total_weight = sum(individual[i] * items[i]["weight"] for i in range(len(items)))
    total_value = sum(individual[i] * items[i]["value"] for i in range(len(items)))
    if total_weight > MAX_WEIGHT:
        return 0  
    return total_value

def initialize_population(pop_size, chromosome_length):
    return [rd.choices([0, 1], k=chromosome_length) for _ in range(pop_size)]

def roulette_wheel_selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    if total_fitness == 0:
        return rd.choices(population, k=len(population))  
    probabilities = [f / total_fitness for f in fitness_values]
    return [population[np.searchsorted(np.cumsum(probabilities), rd.random())] for _ in range(len(population))]

def stochastic_universal_sampling(population, fitness_values):
    total_fitness = sum(fitness_values)
    if total_fitness == 0:
        return rd.choices(population, k=len(population))  
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


def crossover(parent_a, parent_b):
    cut_point = rd.randint(1, len(parent_a) - 1)
    offspring_a = parent_a[:cut_point] + parent_b[cut_point:]
    offspring_b = parent_b[:cut_point] + parent_a[cut_point:]
    return offspring_a, offspring_b

def mutate(chromosome, mutation_rate=0.1):
    for i in range(len(chromosome)):
        if rd.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]  
    return chromosome

def genetic_algorithm():
    population = initialize_population(POP_SIZE, CHROMOSOME_LENGTH)
    for generation in range(GENERATIONS):
        fitness_values = [fitness_function(ind) for ind in population]
        selected_parents = stochastic_universal_sampling(population, fitness_values)
        next_generation = []
        while len(next_generation) < POP_SIZE:
            parent_a, parent_b = rd.sample(selected_parents, 2)
            offspring_a, offspring_b = crossover(parent_a, parent_b)
            next_generation.extend([offspring_a, offspring_b])
        next_generation = [mutate(ind, MUTATION_RATE) for ind in next_generation]
        population = next_generation[:POP_SIZE]
        max_fitness = max([fitness_function(ind) for ind in population])
        print(f"Generation {generation + 1}: Max Fitness = {max_fitness}")
    fitness_values = [fitness_function(ind) for ind in population]
    best_individual = population[np.argmax(fitness_values)]
    best_fitness = max(fitness_values)
    return best_individual, best_fitness
best_solution, best_profit = genetic_algorithm()
print("\nBest Solution (Item Selection):", best_solution)
print("Best Profit (£ thousands):", best_profit)
selected_items = [items[i] for i in range(len(items)) if best_solution[i] == 1]
print("\nSelected Items:")
for item in selected_items:
    print(f"Type: {item['type']}, Weight: {item['weight']} tonnes, Value: {item['value']}k") 
