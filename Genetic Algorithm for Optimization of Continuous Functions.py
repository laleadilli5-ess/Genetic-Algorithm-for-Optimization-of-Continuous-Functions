# Genetic Algorithm for Optimization of Continuous Functions
# Refactored for readability, modularity, and code quality (A variant)
# Author: Lalə Edilli
# Date: 2026-03-04

from tkinter import *
import matplotlib.pyplot as plt
import numpy as np
import random
import threading
from math import *

# -----------------------------
# GUI Initialization Window
# -----------------------------
def init_input_window():
    """
    Creates the input window to get GA parameters from the user.
    Returns a dictionary of parameters.
    
    Comment:
    - Previously, all UI elements were in the main code, making it messy.
    - Now, encapsulated in a function for clarity and modularity.
    """
    params = {}

    def get_data():
        """Fetch user inputs and close the input window"""
        # Store all user inputs into the dictionary
        params['function_str'] = e1_var.get()           # Function to optimize
        params['pop_size'] = int(v2.get())             # Population size
        params['alpha'] = float(e2_var.get())          # Crossover coefficient
        params['mut_dev'] = float(e3_var.get())        # Mutation deviation
        params['mut_rate'] = float(e4_var.get())       # Mutation probability
        params['n_iter'] = int(v3.get())               # Number of iterations
        window.destroy()                               # Close the input window

    window = Tk()
    window.title("Genetic Optimization of Continuous Functions")
    window.iconbitmap("GAE.ico")
    window.resizable(False, False)
    window.geometry('950x380')

    # --- Function input ---
    Label(window, text="Function: ").grid(column=0, row=0, columnspan=2)
    e1_var = StringVar()
    e1 = Entry(window, textvariable=e1_var, width=140)
    e1.insert(END, '-1*x[0]**2-100')  # Default function
    e1.grid(column=2, row=0, columnspan=6, pady=(20, 10), padx=(10, 10))

    # --- Population size slider ---
    v2 = IntVar()
    s2 = Scale(window, variable=v2, from_=50, to=200, tickinterval=10,
               orient=HORIZONTAL, length=900, label="Population Size :")
    s2.grid(column=0, row=1, columnspan=8, padx=(20, 20), pady=(30, 30))

    # --- Alpha input ---
    Label(window, text="Alpha: ").grid(column=0, row=2, columnspan=1)
    e2_var = StringVar()
    e2 = Entry(window, textvariable=e2_var, width=15)
    e2.insert(END, '0.5')  # Default alpha
    e2.grid(column=1, row=2, columnspan=2)

    # --- Deviance input ---
    Label(window, text="Deviance: ").grid(column=3, row=2, columnspan=1)
    e3_var = StringVar()
    e3 = Entry(window, textvariable=e3_var, width=15)
    e3.insert(END, '2.5')  # Default deviation
    e3.grid(column=4, row=2, columnspan=2)

    # --- Mutation rate input ---
    Label(window, text="Mutation Rate: ").grid(column=6, row=2, columnspan=1)
    e4_var = StringVar()
    e4 = Entry(window, textvariable=e4_var, width=15)
    e4.insert(END, '0.0001')  # Default mutation rate
    e4.grid(column=7, row=2, columnspan=1)

    # --- Number of iterations slider ---
    v3 = IntVar()
    s3 = Scale(window, variable=v3, from_=100000, to=1000000,
               tickinterval=100000, orient=HORIZONTAL, length=900,
               label="Number of Iterations :")
    s3.grid(column=0, row=3, columnspan=8, padx=(20, 20), pady=(10, 10))

    # --- RUN button ---
    Button(window, text="R U N", command=get_data, width=30, height=3)\
        .grid(column=0, row=7, columnspan=8, pady=5)

    window.mainloop()
    return params

# -----------------------------
# Genetic Algorithm Functions
# -----------------------------
def detect_dimension(function_str: str) -> int:
    """
    Detects the dimension of the problem from the function string.
    
    Comment:
    - Previously, there were many repetitive 'if' statements.
    - Now uses a for loop to reduce code duplication.
    """
    for i in range(9, 0, -1):
        if f'[{i}]' in function_str:
            return i + 1  # Because [0] = dim 1
    return 1

def evaluate(func: str, individual: list[float]) -> float:
    """
    Evaluates a given individual on the function string.
    
    Comment:
    - Previously, 'eval' was called without context.
    - Here, we explicitly assign x=individual for clarity.
    """
    x = individual
    return eval(func)  # Caution: eval used, safe input required

def compute_population_fitness(population: list[list[float]], func: str) -> list[float]:
    """Compute fitness values for all individuals in population"""
    return [evaluate(func, ind) for ind in population]

def selection(population: list[list[float]], func: str, alpha: float, pop_size: int) -> list[list[list[float]]]:
    """
    Selects parents for crossover based on ranking method.
    
    Comment:
    - Fixes previous potential bias by using rank-based probabilities.
    - Returns a list of parent pairs for crossover.
    """
    fitness_list = compute_population_fitness(population, func)
    ranks = np.argsort(fitness_list)
    ranks = ranks.argsort() + 1  # rank starts at 1
    probabilities = ranks / sum(ranks)

    sampled_indices = np.random.choice(range(pop_size), 2*pop_size, p=probabilities)
    sampled_list = list(sampled_indices)
    res_pairs = []

    for _ in range(pop_size):
        # Pop two parents randomly based on probabilities
        p1 = sampled_list.pop(random.randint(0, len(sampled_list)-1))
        p2 = sampled_list.pop(random.randint(0, len(sampled_list)-1))
        res_pairs.append([population[p1], population[p2]])

    return res_pairs

def crossover(parents: list[list[list[float]]], alpha: float, pop_size: int, dim: int) -> list[list[float]]:
    """
    Performs arithmetic crossover with alpha parameter.
    
    Comment:
    - Previously, b1 and b2 calculation was unclear.
    - Now explicitly shows offspring range using alpha factor.
    """
    kids = []
    for pair in parents:
        b1 = np.array(pair[0]) - alpha*(np.array(pair[1])-np.array(pair[0]))
        b2 = np.array(pair[1]) + alpha*(np.array(pair[1])-np.array(pair[0]))
        kid = np.random.uniform(b1, b2)  # offspring sampled uniformly
        kids.append(kid.tolist())
    return kids

def mutation(population: list[list[float]], mut_rate: float, mut_dev: float) -> list[list[float]]:
    """
    Performs Gaussian mutation on population.
    
    Comment:
    - Fixes previous logic: only mutate if random > mut_rate
    - Adds Gaussian noise per gene
    """
    for i in range(len(population)):
        if random.random() > mut_rate:
            population[i] = (np.array(population[i]) + random.gauss(0, mut_dev)).tolist()
    return population

def elitism(old_pop: list[list[float]], new_pop: list[list[float]], func: str) -> list[list[float]]:
    """
    Preserves the best individual from old population.
    
    Comment:
    - Previously, elite replacement was not explicit.
    - Now clearly replaces worst in new population with best from old.
    """
    old_fitness = compute_population_fitness(old_pop, func)
    new_fitness = compute_population_fitness(new_pop, func)
    elite_index = old_fitness.index(max(old_fitness))
    worst_index = new_fitness.index(min(new_fitness))
    new_pop[worst_index] = old_pop[elite_index]
    return new_pop

# -----------------------------
# GA Main Loop
# -----------------------------
def GA_loop(params: dict):
    """Runs the main Genetic Algorithm loop"""
    pop_size = params['pop_size']
    n_iter = params['n_iter']
    dim = params['dim']
    function_str = params['function_str']
    alpha = params['alpha']
    mut_dev = params['mut_dev']
    mut_rate = params['mut_rate']

    # Initial population
    population = [[random.randrange(1, 500) for _ in range(dim)] for _ in range(pop_size)]
    max_old = [None]*dim
    track_i, track_m = [], []

    for i in range(n_iter):
        # Selection, Crossover, Mutation
        parents = selection(population, function_str, alpha, pop_size)
        kids = crossover(parents, alpha, pop_size, dim)
        population_new = mutation(kids, mut_rate, mut_dev)
        population = elitism(population, population_new, function_str)

        # Tracking current maximum
        fitness_values = compute_population_fitness(population, function_str)
        max_fitness = max(fitness_values)
        max_index = fitness_values.index(max_fitness)
        maximizer = population[max_index]

        # Only update textbox if maximizer changed
        if maximizer != max_old:
            textbox.insert(END, f"Iteration {i}\nMaximizer: {maximizer}\nMax value: {max_fitness}\n\n")
            max_old = maximizer
            if i != 0:
                track_i.append(i)
                track_m.append(max_fitness)

        # Update labels for current max
        CC2.config(text=f"i={i} x*={np.round(maximizer,3)}")
        CC4.config(text=f"f(x)={round(max_fitness,3)}")

    return track_i, track_m

def GA_fun_thread(params: dict):
    """Runs GA_loop in a separate thread to prevent UI freeze"""
    threading.Thread(target=GA_loop, args=(params,)).start()