# Genetic Algorithm for Continuous Function Optimization

from tkinter import *
import matplotlib.pyplot as plt
import numpy as np
import random
import threading
from math import *

# -----------------------------
# GUI Input Window
# -----------------------------
def init_input_window():
    """
    This function creates a window to get GA parameters from the user.
    comment: I made this function to keep UI code separate from GA logic.
    """
    params = {}

    def get_data():
        try:
            # read inputs from user and store in dictionary
            params['function_str'] = e1_var.get()  # the function to optimize
            params['pop_size'] = max(2, int(v2.get()))  # make sure pop size >=2
            params['alpha'] = max(0.0, min(1.0, float(e2_var.get())))  # alpha between 0 and 1
            params['mut_dev'] = max(0.0, float(e3_var.get()))  # mutation deviation cannot be negative
            params['mut_rate'] = max(0.0, min(1.0, float(e4_var.get())))  # mutation probability 0-1
            params['n_iter'] = max(1, int(v3.get()))  # at least 1 iteration
            window.destroy()  # close the input window
        except Exception as e:
            error_lbl.config(text=f"Input Error: {e}")  # show error if user inputs wrong

    window = Tk()
    window.title("Genetic Optimization of Continuous Functions")
    window.iconbitmap("GAE.ico")
    window.geometry('950x380')
    window.resizable(False, False)

    # function input
    Label(window, text="Function: ").grid(column=0,row=0)
    e1_var = StringVar()
    e1 = Entry(window, textvariable=e1_var, width=140)
    e1.insert(END, '-1*x[0]**2-100')  # default function
    e1.grid(column=1,row=0,columnspan=6)

    # population size slider
    v2 = IntVar()
    Scale(window, variable=v2, from_=10, to=200, orient=HORIZONTAL, length=900,
          label="Population Size").grid(column=0,row=1,columnspan=8)

    # alpha input
    Label(window, text="Alpha: ").grid(column=0,row=2)
    e2_var = StringVar()
    e2 = Entry(window, textvariable=e2_var, width=10)
    e2.insert(END, '0.5')
    e2.grid(column=1,row=2)

    # deviation input
    Label(window, text="Deviation: ").grid(column=2,row=2)
    e3_var = StringVar()
    e3 = Entry(window, textvariable=e3_var, width=10)
    e3.insert(END, '2.5')
    e3.grid(column=3,row=2)

    # mutation rate input
    Label(window, text="Mutation Rate: ").grid(column=4,row=2)
    e4_var = StringVar()
    e4 = Entry(window, textvariable=e4_var, width=10)
    e4.insert(END, '0.001')
    e4.grid(column=5,row=2)

    # number of iterations
    v3 = IntVar()
    Scale(window, variable=v3, from_=1000, to=50000, orient=HORIZONTAL, length=900,
          label="Number of Iterations").grid(column=0,row=3,columnspan=8)

    # label to show error
    error_lbl = Label(window, text="", fg="red")
    error_lbl.grid(column=0,row=4,columnspan=8)

    Button(window, text="R U N", width=30, height=3, command=get_data).grid(column=0,row=5,columnspan=8)

    window.mainloop()
    return params

# -----------------------------
# GA Core Functions
# -----------------------------
def detect_dimension(func_str: str) -> int:
    """Detect dimension from function string"""
    # comment: I check string to see highest index, so we know problem dimension
    for i in range(9, 0, -1):
        if f'[{i}]' in func_str:
            return i+1
    return 1  # default 1D if no index found

def evaluate(func_str: str, individual: list[float]) -> float:
    """Evaluate individual"""
    x = individual
    try:
        return eval(func_str)
    except Exception:
        # comment: if eval fails, I return negative infinity so it does not break GA
        return -np.inf

def compute_fitness(population: list[list[float]], func_str: str) -> list[float]:
    """Compute fitness for all individuals"""
    # comment: just apply evaluate function to each individual
    return [evaluate(func_str, ind) for ind in population]

def selection(population: list[list[float]], func_str: str, pop_size: int) -> list[list[list[float]]]:
    """Rank-based selection of parents"""
    fitness = compute_fitness(population, func_str)
    ranks = np.argsort(fitness).argsort()+1
    probs = ranks / sum(ranks)
    # comment: I want higher ranked individuals to be more likely chosen
    selected_pairs = []
    indices = list(range(pop_size))
    sampled = np.random.choice(indices, 2*pop_size, p=probs)
    sampled_list = list(sampled)
    for _ in range(pop_size):
        #comment: pop two parents randomly
        p1 = sampled_list.pop(random.randint(0, len(sampled_list)-1))
        p2 = sampled_list.pop(random.randint(0, len(sampled_list)-1))
        selected_pairs.append([population[p1], population[p2]])
    return selected_pairs

def crossover(parents: list[list[list[float]]], alpha: float) -> list[list[float]]:
    """Arithmetic crossover"""
    kids = []
    for pair in parents:
        low = np.array(pair[0]) - alpha*(np.array(pair[1])-np.array(pair[0]))
        high = np.array(pair[1]) + alpha*(np.array(pair[1])-np.array(pair[0]))
        kid = np.random.uniform(low, high)
        #comment: I create new individual between b1 and b2
        kids.append(kid.tolist())
    return kids

def mutation(population: list[list[float]], mut_rate: float, mut_dev: float) -> list[list[float]]:
    """Gaussian mutation"""
    for i in range(len(population)):
        if random.random() < mut_rate:
            # comment: I add gaussian noise to each gene
            population[i] = (np.array(population[i]) + np.random.normal(0, mut_dev, len(population[i]))).tolist()
    return population

def elitism(old_pop: list[list[float]], new_pop: list[list[float]], func_str: str) -> list[list[float]]:
    """Keep best individual from old population"""
    old_fit = compute_fitness(old_pop, func_str)
    new_fit = compute_fitness(new_pop, func_str)
    elite_idx = old_fit.index(max(old_fit))
    worst_idx = new_fit.index(min(new_fit))
    new_pop[worst_idx] = old_pop[elite_idx]
    #comment: I replace worst in new pop with best from old pop
    return new_pop

# -----------------------------
# GA Loop
# -----------------------------
def GA_loop(params: dict, textbox: Text, CC2: Label, CC4: Label):
    """Run GA with all improvements"""
    pop_size = params['pop_size']
    n_iter = params['n_iter']
    alpha = params['alpha']
    mut_rate = params['mut_rate']
    mut_dev = params['mut_dev']
    func_str = params['function_str']
    dim = detect_dimension(func_str)

    population = [[random.randint(1,500) for _ in range(dim)] for _ in range(pop_size)]
    max_old = [None]*dim
    track_i, track_m = [], []

    for i in range(n_iter):
        parents = selection(population, func_str, pop_size)
        kids = crossover(parents, alpha)
        population_new = mutation(kids, mut_rate, mut_dev)
        population = elitism(population, population_new, func_str)

        fitness = compute_fitness(population, func_str)
        max_fit = max(fitness)
        maximizer = population[fitness.index(max_fit)]

        if maximizer != max_old:
            #comment: only update textbox if new maximizer is different
            textbox.insert(END, f"Iteration {i}\nMaximizer: {maximizer}\nMax: {max_fit}\n\n")
            max_old = maximizer
            if i != 0:
                track_i.append(i)
                track_m.append(max_fit)

        CC2.config(text=f"i={i} x*={np.round(maximizer,3)}")
        CC4.config(text=f"f(x)={round(max_fit,3)}")

    return track_i, track_m

def GA_threaded(params: dict, textbox: Text, CC2: Label, CC4: Label):
    """Run GA loop in separate thread to prevent UI freezing"""
    threading.Thread(target=GA_loop, args=(params, textbox, CC2, CC4)).start()

# -----------------------------
# Status Window
# -----------------------------
def status_window(params):
    """GUI to show GA progress and results"""
    root = Tk()
    root.title("GA Status")
    root.geometry("800x600")
    root.resizable(False, False)
    root.iconbitmap("GAE.ico")

    scrollbar = Scrollbar(root)
    scrollbar.pack(side=RIGHT, fill=Y)
    textbox = Text(root, width=100)
    textbox.pack()
    textbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=textbox.yview)

    Label(root, text=f"Function={params['function_str']}").pack()
    Label(root, text=f"Pop={params['pop_size']} Alpha={params['alpha']} MutDev={params['mut_dev']} MutRate={params['mut_rate']} Iter={params['n_iter']}").pack()

    CC2 = Label(root, text="Current Maximizer")
    CC2.pack()
    CC4 = Label(root, text="Current Max")
    CC4.pack()

    Button(root, text="START", command=lambda: GA_threaded(params, textbox, CC2, CC4)).pack(pady=5)

    track_i, track_m = [], []
    def plot_progress():
        plt.plot(track_i, track_m)
        plt.title("GA Progress")
        plt.xlabel("Iteration")
        plt.ylabel("Max Fitness")
        plt.show(block=False)
    Button(root, text="PLOT", command=plot_progress).pack(pady=5)

    root.mainloop()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    params = init_input_window()
    status_window(params)