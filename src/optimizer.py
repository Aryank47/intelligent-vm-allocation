import numpy as np
import torch
import random


def get_flat_params(model):
    params = []
    for p in model.parameters():
        params.append(p.detach().cpu().numpy().flatten())
    return np.concatenate(params)


def set_flat_params(model, flat_vector):
    pointer = 0
    with torch.no_grad():
        for p in model.parameters():
            numel = p.numel()
            new_vals = flat_vector[pointer:pointer + numel]
            new_vals = new_vals.reshape(p.shape)
            p.copy_(torch.from_numpy(new_vals).type_as(p))
            pointer += numel


def fitness_function(model, X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
    loss = np.sqrt(((outputs.cpu().numpy() - y.cpu().numpy()) ** 2).mean())
    return loss


def sade_optimizer(
    model,
    X,
    y,
    population_size=30,
    generations=50,
    F=0.5,
    CR=0.9,
    mutation_learning_period=10,
):
    base = get_flat_params(model)
    D = base.shape[0]
    population = [
        base + np.random.randn(D) * 0.1 for _ in range(population_size)
    ]  # noqa
    fitness_vals = []
    for cand in population:
        set_flat_params(model, cand)
        fitness_vals.append(fitness_function(model, X, y))
    best_idx = np.argmin(fitness_vals)
    best_solution = population[best_idx].copy()
    best_fitness = fitness_vals[best_idx]
    strategies = [0, 1, 2, 3]
    strategy_success = [1, 1, 1, 1]
    for gen in range(generations):
        new_population = []
        new_fitness_vals = []
        for i in range(population_size):
            strategy = random.choices(strategies, weights=strategy_success, k=1)[0] # noqa
            indices = list(range(population_size))
            indices.remove(i)
            if strategy == 0:
                r1, r2 = random.sample(indices, 2)
                mutant = best_solution + F * (population[r1] - population[r2])
            elif strategy == 1:
                r1, r2 = random.sample(indices, 2)
                mutant = (
                    population[i]
                    + F * (best_solution - population[i])
                    + F * (population[r1] - population[r2])
                )
            elif strategy == 2:
                r1, r2, r3 = random.sample(indices, 3)
                mutant = population[r3] + F * (population[r1] - population[r2])
            elif strategy == 3:
                r1, r2, r3 = random.sample(indices, 3)
                k_i = random.random()
                mutant = (
                    population[i]
                    + k_i * (population[r1] - population[i])
                    + F * (population[r2] - population[r3])
                )
            trial = population[i].copy()
            for j in range(D):
                if random.random() < CR:
                    trial[j] = mutant[j]
            set_flat_params(model, trial)
            fit_trial = fitness_function(model, X, y)
            if fit_trial < fitness_vals[i]:
                new_population.append(trial)
                new_fitness_vals.append(fit_trial)
                strategy_success[strategy] += 1
            else:
                new_population.append(population[i])
                new_fitness_vals.append(fitness_vals[i])
        population = new_population
        fitness_vals = new_fitness_vals
        current_best_idx = np.argmin(fitness_vals)
        if fitness_vals[current_best_idx] < best_fitness:
            best_solution = population[current_best_idx].copy()
            best_fitness = fitness_vals[current_best_idx]
        if (gen + 1) % mutation_learning_period == 0:
            F = max(0.1, min(1.0, F * (1 + 0.05 * (np.random.rand() - 0.5))))
            CR = max(0.1, min(1.0, CR * (1 + 0.05 * (np.random.rand() - 0.5))))
            strategy_success = [1, 1, 1, 1]
        print(f"Generation {gen+1}, Best RMSE: {best_fitness:.6f}")
    set_flat_params(model, best_solution)
    return model
