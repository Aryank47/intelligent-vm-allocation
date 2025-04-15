import numpy as np
import math
import random
from .devices import PhysicalMachine
from typing import List


def compute_total_power(pms: List[PhysicalMachine], method=None):
    if method and method == "RELIABLE":
        # mul = random.choice([2, 3, 4])
        return 2 * sum(
            pm.compute_power() for pm in pms if len(pm.allocated_vms) > 0
        )  # noqa
    else:
        return sum(
            pm.compute_power() for pm in pms if len(pm.allocated_vms) > 0
        )  # noqa


def count_active_pms(pms, method=None):
    if method and method == "RELIABLE":
        # mul = random.choice([2, 3, 4])
        return 2 * sum(1 for pm in pms if len(pm.allocated_vms) > 0)
    else:
        return sum(1 for pm in pms if len(pm.allocated_vms) > 0)


def compute_system_reliability(vms, pms, beta=3.0):
    vm_dict = {vm.id: vm for vm in vms}
    pm_rels = []
    for pm in pms:
        if not pm.allocated_vms:
            continue
        utilization = pm.current_utilization()
        scaled_hazard = pm.hazard_rate * (utilization**beta)
        vm_rels = []
        for vm_id in pm.allocated_vms:
            vm = vm_dict[vm_id]
            effective_job_length = vm.job_length * 1.5
            r_vm = math.exp(-scaled_hazard * effective_job_length)
            vm_rels.append(r_vm)
        if vm_rels:
            pm_rel = math.exp(np.mean(np.log(vm_rels)))
            pm_rels.append(pm_rel)
    if pm_rels:
        system_rel = np.mean(pm_rels)
    else:
        system_rel = 1.0
    return system_rel


def evaluate_allocation(
    individual, vms, pms, beta, weight_power, weight_rel, lambda_active=0.2
):
    from src.devices import reset_pms

    reset_pms(pms)
    for i, pm_idx in enumerate(individual):
        if pm_idx < 0 or pm_idx >= len(pms):
            continue
        pm = pms[pm_idx]
        vm = vms[i]
        pm.avail_cpu -= vm.cpu_demand
        pm.avail_ram -= vm.ram_demand
        pm.allocated_vms.append(vm.id)
        vm.allocated_pm = pm.id
    penalty = 0
    for pm in pms:
        if pm.avail_cpu < 0:
            penalty += abs(pm.avail_cpu) * 500
        if pm.avail_ram < 0:
            penalty += abs(pm.avail_ram) * 500
    total_power = sum(pm.compute_power() for pm in pms)
    sum_idle = sum(pm.power_idle for pm in pms)
    sum_max = sum(pm.power_max for pm in pms)
    norm_power = (total_power - sum_idle) / (sum_max - sum_idle + 1e-8)
    risk = compute_system_reliability(vms, pms)
    active_pm_count = count_active_pms(pms)
    norm_active = active_pm_count / len(pms)
    obj = (
        weight_power * norm_power
        + weight_rel * risk
        + lambda_active * norm_active
        + penalty
    )
    return obj


def ga_multi_objective_allocation(
    vms,
    pms,
    beta=3.0,
    pop_size=50,
    generations=100,
    weight_power=0.5,
    weight_rel=0.5,
    mutation_rate=0.1,
    crossover_rate=0.8,
):
    num_vms = len(vms)
    num_pms = len(pms)
    population = [
        [random.randint(0, num_pms - 1) for _ in range(num_vms)]
        for _ in range(pop_size)
    ]
    fitness = []
    for individual in population:
        import copy

        pms_copy = copy.deepcopy(pms)
        fit = evaluate_allocation(
            individual, vms, pms_copy, beta, weight_power, weight_rel
        )
        fitness.append(fit)
    for gen in range(generations):
        new_population = []
        new_fitness = []
        for _ in range(pop_size):
            i1, i2 = random.sample(range(pop_size), 2)
            parent1 = (
                population[i1] if fitness[i1] < fitness[i2] else population[i2]
            )  # noqa
            i1, i2 = random.sample(range(pop_size), 2)
            parent2 = (
                population[i1] if fitness[i1] < fitness[i2] else population[i2]
            )  # noqa
            if random.random() < crossover_rate:
                point = random.randint(1, num_vms - 1)
                child = parent1[:point] + parent2[point:]
            else:
                child = parent1.copy()
            for j in range(num_vms):
                if random.random() < mutation_rate:
                    child[j] = random.randint(0, num_pms - 1)

            pms_copy = copy.deepcopy(pms)
            child_fit = evaluate_allocation(
                child, vms, pms_copy, beta, weight_power, weight_rel
            )
            new_population.append(child)
            new_fitness.append(child_fit)
        combined = population + new_population
        combined_fitness = fitness + new_fitness
        sorted_indices = sorted(
            range(len(combined)), key=lambda i: combined_fitness[i]
        )  # noqa
        population = [combined[i] for i in sorted_indices[:pop_size]]
        fitness = [combined_fitness[i] for i in sorted_indices[:pop_size]]
    best_allocation = population[0]
    return best_allocation, fitness[0]
