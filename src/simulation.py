import time
import copy
import pandas as pd
from src.devices import create_multiple_pms, reset_pms
from src.allocation import random_allocation, reliable_allocation
from src.metrics import (
    compute_total_power,
    count_active_pms,
    compute_system_reliability,
    ga_multi_objective_allocation,
)
from src.training import predict_vm_usage


def run_dynamic_consolidation(
    vms, pms, predictor, time_steps=5, D_min=0, D_max=100, beta=3.0
):
    reset_pms(pms)
    for vm in vms:
        vm.allocated_pm = None
    # Use predictive allocation (tighter consolidation)
    reliable_allocation(
        vms,
        pms,
        use_prediction=True,
        max_utilization=0.9,
        predictor=predictor,
        D_min=D_min,
        D_max=D_max,
    )
    results = []
    for t in range(time_steps):
        total_power = compute_total_power(pms)
        risk = compute_system_reliability(vms, pms)
        active_pms_count = count_active_pms(pms)
        results.append(
            {
                "time": t,
                "Total Power": total_power,
                "Reliability": risk,
                "ActivePMs": active_pms_count,
            }
        )
        for vm in vms:
            predicted_val = predict_vm_usage(
                predictor, vm, input_window=500, D_min=D_min, D_max=D_max
            )
            vm.history.pop(0)
            vm.history.append(predicted_val)
            vm.cpu_demand = max(1, int(predicted_val))
        reset_pms(pms)
        for vm in vms:
            vm.allocated_pm = None
        reliable_allocation(
            vms,
            pms,
            use_prediction=True,
            max_utilization=0.9,
            predictor=predictor,
            D_min=D_min,
            D_max=D_max,
        )
    return pd.DataFrame(results)


def run_full_grid_search_simulation(
    full_vms, predictor, test_rmse, D_min, D_max, scales, beta=3.0
):
    results = []
    for num_vms in scales:
        print(f"\nRunning simulation for {num_vms} VMs:")
        # Update each VM's predicted CPU usage and current CPU demand.
        for vm in full_vms[:num_vms]:
            pred_val = predict_vm_usage(
                predictor, vm, input_window=500, D_min=D_min, D_max=D_max
            )
            vm.predicted_cpu = pred_val
            vm.cpu_demand = int(pred_val)
        base_pms = create_multiple_pms(num_vms)
        # (a) Random Allocation
        pms1 = copy.deepcopy(base_pms)
        for vm in full_vms[:num_vms]:
            vm.allocated_pm = None
        start_time = time.time()
        random_allocation(full_vms[:num_vms], pms1)
        power_random = compute_total_power(pms1)
        active_random = count_active_pms(pms1)
        end_time = time.time()
        risk_random = compute_system_reliability(full_vms[:num_vms], pms1)
        results.append(
            {
                "VMs": num_vms,
                "Algorithm": "Random",
                "Total Power": power_random,
                "Active PMs": active_random,
                "Reliability": risk_random,
                "Test RMSE": test_rmse,
                "Exec. time (ms)": (end_time - start_time) * 1000,
            }
        )
        # (b) Reliable Allocation (Current Demand)
        pms2 = copy.deepcopy(base_pms)
        reset_pms(pms2)
        for vm in full_vms[:num_vms]:
            vm.allocated_pm = None
        start_time = time.time()
        reliable_allocation(full_vms[:num_vms], pms2, use_prediction=False)
        end_time = time.time()
        power_rel_current = compute_total_power(pms2, method="RELIABLE")
        active_rel_current = count_active_pms(pms2, method="RELIABLE")
        risk_rel_current = compute_system_reliability(full_vms[:num_vms], pms2)
        results.append(
            {
                "VMs": num_vms,
                "Algorithm": "Reliable_Current",
                "Total Power": power_rel_current,
                "Active PMs": active_rel_current,
                "Reliability": risk_rel_current,
                "Test RMSE": test_rmse,
                "Exec. time (ms)": (end_time - start_time) * 1000,
            }
        )
        # (c) Predicted Allocation (Using NN Predictions)
        pms3 = copy.deepcopy(base_pms)
        reset_pms(pms3)
        for vm in full_vms[:num_vms]:
            vm.allocated_pm = None
            pred_val = predict_vm_usage(
                predictor, vm, input_window=500, D_min=D_min, D_max=D_max
            )
            vm.cpu_demand = int(pred_val)
            vm.predicted_cpu = pred_val
        start_time = time.time()
        reliable_allocation(
            full_vms[:num_vms],
            pms3,
            use_prediction=True,
            max_utilization=0.7,
            pred_scaling=0.9,
            predictor=predictor,
            D_min=D_min,
            D_max=D_max,
        )
        end_time = time.time()
        power_predictive = compute_total_power(pms3)
        active_predictive = count_active_pms(pms3)
        risk_predictive = compute_system_reliability(full_vms[:num_vms], pms3)
        results.append(
            {
                "VMs": num_vms,
                "Algorithm": "Predictive",
                "Total Power": power_predictive,
                "Active PMs": active_predictive,
                "Reliability": risk_predictive,
                "Test RMSE": test_rmse,
                "Exec. time (ms)": (end_time - start_time) * 1000,
            }
        )
        # (d) GA-based Multi-Objective Allocation
        pms4 = copy.deepcopy(base_pms)
        reset_pms(pms4)
        for vm in full_vms[:num_vms]:
            vm.allocated_pm = None
        start_time = time.time()
        best_alloc, best_obj = ga_multi_objective_allocation(
            full_vms[:num_vms],
            pms4,
            beta=beta,
            pop_size=50,
            generations=100,
            weight_power=0.5,
            weight_rel=0.5,
        )
        end_time = time.time()
        for i, pm_idx in enumerate(best_alloc):
            vm = full_vms[i]
            pms4[pm_idx].avail_cpu -= vm.cpu_demand
            pms4[pm_idx].avail_ram -= vm.ram_demand
            pms4[pm_idx].allocated_vms.append(vm.id)
            vm.allocated_pm = pms4[pm_idx].id
        power_ga = compute_total_power(pms4)
        active_ga = count_active_pms(pms4)
        risk_ga = compute_system_reliability(full_vms[:num_vms], pms4)
        results.append(
            {
                "VMs": num_vms,
                "Algorithm": "GA_MultiObjective",
                "Total Power": power_ga,
                "Active PMs": active_ga,
                "Reliability": risk_ga,
                "Test RMSE": test_rmse,
                "Exec. time (ms)": (end_time - start_time) * 1000,
                "Objective": best_obj,
            }
        )
    df_results = pd.DataFrame(results)
    print("\nGrid Search Results:")
    print(df_results)
    output_csv = "outputs/grid_search_results.csv"
    df_results.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    return df_results
