import os
import matplotlib.pyplot as plt
from src.devices import create_synthetic_vms
from src.training import train_predictor_with_split
from src.simulation import run_full_grid_search_simulation
from src.plotting import plot_predicted_vs_actual_cpu


os.makedirs("outputs", exist_ok=True)

# Simulation run
if __name__ == "__main__":

    full_vms = create_synthetic_vms(
        num_vms=5000, history_length=800, scale_factor=100
    )  # noqa

    predictor, test_rmse, test_mae, D_min, D_max = train_predictor_with_split(
        full_vms,
        input_window=500,
        hidden_size=25,
        generations=50,
        population_size=25,
        F=0.5,
        CR=0.9,
        train_ratio=0.65,
    )
    print(f"Trained predictor once; Test RMSE (normalized): {test_rmse:.6f}")

    scales = [50, 100, 200, 400, 600, 800, 1000, 1200, 1600]
    grid_results = run_full_grid_search_simulation(
        full_vms, predictor, test_rmse, D_min, D_max, scales=scales, beta=3.0
    )

    algorithms = grid_results["Algorithm"].unique()
    for metric in ["Total Power", "Active PMs", "Reliability"]:
        plt.figure(figsize=(10, 6))
        for algo in algorithms:
            subset = grid_results[grid_results["Algorithm"] == algo]
            plt.plot(subset["VMs"], subset[metric], marker="o", label=algo)
        plt.xlabel("Number of VMs")
        plt.ylabel(metric)
        plt.title(f"{metric} vs. Number of VMs")
        plt.legend()
        plt.savefig(f"outputs/{metric.replace(' ', '_')}_vs_VMs.png")
        plt.show()

    print("D_min =", D_min)
    print("D_max =", D_max)

    # Plot CPU prediction for a sample VM (e.g., the 501st VM)
    plot_predicted_vs_actual_cpu(
        full_vms[500], predictor, input_window=500, D_min=D_min, D_max=D_max
    )
