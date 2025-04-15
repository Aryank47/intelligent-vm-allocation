import matplotlib.pyplot as plt
import numpy as np
import torch
from src.training import device


def plot_predicted_vs_actual_cpu(vm, predictor, input_window, D_min, D_max):
    actual = np.array(vm.history)
    n_samples = len(actual) - input_window
    windows = np.array(
        [actual[i:i + input_window] for i in range(n_samples)],
        dtype=np.float32,  # noqa
    )
    windows_norm = (windows - D_min) / (D_max - D_min + 1e-8)
    input_tensor = torch.from_numpy(windows_norm).to(device)
    predictor.eval()
    with torch.no_grad():
        preds_norm = predictor(input_tensor).cpu().numpy().flatten()
    predicted = preds_norm * (D_max - D_min) + D_min
    plt.figure(figsize=(8, 4))
    plt.plot(
        range(input_window, len(actual)),
        actual[input_window:],
        label="Actual CPU Usage",
    )
    plt.plot(
        range(input_window, len(actual)),
        predicted,
        label="Predicted CPU Usage",
        linestyle="--",
    )
    plt.xlabel("Time Step")
    plt.ylabel("CPU Usage")
    plt.title(f"Predicted vs Actual CPU Usage for {vm.id}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/predicted_vs_actual_{vm.id}.png")
    plt.show()
