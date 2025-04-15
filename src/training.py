import numpy as np
import torch

from src.predictor import UsagePredictor
from src.optimizer import sade_optimizer

# Set device (for Apple Silicon, use MPS if available)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


def train_predictor_with_split(
    vms,
    input_window=500,
    hidden_size=30,
    generations=50,
    population_size=30,
    F=0.5,
    CR=0.9,
    train_ratio=0.75,
):
    X_list = []
    y_list = []
    print(f"Preparing data for {len(vms)} VMs...")
    for vm in vms:
        hist = vm.history
        if len(hist) < input_window + 1:
            continue
        for i in range(len(hist) - input_window):
            X_list.append(hist[i : i + input_window])
            y_list.append(hist[i + input_window])
    print(f"Data prepared: {len(X_list)} samples.")
    X_all = np.array(X_list, dtype=np.float32)
    y_all = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    D_min = X_all.min()
    D_max = X_all.max()
    X_norm = (X_all - D_min) / (D_max - D_min + 1e-8)
    y_norm = (y_all - D_min) / (D_max - D_min + 1e-8)
    indices = np.arange(len(X_norm))
    np.random.shuffle(indices)
    split_index = int(train_ratio * len(indices))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    X_train = torch.from_numpy(X_norm[train_indices]).to(device)
    y_train = torch.from_numpy(y_norm[train_indices]).to(device)
    X_test = torch.from_numpy(X_norm[test_indices]).to(device)
    y_test = torch.from_numpy(y_norm[test_indices]).to(device)
    model = UsagePredictor(
        input_size=input_window, hidden_size=hidden_size, dropout_prob=0.5
    ).to(device)
    print("Starting SADE optimization for the predictor...")
    model = sade_optimizer(
        model,
        X_train,
        y_train,
        population_size=population_size,
        generations=generations,
        F=F,
        CR=CR,
    )
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
    test_rmse = np.sqrt(
        ((outputs.cpu().numpy() - y_test.cpu().numpy()) ** 2).mean()
    )  # noqa
    test_mae = np.abs((outputs.cpu().numpy() - y_test.cpu().numpy())).mean()
    print(f"Test RMSE (normalized): {test_rmse:.6f}")
    print(f"Test MAE (normalized): {test_mae:.6f}")
    return model, test_rmse, test_mae, D_min, D_max


def predict_vm_usage(model, vm, input_window=500, D_min=0, D_max=1):
    if len(vm.history) < input_window:
        return vm.cpu_demand
    import torch

    input_data = np.array(vm.history[-input_window:], dtype=np.float32)
    input_norm = (input_data - D_min) / (D_max - D_min + 1e-8)
    input_tensor = torch.from_numpy(input_norm).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        predicted_norm = model(input_tensor).item()
    predicted = predicted_norm * (D_max - D_min) + D_min
    return predicted
