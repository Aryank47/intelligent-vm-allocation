import numpy as np
import random


# Define Classes for PM and VM
class VirtualMachine:
    def __init__(
        self, id, cpu_demand, ram_demand, job_length, history, history_mem=None
    ):
        self.id = id
        self.cpu_demand = cpu_demand  # current CPU demand (units)
        self.ram_demand = ram_demand  # current RAM demand (GB)
        # job length (for reliability calculation)
        self.job_length = job_length
        self.history = history  # historical CPU usage (list of floats)
        # predicted future CPU usage (to be computed)
        self.predicted_cpu = None
        self.allocated_pm = None  # PM id after allocation
        self.history_mem = history_mem if history_mem else []


class PhysicalMachine:
    def __init__(
        self,
        id,
        cpu_capacity,
        ram_capacity,
        power_max,
        power_idle,
        hazard_rate,
        reliability_score,
    ):
        self.id = id
        self.total_cpu = cpu_capacity
        self.avail_cpu = cpu_capacity
        self.total_ram = ram_capacity
        self.avail_ram = ram_capacity
        self.power_max = power_max
        self.power_idle = power_idle
        self.hazard_rate = hazard_rate  # used as lambda_max (1/MTBF)
        self.reliability_score = (
            reliability_score  # (not used directly in our calculation)
        )
        self.allocated_vms = []  # list of VM ids

    def reset(self):
        self.avail_cpu = self.total_cpu
        self.avail_ram = self.total_ram
        self.allocated_vms = []

    def current_utilization(self):
        used_cpu = self.total_cpu - self.avail_cpu
        return used_cpu / self.total_cpu if self.total_cpu > 0 else 0

    def compute_power(self):
        # Power = P_idle + (P_max - P_idle)*utilization.
        utilization = self.current_utilization()
        return (
            self.power_idle + (self.power_max - self.power_idle) * utilization
        )  # noqa


# Synthetic Data Generation (with turbulence)
def create_synthetic_vms(num_vms=20, history_length=1000, scale_factor=50):
    """
    Create synthetic VMs with noisy and turbulent CPU usage history.
    """

    def generate_synthetic_history(length=10, base=50, noise_level=15):
        t = np.arange(length)
        history = (
            base
            + 10 * np.sin(2 * np.pi * t / length)
            + np.random.randn(length) * noise_level
        )
        spikes = np.random.choice(
            [0, 20 * np.random.rand()], size=length, p=[0.95, 0.05]
        )
        history = history + spikes
        return np.clip(history, 0, None).tolist()

    vms = []
    for i in range(num_vms):
        history = generate_synthetic_history(
            length=history_length, base=random.uniform(30, 70), noise_level=10
        )
        cpu_demand = int(history[-1] * scale_factor)
        ram_demand = random.choice([1, 2, 4])
        job_length = random.uniform(2, 5)
        vms.append(
            VirtualMachine(
                id=f"VM{i+1}",
                cpu_demand=cpu_demand,
                ram_demand=ram_demand,
                job_length=job_length,
                history=history,
            )
        )
    return vms


# Physical Machines generation


def create_multiple_pms(num_vms):
    num_pms = max(1, int(np.ceil(num_vms / 1.67)))
    pm_list = []
    pm_types = ["S1", "S2", "S3"]
    probabilities = [0.4, 0.3, 0.3]
    MTBF_S1 = 10
    MTBF_S2 = 12
    MTBF_S3 = 15
    for i in range(num_pms):
        pm_type = random.choices(pm_types, weights=probabilities, k=1)[0]
        if pm_type == "S1":
            cpu_capacity = 5320
            ram_capacity = 4
            power_max = 135
            power_idle = 93.7
            hazard_rate = 1.0 / MTBF_S1
            reliability_score = 0.80
        elif pm_type == "S2":
            cpu_capacity = 12268
            ram_capacity = 8
            power_max = 113
            power_idle = 42.3
            hazard_rate = 1.0 / MTBF_S2
            reliability_score = 0.70
        elif pm_type == "S3":
            cpu_capacity = 36804
            ram_capacity = 16
            power_max = 222
            power_idle = 58.4
            hazard_rate = 1.0 / MTBF_S3
            reliability_score = 0.90
        pm_list.append(
            PhysicalMachine(
                id=f"PM{i+1}",
                cpu_capacity=cpu_capacity,
                ram_capacity=ram_capacity,
                power_max=power_max,
                power_idle=power_idle,
                hazard_rate=hazard_rate,
                reliability_score=reliability_score,
            )
        )
    print(f"Total PMs created: {len(pm_list)}")
    return pm_list


def reset_pms(pms):
    for pm in pms:
        pm.reset()
