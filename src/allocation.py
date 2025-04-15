from src.training import predict_vm_usage


def ffd_allocation(vms, pms):
    unallocated = [vm for vm in vms if vm.allocated_pm is None]
    unallocated.sort(key=lambda vm: vm.cpu_demand, reverse=True)
    sorted_pms = sorted(pms, key=lambda pm: pm.avail_cpu, reverse=True)
    for vm in unallocated:
        for pm in sorted_pms:
            if vm.cpu_demand <= pm.avail_cpu and vm.ram_demand <= pm.avail_ram:
                pm.avail_cpu -= vm.cpu_demand
                pm.avail_ram -= vm.ram_demand
                pm.allocated_vms.append(vm.id)
                vm.allocated_pm = pm.id
                break


def random_allocation(vms, pms):
    allocation = {}
    import random

    random.shuffle(vms)
    for vm in vms:
        allocated = False
        random_pms = pms.copy()
        random.shuffle(random_pms)
        for pm in random_pms:
            if vm.cpu_demand <= pm.avail_cpu and vm.ram_demand <= pm.avail_ram:
                pm.avail_cpu -= vm.cpu_demand
                pm.avail_ram -= vm.ram_demand
                pm.allocated_vms.append(vm.id)
                vm.allocated_pm = pm.id
                allocation[vm.id] = pm.id
                allocated = True
                break
        if not allocated:
            allocation[vm.id] = None
    # Secondary allocation with FFD for any unallocated VMs
    ffd_allocation(vms, pms)
    return {vm.id: vm.allocated_pm for vm in vms}


def reliable_allocation(
    vms,
    pms,
    use_prediction=False,
    max_utilization=None,
    pred_scaling=0.7,
    predictor=None,
    D_min=None,
    D_max=None,
):
    """
    Allocate VMs to PMs using a First-Fit Decreasing (FFD) heuristic.
    For predicted allocation, if predictor is provided,
    use it to update the VM's demand.
    """
    if use_prediction:
        max_utilization = 0.7  # allowed load up to 70% of capacity
    else:
        max_utilization = 1.0

    vms_sorted = sorted(vms, key=lambda vm: vm.cpu_demand, reverse=True)
    pms_sorted = sorted(pms, key=lambda pm: pm.total_cpu, reverse=True)

    for vm in vms_sorted:
        if (
            use_prediction
            and vm.predicted_cpu is not None
            and predictor is not None
        ):
            demand = predict_vm_usage(
                predictor, vm, input_window=500, D_min=D_min, D_max=D_max
            )
            vm.cpu_demand = demand
        else:
            demand = 1.2 * vm.cpu_demand

        allocated = False
        for pm in pms_sorted:
            if pm.avail_cpu >= demand and pm.avail_ram >= vm.ram_demand:
                used_cpu = pm.total_cpu - pm.avail_cpu
                if used_cpu + demand < pm.total_cpu * max_utilization:
                    pm.avail_cpu -= demand
                    pm.avail_ram -= vm.ram_demand
                    pm.allocated_vms.append(vm.id)
                    vm.allocated_pm = pm.id
                    allocated = True
                    break
        if not allocated:
            vm.allocated_pm = None
    return {vm.id: vm.allocated_pm for vm in vms}
