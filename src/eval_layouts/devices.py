"""Choose a free server GPU before starting another isolated Kit process."""

import subprocess


def choose_device(requested, reserve_mib=12000):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.total,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    available = {}
    for line in result.stdout.splitlines():
        index, total, used = map(int, line.split(","))
        available[index] = total - used
    selected = max(available, key=available.get)
    if available.get(requested, 0) >= max(reserve_mib, available[selected] * 0.75):
        return requested
    print("EVAL_GPU_FALLBACK", requested, "->", selected, "free_mib", available[selected], flush=True)
    return selected
