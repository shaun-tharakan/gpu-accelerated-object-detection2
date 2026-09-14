import torch

def check_device_availability():
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU Only"
    return {
        "cuda_available": cuda_available,
        "device_name": device_name,
        "device_count": torch.cuda.device_count() if cuda_available else 0
    }

def reset_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def get_peak_gpu_memory_mb():
    if torch.cuda.is_available():
        peak_bytes = torch.cuda.max_memory_allocated()
        return round(peak_bytes / (1024 * 1024), 2)
    return 0.0

if __name__ == "__main__":
    print("Device Info:", check_device_availability())