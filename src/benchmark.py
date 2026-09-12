import os
import sys
import cv2
import numpy as np
import pandas as pd

# Add root folder to sys.path so imports work smoothly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detection import YOLODetector
from src.inference import run_inference
from src.gpu_utils import get_peak_gpu_memory_mb, reset_gpu_memory

def run_batch_benchmark(image_path, batch_size=1, device="cpu", warmup_iters=5, measure_iters=50, model_name="yolov8n.pt"):
    """
    Runs a controlled batch-image benchmark.
    """
    img = cv2.imread(image_path)
    
    # Auto-generate a valid synthetic image if the file cannot be loaded or is corrupted
    if img is None:
        print(f"Generating valid sample image at: {image_path}")
        img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
        cv2.imwrite(image_path, img)

    image_size = f"{img.shape[1]}x{img.shape[0]}"
    
    # Create batch list
    batch_input = [img for _ in range(batch_size)]
    
    print(f"\n--- Benchmarking {model_name} on {device.upper()} (Batch Size: {batch_size}) ---")
    detector = YOLODetector(model_name=model_name)
    
    if device == "cuda":
        reset_gpu_memory()

    # Warm-up Iterations
    print(f"Performing {warmup_iters} warm-up iterations...")
    for _ in range(warmup_iters):
        _ = run_inference(detector, batch_input, device=device)
        
    # Measured Iterations
    print(f"Performing {measure_iters} measured iterations...")
    total_latency_ms = 0.0
    
    for _ in range(measure_iters):
        _, latency_ms = run_inference(detector, batch_input, device=device)
        total_latency_ms += latency_ms
        
    avg_latency_ms = total_latency_ms / measure_iters
    avg_latency_sec = avg_latency_ms / 1000.0
    
    throughput_fps = batch_size / avg_latency_sec if avg_latency_sec > 0 else 0
    vram_mb = get_peak_gpu_memory_mb() if device == "cuda" else 0.0

    record = {
        "device": device,
        "model": model_name,
        "batch_size": batch_size,
        "image_size": image_size,
        "warmup_count": warmup_iters,
        "measured_count": measure_iters,
        "avg_latency_ms": round(avg_latency_ms, 2),
        "throughput_fps": round(throughput_fps, 2),
        "gpu_memory_mb": vram_mb
    }
    
    return record

def save_results_to_csv(results, output_path="results/raw/benchmark_results.csv"):
    """Saves benchmark results to CSV for visualization."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\nSuccess! Benchmark results saved to {output_path}")

if __name__ == "__main__":
    # Ensure test.jpg points to root directory
    test_image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test.jpg"))
    all_results = []
    
    batch_sizes = [1, 4, 8, 16]
    devices = ["cpu", "cuda"]
    
    for device in devices:
        for b_size in batch_sizes:
            try:
                res = run_batch_benchmark(test_image_path, batch_size=b_size, device=device)
                all_results.append(res)
            except Exception as e:
                print(f"Skipping {device} batch {b_size} due to error: {e}")
                
    if all_results:
        save_results_to_csv(all_results)