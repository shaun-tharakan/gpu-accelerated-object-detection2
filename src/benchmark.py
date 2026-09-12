import cv2
from src.detection import YOLODetector
from src.inference import run_inference

def run_base_benchmark(image_path, device="cpu", warmup_iters=5, measure_iters=50, model_name="yolov8n.pt"):
    """
    Runs a controlled single-image benchmark.
    
    Args:
        image_path: Path to the test image.
        device: "cpu" or "cuda".
        warmup_iters: Number of unmeasured warm-up runs.
        measure_iters: Number of measured runs.
        model_name: The YOLO model to use.
        
    Returns:
        A dictionary containing structured benchmark records.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    image_size = f"{img.shape[1]}x{img.shape[0]}"
    
    # 1. Initialize Model
    print(f"Loading {model_name} on {device}...")
    detector = YOLODetector(model_name=model_name)
    
    # 2. Warm-up Iterations
    print(f"Performing {warmup_iters} warm-up iterations...")
    for _ in range(warmup_iters):
        # We discard the outputs and latency during warm-up
        _ = run_inference(detector, img, device=device)
        
    # 3. Measured Iterations
    print(f"Performing {measure_iters} measured iterations...")
    total_latency_ms = 0.0
    
    for _ in range(measure_iters):
        _, latency_ms = run_inference(detector, img, device=device)
        total_latency_ms += latency_ms
        
    # 4. Calculate Averages
    avg_latency_ms = total_latency_ms / measure_iters
    
    # Convert average latency to seconds to calculate throughput (FPS)
    avg_latency_sec = avg_latency_ms / 1000.0
    throughput_images_per_sec = 1.0 / avg_latency_sec if avg_latency_sec > 0 else 0
    
    # 5. Output Contract Format
    # Note for Partner B: You can extend this dict with batch_size and gpu_memory_mb
    record = {
        "device": device,
        "model": model_name,
        "batch_size": 1,  # Base benchmark is single-image
        "image_size": image_size,
        "warmup_count": warmup_iters,
        "measured_count": measure_iters,
        "avg_latency_ms": round(avg_latency_ms, 2),
        "throughput_images_per_sec": round(throughput_images_per_sec, 2)
    }
    
    return record

# Quick test execution block so you can test it locally before Partner B takes over
if __name__ == "__main__":
    # To test this, make sure you have a sample image like 'test.jpg' in your project root
    # result = run_base_benchmark("test.jpg", device="cpu")
    # print(result)
    pass