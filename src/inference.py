import time
import torch

def run_inference(detector, input_data, device="cpu"):
    """
    Executes model inference on the specified device with proper synchronization and timing.
    
    Args:
        detector: The initialized YOLODetector instance from detection.py.
        input_data: The frame (OpenCV/numpy BGR format) to process.
        device: "cpu" or "cuda".
        
    Returns:
        A tuple containing: (detections, latency_ms).
    """
    # Graceful fallback: check if CUDA is requested but not actually available
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA is not available. Falling back to CPU.")
        device = "cpu"
        
    # Ensure the model is on the correct device before running
    detector.model.to(device)
    
    # --- START TIMING ---
    if device == "cuda":
        # Synchronize before starting the timer to ensure the GPU is ready
        torch.cuda.synchronize() 
        
    start_time = time.perf_counter()
    
    # Run the actual detection using your existing interface
    detections = detector.detect_frame(input_data, device=device)
    
    # --- END TIMING ---
    if device == "cuda":
        # Synchronize before stopping the timer so we don't measure prematurely
        torch.cuda.synchronize()
        
    end_time = time.perf_counter()
    
    # Calculate latency in milliseconds
    latency_ms = (end_time - start_time) * 1000.0
    
    return detections, latency_ms