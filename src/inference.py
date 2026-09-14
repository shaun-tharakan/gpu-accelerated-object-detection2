import time
import torch

def run_inference(detector, input_data, device="cpu"):
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA is not available. Falling back to CPU.")
        device = "cpu"
        
    detector.model.to(device)
    
    if device == "cuda":
        torch.cuda.synchronize() 
        
    start_time = time.perf_counter()
    
    detections = detector.detect_frame(input_data, device=device)
    
    if device == "cuda":
        torch.cuda.synchronize()
        
    end_time = time.perf_counter()
    
    latency_ms = (end_time - start_time) * 1000.0
    
    return detections, latency_ms