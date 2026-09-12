import time
import cv2
import gradio as gr
import torch

# Import Partner A's shared interface if available; otherwise use standard YOLO fallback
try:
    from src.detection import detect_frame
except ImportError:
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    
    def detect_frame(frame, device="cpu"):
        results = model(frame, device=device, verbose=False)
        annotated_frame = results[0].plot()
        boxes = results[0].boxes
        detections = [{"box": box.xyxy[0].tolist(), "conf": float(box.conf[0])} for box in boxes]
        return annotated_frame, detections

def process_image(input_image, device):
    if input_image is None:
        return None, "No image uploaded.", "0.0 FPS"

    start_time = time.perf_counter()
    
    # Execute detection on selected device
    output_image, detections = detect_frame(input_image, device=device)
    
    if torch.cuda.is_available() and device == "cuda":
        torch.cuda.synchronize()
        
    elapsed_time = (time.perf_counter() - start_time) * 1000  # ms
    fps = 1000.0 / elapsed_time if elapsed_time > 0 else 0.0

    metrics_text = f"Device: {device.upper()}\nLatency: {elapsed_time:.2f} ms\nObjects Detected: {len(detections)}"
    fps_text = f"{fps:.1f} FPS"

    return output_image, metrics_text, fps_text

with gr.Blocks(title="GPU Object Detection Benchmarking") as demo:
    gr.Markdown("# GPU-Accelerated Real-Time Object Detection")
    gr.Markdown("Compare real-time performance between CPU and CUDA GPU execution.")

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="numpy", label="Upload Input Image")
            device_selector = gr.Radio(choices=["cpu", "cuda"], value="cpu", label="Target Device")
            run_btn = gr.Button("Run Inference", variant="primary")

        with gr.Column():
            output_img = gr.Image(type="numpy", label="Detection Output")
            metrics_box = gr.Textbox(label="Inference Metrics", interactive=False)
            fps_box = gr.Textbox(label="Throughput", interactive=False)

    run_btn.click(
        fn=process_image,
        inputs=[input_img, device_selector],
        outputs=[output_img, metrics_box, fps_box]
    )

if __name__ == "__main__":
    demo.launch()