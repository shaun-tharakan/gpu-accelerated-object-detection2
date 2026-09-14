import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import cv2
import gradio as gr
import torch

from src.detection import YOLODetector
from src.benchmark import run_batch_benchmark, save_results_to_csv
from src.visualization import generate_benchmark_charts

detector = YOLODetector(model_name="yolov8n.pt")

def process_single_image(input_image, device):
    if input_image is None:
        return None, "No image uploaded.", "0.0 FPS"

    start_time = time.perf_counter()
    
    detections = detector.detect_frame(input_image, device=device)
    
    output_image = detector.annotate_frame(input_image, detections)
    
    if torch.cuda.is_available() and device == "cuda":
        torch.cuda.synchronize()
        
    elapsed_time = (time.perf_counter() - start_time) * 1000
    fps = 1000.0 / elapsed_time if elapsed_time > 0 else 0.0

    metrics_text = f"**Device:** {device.upper()}\n\n**Latency:** {elapsed_time:.2f} ms\n\n**Objects Detected:** {len(detections)}"
    fps_text = f"{fps:.1f} FPS"

    return output_image, metrics_text, fps_text

def run_ui_benchmark(image_path):
    if image_path is None:
        return "Please upload an image.", None
    
    all_results = []
    batch_sizes = [1, 4, 8, 16]
    devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    
    for device in devices:
        for b_size in batch_sizes:
            try:
                res = run_batch_benchmark(image_path, batch_size=b_size, device=device, warmup_iters=2, measure_iters=10)
                all_results.append(res)
            except Exception as e:
                print(f"Skipping {device} batch {b_size}: {e}")

    if all_results:
        csv_path = "results/raw/ui_benchmark_results.csv"
        save_results_to_csv(all_results, csv_path)
        generate_benchmark_charts(csv_path=csv_path, output_dir="results/figures")
        
        chart_images = [
            "results/figures/latency_vs_batch.png",
            "results/figures/throughput_vs_batch.png",
        ]
        if "cuda" in devices:
            chart_images.extend([
                "results/figures/gpu_speedup.png",
                "results/figures/vram_vs_batch.png"
            ])
            
        return "✅ Benchmark Complete! Charts generated.", chart_images
    
    return "❌ Benchmark failed.", None

custom_theme = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="indigo",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
)

with gr.Blocks(title="GPU Object Detection", theme=custom_theme) as demo:
    gr.Markdown(
        """
        # 🚀 GPU-Accelerated Real-Time Object Detection
        Test inference speeds and compare hardware scaling directly in the browser!
        """
    )

    with gr.Tabs():
        
        with gr.TabItem("🎯 Live Detection"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_img = gr.Image(type="numpy", label="Upload Input Image")
                    device_selector = gr.Radio(choices=["cpu", "cuda"], value="cpu", label="Target Hardware", interactive=True)
                    run_btn = gr.Button("Run Inference", variant="primary")

                with gr.Column(scale=1):
                    output_img = gr.Image(type="numpy", label="Detection Output")
                    with gr.Row():
                        metrics_box = gr.Markdown(label="Inference Metrics")
                        fps_box = gr.Textbox(label="Throughput", interactive=False)

            run_btn.click(
                fn=process_single_image,
                inputs=[input_img, device_selector],
                outputs=[output_img, metrics_box, fps_box]
            )

        with gr.TabItem("📊 Hardware Benchmarking"):
            gr.Markdown("Upload an image to simulate a video stream and measure latency, throughput, and VRAM across different batch sizes.")
            with gr.Row():
                with gr.Column(scale=1):
                    bench_img = gr.Image(type="filepath", label="Upload Test Image")
                    bench_btn = gr.Button("Run Full Benchmark", variant="primary")
                    status_text = gr.Textbox(label="Status", interactive=False)
                
                with gr.Column(scale=2):
                    gallery = gr.Gallery(label="Performance Charts", show_label=True, columns=2, object_fit="contain")
                    
            bench_btn.click(
                fn=run_ui_benchmark,
                inputs=[bench_img],
                outputs=[status_text, gallery]
            )

if __name__ == "__main__":
    demo.launch()