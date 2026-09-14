# GPU-Accelerated Real-Time Object Detection & Hardware Benchmarking

## What is This Project Actually Doing?

At its core, this project isn't about building a new object detector—**YOLOv8n** is simply used as a real-world AI workload to study a bigger engineering question:

> *How does running an AI workload on a GPU change its performance compared to running it on a CPU as the workload increases?*

Instead of just asking *"Can AI detect objects?"* (which YOLO already does), this project investigates how computer hardware behaves when you push it harder. By testing both a standard processor (CPU) and an NVIDIA graphics card (GPU via CUDA), we can measure the exact performance trade-offs of GPU acceleration.

---

## The Two Main Parts of the Application

### 1. Normal Object Detection (Interactive Mode)

This part lets you see the AI in action. You feed it an input (like an image or a live webcam feed), choose your execution device (**CPU** or **CUDA GPU**), and watch it detect objects with bounding boxes and confidence scores. It proves that the same AI model can run across different hardware environments.

### 2. Hardware Benchmarking (The Core Experiment)

This is where the real scientific analysis happens. Instead of processing a single image casually, the benchmark tests **different batch sizes** (e.g., processing 1, 4, 8, or 16 images together in a single operation).

By grouping inputs into batches, we give the GPU multiple pieces of data to process simultaneously, exposing it to massive parallelism. For every batch size, the system records critical performance metrics to generate comprehensive reports and graphs.

---

## What Do the Results and Four Graphs Tell Us?

The four generated graphs aren't random charts; together, they tell the complete story of how hardware handles scaling AI workloads:

1. **Latency vs. Batch Size (How long does it take?):**
Measures how long a single inference operation takes. Naturally, asking the model to process 16 images at once takes longer than processing 1 image. However, individual batch latency must always be paired with throughput.
2. **Throughput vs. Batch Size (How much work gets done per second?):**
Calculates images processed per second. Increasing the batch size lets the GPU utilize its many processing units in parallel, drastically increasing throughput—right up until it hits a point of diminishing returns where hardware saturation kicks in.
3. **VRAM vs. Batch Size (What is the memory cost?):**
Tracks how much GPU memory (VRAM) is consumed as the workload grows. Larger batches require significantly more memory. This highlights a classic engineering trade-off: higher throughput and better parallelism cost more VRAM.
4. **GPU Speedup (How much faster is CUDA?):**
Directly compares CPU performance against GPU performance (using the ratio of CPU latency to GPU latency, such as a $5\times$ speedup). This graph answers the ultimate question of how much acceleration CUDA provides for the workload.

*(Note on Performance: When running AI models repeatedly, you might notice initial runs are slightly slower due to one-time startup tasks like model initialization and memory allocation. Benchmarks account for this to measure true, steady-state performance).*

---

## How to Use It

### Step 1: Clone and Set Up the Environment

Open your terminal (Windows PowerShell) and run:

```bash
git clone https://github.com/shaun-tharakan/gpu-accelerated-object-detection2.git
cd gpu-accelerated-object-detection2

# Create and activate your virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

```

### Step 2: Install Dependencies

Install the required Python libraries (including PyTorch, Ultralytics YOLO, OpenCV, and Gradio):

```bash
pip install -r requirements.txt

```

### Step 3: Launch the Application

Start the interactive web interface:

```bash
python demo/app.py

```

Open the local URL displayed in your terminal (usually `[http://127.0.0.1:7860](http://127.0.0.1:7860)`) in your web browser. From there, you can test image/webcam object detection and run the hardware benchmark suite to generate your CSV results and performance figures!
