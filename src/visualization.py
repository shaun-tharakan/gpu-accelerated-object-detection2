import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_benchmark_charts(csv_path="results/raw/benchmark_results.csv", output_dir="results/figures"):
    """Plots Latency, Throughput, Speedup, and VRAM charts from CSV results."""
    if not os.path.exists(csv_path):
        print(f"Error: Path '{csv_path}' not found. Run benchmarks first.")
        return

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    cpu_df = df[df["device"] == "cpu"].sort_values("batch_size")
    cuda_df = df[df["device"] == "cuda"].sort_values("batch_size")

    # 1. Latency vs Batch Size
    plt.figure(figsize=(8, 5))
    if not cpu_df.empty:
        plt.plot(cpu_df["batch_size"], cpu_df["avg_latency_ms"], marker='o', label="CPU")
    if not cuda_df.empty:
        plt.plot(cuda_df["batch_size"], cuda_df["avg_latency_ms"], marker='s', label="CUDA GPU")
    plt.title("Inference Latency vs Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel("Avg Latency (ms)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "latency_vs_batch.png"))
    plt.close()

    # 2. Throughput vs Batch Size
    plt.figure(figsize=(8, 5))
    if not cpu_df.empty:
        plt.plot(cpu_df["batch_size"], cpu_df["throughput_fps"], marker='o', label="CPU")
    if not cuda_df.empty:
        plt.plot(cuda_df["batch_size"], cuda_df["throughput_fps"], marker='s', label="CUDA GPU")
    plt.title("Throughput (FPS) vs Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (Images / sec)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "throughput_vs_batch.png"))
    plt.close()

    # 3. Speedup Multiplier
    if not cpu_df.empty and not cuda_df.empty:
        merged = pd.merge(cpu_df, cuda_df, on="batch_size", suffixes=('_cpu', '_cuda'))
        merged["speedup"] = merged["avg_latency_ms_cpu"] / merged["avg_latency_ms_cuda"]

        plt.figure(figsize=(8, 5))
        plt.plot(merged["batch_size"], merged["speedup"], marker='^', color='green', label="GPU Speedup (x)")
        plt.title("GPU Speedup (CPU Latency / GPU Latency)")
        plt.xlabel("Batch Size")
        plt.ylabel("Speedup Multiplier (x)")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "gpu_speedup.png"))
        plt.close()

    # 4. GPU VRAM Usage
    if not cuda_df.empty and "gpu_memory_mb" in cuda_df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(cuda_df["batch_size"], cuda_df["gpu_memory_mb"], marker='d', color='purple', label="VRAM Allocation")
        plt.title("Peak GPU VRAM Usage vs Batch Size")
        plt.xlabel("Batch Size")
        plt.ylabel("Memory (MB)")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "vram_vs_batch.png"))
        plt.close()

    print(f"Charts saved successfully to '{output_dir}/'")

if __name__ == "__main__":
    generate_benchmark_charts()